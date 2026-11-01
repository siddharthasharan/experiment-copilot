"""Experiment Copilot platform server — stdlib only, Render-ready.

Run locally:  python -m copilot_platform.server --port 8000   (from the repo root)
Render:       Web Service, build `pip install -r requirements.txt`,
              start `python -m copilot_platform.server --port $PORT`.
Free tier: no database, no paid service. Sessions live in server
memory and are lost on restart — the UI says so. That is a deliberate
Phase 1 tradeoff, not a hidden limitation: persistence is a later
step, and it must not change the engine.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from copilot_platform.chat import Orchestrator, new_session, welcome_blocks  # noqa: E402
from copilot_platform.engine import FLAGSHIPS, run_flagship  # noqa: E402
from copilot_platform.engine_extra import EXTRA_FLAGSHIPS as _EXTRA_FLAGSHIPS  # noqa: E402

FLAGSHIPS.update(_EXTRA_FLAGSHIPS)
from copilot_platform.scorecard import build_scorecard, example_scorecard  # noqa: E402
from copilot_platform.scenarios import FLAGSHIP_CONFIGS  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_lock = threading.Lock()
_sessions = {}
_saved = []
_orchestrator = None


def _load_examples():
    for cand in [os.path.join(ROOT, "docs", "scorecard_data.json"),
                 os.path.join(ROOT, "showroom", "scorecard_data.json")]:
        if os.path.exists(cand):
            with open(cand) as f:
                data = json.load(f)
            out = []
            for s in data["scenarios"]:
                out.append({"id": s["id"], "title": s["title"], "verdict": s["verdict"],
                            "description": s["description"], "n_users": s["n_users"],
                            "metrics": s["metrics"], "flags": s.get("flags", []),
                            "trap_reveal": s.get("trap_reveal", {}),
                            "expected_verdict_raw": s.get("expected_verdict_raw", "")})
            return out
    return []


def _get_session(sid):
    with _lock:
        s = _sessions.get(sid)
        if s is None:
            s = new_session()
            _sessions[s["id"]] = s
            s["messages"] = welcome_blocks()
        return s


def _summary_blocks(session):
    cfg, result = session["config"], session["result"]
    p = result["primary"]
    label = {"LAUNCH": "Launch", "DONT_LAUNCH": "Do Not Launch",
             "INCONCLUSIVE": "Inconclusive", "INVALID": "Invalid Experiment"}[result["verdict"]]
    text = (f"Your experiment is complete.\n\n"
            f"{cfg['name']}: the primary metric, {cfg['primary']}, moved {p['lift_pct']:+.2f}% "
            f"(95% CI [{p['ci_low_pct']:+.2f}%, {p['ci_high_pct']:+.2f}%], p={p['p_value']:.3g}).\n\n"
            f"Recommendation: {label}. I've opened the scorecard with the statistical results, "
            f"business impact, and guardrails — and the conversation stays here, so you can ask "
            f"follow-ups about this exact run.")
    return [{"type": "text", "text": text},
            {"type": "chips", "items": cfg["followups"]},
            {"type": "scorecard_open", "scorecard": session["scorecard"]}]


def _run_thread(session):
    sid = session["scenario_id"]
    cfg = session["config"]
    seed = cfg.get("seed", 42)
    stages = []

    def progress(name, detail):
        with _lock:
            stages.append({"stage": name, "detail": detail, "at": time.time()})

    try:
        import re as _re
        _m = _re.search(r"\d+", str(cfg.get("duration", "")))
        _days = int(_m.group(0)) if _m else None
        df, result = run_flagship(sid, seed=seed, progress=progress, duration_days=_days)
        trend = result.pop("trend")
        sc = build_scorecard(cfg, result, trend, seed)
        with _lock:
            session["df"] = df
            session["result"] = result
            session["trend"] = trend
            session["scorecard"] = sc
            session["state"] = "done"
            session["run"] = {"done": True, "stages": stages, "error": None}
            session["messages"].extend(_summary_blocks(session))
            _saved.append({"id": session["id"], "name": cfg["name"],
                           "industry": cfg["industry"], "verdict": result["verdict"],
                           "primary_lift_pct": result["primary"]["lift_pct"],
                           "at": time.time()})
    except Exception as e:  # surfaced honestly in the run panel
        with _lock:
            session["run"] = {"done": True, "stages": stages, "error": str(e)}


def _pdf_clean(text):
    """ASCII-only text for the standard Helvetica fonts.

    The old PDF wrote UTF-8 bytes (em dashes, etc.) into a WinAnsi font,
    which rendered as mojibake ("a" after "Copilot") and overlapped lines.
    """
    repl = {"\u2014": "-", "\u2013": "-", "\u00b7": "-", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u00b1": "+/-", "\u2264": "<=",
            "\u2265": ">=", "\u00d7": "x"}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("ascii", "ignore").decode("ascii")


def _pdf_wrap(text, width=92):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def _executive_pdf(sc):
    """Executive scorecard PDF: dashboard sections + clear recommendation.

    Hand-rolled, stdlib only, multi-page, no overlapping lines: every
    line gets its own y position with proper spacing, headings are bold,
    and the recommendation is boxed and large. Sections mirror the web
    scorecard (decision, KPIs, metrics, PM view, watch-outs) so the PDF
    is a dashboard on paper, not a dump of raw lines.
    """
    import datetime
    items = []  # (kind, text) — kind: title, name, rec, h2, body, bullet
    items.append(("title", "Experiment Copilot - Executive Scorecard"))
    items.append(("name", sc["experiment_name"]))
    items.append(("rec", "RECOMMENDATION: " + sc["recommendation"]))
    if sc.get("business_reason"):
        items.extend(("body", l) for l in _pdf_wrap(sc["business_reason"]))
    items.append(("h2", "Decision at a glance"))
    dec = [f"Primary metric: {sc['primary_metric']}"]
    if sc["primary_lift_pct"] is not None:
        dec.append(f"Lift: {sc['primary_lift_pct']:+.2f}%  |  95% CI: [{sc['ci_low_pct']:+.2f}%, {sc['ci_high_pct']:+.2f}%]  |  p = {sc['p_value']:.3g}")
    dec.append(f"Guardrails: {sc['guardrail_status']}  |  SRM: {sc['srm_status']}  |  Seed: {sc['seed']}")
    if sc.get("precomputed"):
        dec.append("Precomputed demonstration (1M users, seed 42) - not a live run.")
    items.extend(("body", l) for d in dec for l in _pdf_wrap(d))
    items.append(("h2", "Key metrics"))
    for k in sc.get("kpis", []):
        items.extend(("body", l) for l in _pdf_wrap(f"{k['label']}: {k['value']}"))
    items.append(("h2", "Metrics table"))
    for m in sc.get("metrics", []):
        lift = m.get("lift_pct")
        row = f"{m.get('name', 'metric')}: lift {lift:+.2f}%" if lift is not None else f"{m.get('name', 'metric')}"
        if m.get("p_value") is not None:
            row += f", p = {m['p_value']:.3g}"
        if m.get("guardrail"):
            row += " (guardrail)"
        items.extend(("body", l) for l in _pdf_wrap(row))
    pm = sc.get("pm", {})
    items.append(("h2", "PM view"))
    for label, key in [("What happened?", "what"), ("Why does it matter?", "why"),
                       ("Business impact", "impact"), ("Should we launch?", "launch"),
                       ("What should we do next?", "next")]:
        if pm.get(key):
            items.append(("body", label))
            items.extend(("body", l) for l in _pdf_wrap(pm[key]))
    if sc.get("watchouts"):
        items.append(("h2", "Watch-outs and limitations"))
        for w in sc["watchouts"]:
            items.extend(("bullet", l) for l in _pdf_wrap(w))
    items.append(("h2", "About this scorecard"))
    items.extend(("body", l) for l in _pdf_wrap(
        f"Synthetic demo data - not real user data. Generated: "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Platform version: platform-1.0 | Experiment: {sc['experiment_name']}"))
    # Paginate: (font, size, height) per kind; new page when y would drop below 50.
    spec = {"title": ("F2", 16, 26), "name": ("F2", 13, 22), "rec": ("F2", 17, 34),
            "h2": ("F2", 11, 20), "body": ("F1", 10, 14), "bullet": ("F1", 10, 14)}
    pages, cur, y = [], [], 760
    for kind, text in items:
        font, size, h = spec[kind]
        if y - h < 50:
            pages.append(cur)
            cur, y = [], 760
        if kind == "bullet":
            text = "- " + text
        cur.append((kind, font, size, y, _pdf_clean(text)))
        y -= h
    if cur:
        pages.append(cur)
    # Build objects: 1 catalog, 2 pages root, then per page (page, content),
    # then fonts. Page ids and content ids are computed, not guessed.
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    content_ids = [4 + 2 * i for i in range(n)]
    font_reg, font_bold = 3 + 2 * n, 4 + 2 * n
    objs = {}
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    for i, page in enumerate(pages):
        parts = []
        for kind, font, size, line_y, text in page:
            if kind == "rec":
                parts.append(f"0.42 0.16 0.85 RG 1 w 50 {line_y - 8} 512 28 re S")
            safe = text.replace("\\", "").replace("(", "[").replace(")", "]")
            parts.append(f"BT /{font} {size} Tf 50 {line_y} Td ({safe}) Tj ET")
        body = "\n".join(parts).encode("latin-1", "ignore")
        objs[page_ids[i]] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                             f"/Contents {content_ids[i]} 0 R /Resources << /Font << /F1 {font_reg} 0 R "
                             f"/F2 {font_bold} 0 R >> >> >>").encode()
        objs[content_ids[i]] = (f"<< /Length {len(body)} >>\nstream\n".encode() + body + b"\nendstream")
    objs[font_reg] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objs[font_bold] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i in range(1, max(objs) + 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + objs[i] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {max(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {max(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep logs quiet
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body, ctype, filename=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if filename:
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self._send_bytes(f.read(), "text/html")
        if path.startswith("/static/"):
            name = os.path.basename(path)
            fpath = os.path.join(STATIC, name)
            if os.path.exists(fpath) and name.endswith((".html", ".css", ".js")):
                ctype = {"html": "text/html", "css": "text/css", "js": "application/javascript"}[name.rsplit(".", 1)[-1]]
                with open(fpath, "rb") as f:
                    return self._send_bytes(f.read(), ctype)
            return self._send_json({"error": "not found"}, 404)
        if path == "/api/examples":
            return self._send_json({"examples": _orchestrator.examples})
        if path == "/api/saved":
            with _lock:
                return self._send_json({"saved": list(_saved)})
        if path == "/api/scorecard":
            s = _sessions.get(qs.get("session_id", [""])[0])
            if s and s.get("scorecard"):
                return self._send_json({"scorecard": s["scorecard"]})
            return self._send_json({"error": "no scorecard yet"}, 404)
        if path == "/api/run-status":
            s = _sessions.get(qs.get("session_id", [""])[0])
            if not s or not s.get("run"):
                return self._send_json({"started": False})
            run = s["run"]
            out = {"started": True, "done": run["done"], "stages": run["stages"], "error": run["error"]}
            if run["done"] and not run["error"]:
                out["summary"] = _summary_blocks(s)
                out["scorecard"] = s["scorecard"]
            return self._send_json(out)
        if path == "/api/tables":
            s = _sessions.get(qs.get("session_id", [""])[0])
            if not s or s.get("df") is None:
                return self._send_json({"error": "run an experiment first"}, 404)
            df = s["df"]
            tables = [{"name": "experiment_units", "rows": int(len(df)),
                       "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                       "sample": df.head(20).to_dict("records")}]
            if s.get("trend") is not None:
                import pandas as pd
                tr = pd.DataFrame(s["trend"])
                tables.append({"name": "daily_trend", "rows": int(len(tr)),
                               "columns": [{"name": c, "dtype": str(tr[c].dtype)} for c in tr.columns],
                               "sample": tr.head(20).to_dict("records")})
            return self._send_json({"tables": tables})
        if path == "/api/download":
            s = _sessions.get(qs.get("session_id", [""])[0])
            typ = qs.get("type", ["scorecard_json"])[0]
            if not s or not s.get("scorecard"):
                return self._send_json({"error": "no scorecard yet"}, 404)
            sc = s["scorecard"]
            if typ == "scorecard_json":
                return self._send_bytes(json.dumps(sc, indent=2).encode(), "application/json", "scorecard.json")
            if typ == "config_json":
                return self._send_bytes(json.dumps(s["config"], indent=2).encode(), "application/json", "experiment-config.json")
            if typ == "metrics_csv":
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["metric", "control", "treatment", "lift_pct", "ci_low_pct", "ci_high_pct", "p_value", "method"])
                for m in sc["metrics"]:
                    w.writerow([m["name"], m["control"], m["treatment"], m["lift_pct"],
                                m["ci_low_pct"], m["ci_high_pct"], m["p_value"], m.get("method", "")])
                return self._send_bytes(buf.getvalue().encode(), "text/csv", "metrics.csv")
            if typ == "pdf":
                return self._send_bytes(_executive_pdf(sc), "application/pdf", "scorecard.pdf")
            return self._send_json({"error": "unknown download type"}, 400)
        return self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_json()
        if parsed.path == "/api/chat":
            sid = body.get("session_id")
            s = _get_session(sid) if sid else _get_session(None)
            if sid is None:
                # First call: return the welcome blocks already stored.
                return self._send_json({"session_id": s["id"], "blocks": s["messages"],
                                        "state": s["state"]})
            blocks = _orchestrator.handle(s, message=body.get("message", ""),
                                          action=body.get("action"),
                                          payload=body.get("payload"))
            with _lock:
                s["messages"].extend(blocks)
            out = {"session_id": s["id"], "blocks": blocks, "state": s["state"]}
            # Example scorecard blocks carry the panel payload directly.
            for b in blocks:
                if b["type"] == "example_scorecard":
                    from copilot_platform.scorecard import example_scorecard as _esc
                    out["scorecard"] = _esc(b["example"])
                    s["scorecard"] = out["scorecard"]
            return self._send_json(out)
        if parsed.path == "/api/run":
            s = _sessions.get(body.get("session_id", ""))
            if not s or s.get("scenario_id") not in FLAGSHIPS:
                return self._send_json({"error": "no runnable flagship selected"}, 400)
            with _lock:
                s["run"] = {"done": False, "stages": [], "error": None}
                s["state"] = "running"
            threading.Thread(target=_run_thread, args=(s,), daemon=True).start()
            return self._send_json({"started": True})
        if parsed.path == "/api/saved-load":
            with _lock:
                item = next((x for x in _saved if x["id"] == body.get("id")), None)
            if not item:
                return self._send_json({"error": "not found"}, 404)
            s = _sessions.get(item["id"])
            if not s or not s.get("scorecard"):
                return self._send_json({"error": "saved run no longer in server memory"}, 410)
            return self._send_json({"session_id": s["id"], "blocks": s["messages"],
                                    "state": s["state"], "scorecard": s["scorecard"]})
        return self._send_json({"error": "not found"}, 404)


def main():
    global _orchestrator
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()
    _orchestrator = Orchestrator(_load_examples())
    print(f"Experiment Copilot platform on http://localhost:{args.port} "
          f"({len(_orchestrator.examples)} example scenarios loaded)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
