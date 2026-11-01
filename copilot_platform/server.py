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


def _pdf_bytes(title, lines):
    """Minimal hand-rolled PDF (text only, stdlib). No dependency, no cost."""
    content = []
    y = 750
    content.append("BT /F1 16 Tf 50 780 Td (Experiment Copilot — Executive Scorecard) Tj ET")
    content.append(f"BT /F1 12 Tf 50 755 Td ({title}) Tj ET")
    for line in lines:
        safe = line.replace("(", "[").replace(")", "]").replace("\\", "")[:110]
        content.append(f"BT /F1 10 Tf 50 {y} Td ({safe}) Tj ET")
        y -= 14
        if y < 40:
            break
    body = "\n".join(content)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(body)} >>\nstream\n{body}\nendstream".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
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
                lines = [f"Experiment: {sc['experiment_name']}", f"Recommendation: {sc['recommendation']}",
                         f"Primary metric: {sc['primary_metric']}",
                         f"Lift: {sc['primary_lift_pct']:+.2f}%  CI [{sc['ci_low_pct']:+.2f}%, {sc['ci_high_pct']:+.2f}%]  p={sc['p_value']:.3g}",
                         f"Guardrails: {sc['guardrail_status']}  SRM: {sc['srm_status']}  Seed: {sc['seed']}"]
                for m in sc["metrics"]:
                    lines.append(f"{m['name']}: {m['lift_pct']:+.2f}% p={m['p_value']:.3g}")
                return self._send_bytes(_pdf_bytes(sc["experiment_name"], lines), "application/pdf", "scorecard.pdf")
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
