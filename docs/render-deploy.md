# Deploying the showroom on Render (static site)

## What gets deployed

`docs/index.html` — the self-contained showroom page, built locally by
`scripts/build_showroom.py` from the 1M-user data and committed to the repo.
Render serves that file as-is. It does not run Python, does not generate
data, and does not need the gitignored `eval_data_v2*` folders.

## Steps

1. Sync `render.yaml` (repo root) and the updated `docs/` to GitHub and push
   to `main`.
2. In Render: **New → Blueprint**, select the `experiment-copilot` repo,
   click **Apply**. Render creates a Static Site named `experiment-copilot`.
3. Wait for the deploy status to go **Live**.
4. Open https://experiment-copilot.onrender.com and check:
   - The showroom loads with 14 scenarios in the sidebar.
   - The first scenario (`clean_win`) shows verdict LAUNCH.
   - `novelty_decay_trap` shows INCONCLUSIVE and `whale_variance_trap`
     shows INCONCLUSIVE — these are the two cases the 1M recalibration
     fixed; if either shows a different verdict, the deployed `docs/`
     is stale and needs a fresh push.

## Notes

- **Every push to `main` redeploys Render automatically.** A push that only
  changes code or the README still redeploys the same `docs/index.html` —
  harmless, just a new deploy.
- **To update the showroom itself:** regenerate the data locally, run
  `scripts/build_showroom.py`, verify 14/14 verdicts, then commit the new
  `docs/index.html` + `docs/scorecard_data.json` and push.
- **Free tier:** static sites do not sleep, so unlike a free Web Service
  there is no ~50-second wake delay on first load.
- **GitHub Pages stays the primary host.** Render is a second URL with
  identical content, useful where `github.io` is blocked or slow.
