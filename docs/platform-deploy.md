# Deploy the chat-first platform on Render (Phase 1)

This is a **separate service** from the existing static showroom
(`experiment-copilot-vzzk`). Do not touch that service. The platform is
a Python Web Service; Render Blueprints do not support static sites,
and this one is not static, so create it manually:

1. Render dashboard → **New → Web Service** → select repo
   `siddharthasharan/experiment-copilot`, branch `main`.
2. Settings:
   - Name: `experiment-copilot-platform` (Render may suffix it if taken —
     use the URL Render shows, never guess one).
   - Region: Oregon (same as the other services).
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -m copilot_platform.server --port $PORT`
   - Instance type: **Free**.
3. Create the service. First deploy takes a few minutes (numpy/pandas/
   scipy install). The log should end with a line like
   `Experiment Copilot platform on http://localhost:<port> (14 example scenarios loaded)`.
4. Open the URL Render assigns. Free Web Services sleep after ~15
   minutes idle and take ~50 seconds to wake — the first load after
   idle will be slow, then fast. That is the free-tier tradeoff, the
   same as SocialPulse.
5. Verify in order:
   - The welcome message and three industry cards appear.
   - Marketplace → DoorDash ZIP promotion → Run → the progress card
     lists real stages and the scorecard opens beside the chat with
     "Do Not Launch" and the contribution reason.
   - Chat follow-up: "Exclude the five largest markets" → a recomputed
     lift, different from the headline.
   - Example Scenarios → pick "Whale variance" → its precomputed
     scorecard opens, labelled precomputed.

## If the URL 404s or the deploy fails

- Check the deploy log first: a failed `pip install` is the usual cause.
- Do **not** reuse the static site's `render.yaml` for this service,
  and do not add a Blueprint for it — Blueprints only support web /
  private / worker / cron services, and this service is created
  manually precisely so its start command is explicit.
- The repo root `render.yaml` (if present in your local copy) belongs
  to the old static-site attempt and is not used by this service.

## Cost

$0 on Render free tier, plus $0 for the code (stdlib server, no paid
APIs, no LLM). The only cost is the free instance sleeping when idle.
