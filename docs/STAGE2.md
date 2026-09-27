# Stage 2 — the working lesson

Stage 2 connects the original NumPy engine to a FastAPI server and React + TypeScript
website. It implements the temperature and learning experiments with actual model
outputs. Recharts supplies loss graphs, Lucide supplies icons, and self-hosted
Fontsource fonts keep the built app independent of external font services.

## Open the app

From `GlassworkV1`:

```bash
.venv/bin/python scripts/serve.py
```

Open **http://127.0.0.1:8000**. This serves the compiled frontend and API on the same
origin. If it says the address is already in use, the existing preview may still
be running. The preview binds only to localhost.

For a fresh checkout or after frontend changes:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-web.txt
cd web
npm ci
npm run build
cd ..
.venv/bin/python scripts/serve.py
```

The existing workspace environment inherits the machine's NumPy installation from
stage 1. A fresh virtual environment as above is fully isolated. Unrelated system
Python packages are not part of this application's dependency set.

For frontend development, run the API on port 8000 and `npm run dev` in `web/`.
Vite forwards `/api` to the backend. Frontend changes then refresh automatically.
The NumPy checkpoint must already exist; `scripts/train.py` creates it.

## What you can do

1. Enter a known lowercase prompt, up to 32 characters, and inspect it.
2. Change temperature (including greedy selection at zero) and see actual next-token
   probabilities and entropy. Model weights do not change.
3. Generate 64 characters with a reproducible sampling seed.
4. Answer a short conceptual question and receive authored feedback.
5. Switch to the learning lab, reset to random weights, and train one or 25 steps.
6. Watch the probability comparison and separate train/held-out loss measurements.
7. Stop between updates, or restore the exact original trained checkpoint.
8. Inspect the real attention matrix for either head of either transformer block.

The small architectural illustration is labeled **conceptual**. Its next-character
prediction is live. It is not an interactive 3D tensor view; that is stage 3.
The local lesson explanations are authored. Gemini integration is stage 4.

## Integration guarantees and boundaries

- Each session has its own model, optimizer and RNG. Changes never write over the
  trained checkpoint on disk or another learner's model.
- FastAPI synchronous routes execute in its thread pool. A per-session lock
  serializes NumPy inference, reset and training. The event loop does not run
  the numerical training directly.
- Mutation requests carry the revision shown to the learner. A stale or simultaneous
  duplicate mutation receives HTTP 409 instead of silently applying twice.
- Training commits one step per browser request. Stop finishes the current request
  and prevents the next. Server endpoints allow at most five steps per request.
- Every successful step returns complete training and held-out loss, step count and
  a SHA-256 fingerprint of the current parameter arrays. These are measured data.
- The frontend ignores older inspection responses when controls change rapidly.
  Before/after comparisons use the same prompt and temperature.
- Temperature, prompt length, generation length and update counts are validated.
  Unknown characters fail clearly without training the model.
- The app uses one origin, no broad CORS, no external inference provider and no API key.
- Sessions are in-memory and expire after one hour without requests. Capacity is 16
  sessions; history retains the latest 501 points. Restarting the server clears sessions.
- Use **one Uvicorn worker**. This local demo does not have distributed session storage,
  persistent user accounts or public deployment protection. Session IDs are random
  capability tokens kept in sessionStorage; do not publish them.
- Browser errors offer a new-session recovery action. A lost response to a training
  request may still have committed its current step; revision checking prevents
  silently applying that same stale request again.

## API

| Method and route | Purpose |
| --- | --- |
| `GET /api/health` | Server and checkpoint availability |
| `POST /api/sessions` | Independent copy of trained checkpoint |
| `GET /api/sessions/{id}` | Model state, fingerprint and loss history |
| `DELETE /api/sessions/{id}` | Release session |
| `POST .../inspect` | Prompt, logits, probabilities, entropy and real trace |
| `POST .../sample` | Seeded autoregressive generation |
| `POST .../train` | Bounded updates with before/after evidence |
| `POST .../reset` | Random initialization or restored checkpoint |

Interactive API documentation: `/docs`. The educational model math remains in
`glasswork/`; HTTP, session and validation responsibilities are in `server/app.py`.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
cd web && npm run build
```

The engine/API suite has **26 tests**, including direct comparisons between API
results and NumPy outputs, training isolation, concurrent stale updates, reset,
session expiry, capacity bounds and validation before mutation.

A real Chromium test exercises the user flow:

```bash
# In a Python environment with Playwright installed and Chromium available:
python scripts/test_browser.py
```

The development machine provides that environment at
`/home/sawoo/playwright-env/bin/python`. A fresh machine can install the Python
`playwright` package and run `python -m playwright install chromium`.

Browser results and screenshots are saved under `artifacts/stage2-*`. The browser
checks generation, temperature, training, stop, reset, inputs, attention, dialog
keyboard behavior and mobile overflow, with external requests blocked.

## Still ahead

Stage 3: interactable 3D model modules driven by the real trace. Stage 4: Gemini
feedback grounded in recorded experiment results. Stage 5: domain and deployment.
The current tiny corpus is not a benchmark for general language ability, and the
attention matrix is not a complete explanation of the model's reasoning.

### Recorded result — 2026-09-26

- **26/26** Python engine and API tests passed (`artifacts/stage2-test-results.txt`).
- TypeScript checking and the production Vite build passed (`artifacts/stage2-build.txt`).
- **18 browser checks** passed in Chromium, with zero uncaught page errors
  (`artifacts/stage2-browser-results.json`).
- Desktop (1440px) and mobile (390px) layouts were exercised. Screenshots:
  `stage2-desktop.png`, `stage2-learning.png`, `stage2-mobile.png` in `artifacts/`.
- Requests outside the local app were blocked during the final browser run.
- The initial JavaScript bundle is approximately **79 KB gzip**; the Recharts
  learning view is loaded separately when selected.

This verification covers the local single-process application, not a public
multi-user deployment or all browsers. No Gemini calls or 3D renderer are included.
