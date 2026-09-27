# Glasswork

Two connected views of language-model math:

- **Model explorer** (`/`): follow a character through the team's NumPy model in
  3D. Inspect real embeddings, Q/K/V, causal attention, output weights and next
  character probabilities. Train the output layer, compare losses, change
  temperature and generate text.
- **The brain of an LLM** (`/brain.html`): explore 2,048 real GPT-2 input embeddings
  in a 3D PCA projection. Search tokens, slice the scene, and try vector arithmetic.
  Rankings use all 768 original dimensions, not the projected coordinates.

## Run the integrated app

Requires Python 3.12+ and Node 22.12+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-web.txt
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python scripts/serve.py
```

Open **http://127.0.0.1:8001**. Both explorers and the API are served from the same
origin. No API keys or model downloads are required: the tiny trained checkpoint
and GPT-2 embedding subset are included. Use `--port 8002` if needed.

For frontend development, keep the API running on 8001 and run `npm run dev`
inside `frontend/`; Vite proxies `/api` to that server.

## Repository layout

```text
frontend/                  One Vite build; React model view + Three.js embedding view
  src/brain/               Embedding math, scene and page controls
  public/embeddings/       Pinned GPT-2 subset, manifest and original license
server/                    FastAPI sessions and team-engine adapter
backend/                   Original team source, preserved
glasswork/team_engine/    Adapted copy used by the running application
glasswork/                Earlier NumPy transformer, retained for regression tests
scripts/                   Server, training and browser checks
tests/                     Engine parity, gradients, checkpoints, API and trace tests
artifacts/                 Checkpoints and verification evidence
docs/                      Architecture and integration notes
```

## What actually learns

The active team model has **640 parameters; only its 40 output weights train**.
Its embeddings and single attention head stay fixed. It uses four-dimensional
vectors and a ten-character vocabulary. This is an educational model, not a
fluent chatbot. The separately visualized GPT-2 embeddings are pretrained and
read-only; they are not weights trained by the small-model lesson.

The 3D displays are educational projections/layouts. Their values are real, but
spatial proximity in PCA can differ from cosine similarity in the original
embedding space. Motion illustrates the operations rather than execution timing.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
cd frontend
npm test
npm run build
cd ..
```

With the integrated server running, use a Python environment containing Playwright
and Chromium to run `scripts/test_browser.py` and `scripts/test_integrated_browser.py`.

The embedding exporter is `frontend/scripts/prepare_embeddings.py`. It uses NumPy,
a pinned model revision and fixed token IDs; regeneration needs network access and
caches the larger source matrix under the ignored `frontend/.cache/` directory.

See [the main2 integration record](docs/MAIN2_INTEGRATION.md) and
[stage 3 notes](docs/STAGE3.md). Earlier stage documents and screenshots are kept
as historical records; this README describes the current combined application.

## Attribution

The original team engine remains under its MIT license. The integration, additional
validation, visualizations and UI were developed with AI coding assistance. GPT-2's
source revision and original license accompany its exported embeddings.
