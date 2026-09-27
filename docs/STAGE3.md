# Stage 3 — inside a prediction

Glasswork now centers the lesson on an interactive 3D view of the team's real
NumPy attention model. One inspector explains the selected operation; sampling
and training controls stay in an experiment drawer until needed.

## Explore it

Run from `GlassworkV1`:

```bash
.venv/bin/python scripts/serve.py
```

Open http://127.0.0.1:8001. If the server was already running before the upgrade,
restart it to load trace schema 4. After frontend changes, run `npm run build`
inside `frontend/` before refreshing.

1. Start with `the cat`. Select a stage in the scene or the timeline.
2. Follow a character with the inspector's selector. Hover over tiles for values.
3. Drag to orbit, scroll to zoom, or focus on the selected stage. Reset restores
   the overview. Play walks through the six operations with explanatory flow
   particles; this animation is not a measurement of execution time.
4. Open **Experiment** to change temperature and generate repeatable text.
5. Open **Learning lab**, choose random weights, and train. Amber marks actual
   output-weight changes. The chart shows measured training and held-out loss.

The six stages are character embeddings, position addition, query/key/value
projections, causal attention, output projection, and next-character probabilities.
The right inspector shows the corresponding equations and live numerical values.

## Architecture

```text
App.tsx — stage selection, walkthrough, inspector and experiment drawer
  ├─ useLab.ts — session lifecycle, trace requests, sampling and training
  ├─ ExplorerScene.tsx — Three.js / React Three Fiber / Drei scene
  ├─ stages.ts — operation descriptions and mathematical notation
  └─ LossChart.tsx — Recharts loss curves, loaded when requested
                        │ existing HTTP API
server/team_adapter.py — schema-4 inspection trace
                        │ captures the same forward-pass arrays
GlassBoxModel — Python + NumPy, handwritten attention and output gradients
```

The trace adds token and position embeddings, Q/K/V matrices, and a copy of the
output weights. The existing attention, hidden values, logits and probabilities
remain available. Capturing a trace does not change predictions or model weights.
Sampling and training use the same session model as the scene. Requests carry
revision checks; each learner gets independent weights. Superseded inspection
responses cannot replace a newer prompt's trace.

## Visual conventions and scope

- Matrix tiles encode numerical magnitude; blue indicates negative values.
  Matrix depths are capped at magnitude 1.5 for readability. Exact values remain
  available in the inspector and on hover.
- Attention displays the causal mask. Probability columns share a true 0–100%
  scale; zero probability has no visible column.
- Up to 16 positions around the selected character are drawn to bound scene
  geometry. The inspector retains the full attention row, up to 128 positions.
- The transparent enclosures are an educational layout, not a physical picture
  of neurons or a frontier model's architecture.
- The active engine has 640 parameters, but only its 40 output-projection weights
  learn. Embeddings and attention parameters stay fixed. Its tiny corpus and
  ten-character vocabulary do not produce fluent general-purpose text.
- Gemini explanations, deployment/domain setup, and a frontier-model comparison
  are not implemented in this stage.

The diagram view provides the same stage navigation and numerical inspector
without WebGL. A lost rendering context switches to that view. Reduced motion
removes flow particles and respects the operating-system preference. Controls
also work through the keyboard-accessible timeline. Fonts and scene assets are
local; no external services are required to use the lesson.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
cd frontend
npm run build
cd ..
# In a Python environment with Playwright and Chromium installed:
python scripts/test_browser.py  # run from GlassworkV1 with the server running
```

The engine tests reconstruct E+P, Q/K/V, masked attention, attention output,
logits and probabilities from the schema-4 trace, and check that tracing is
read-only. Browser checks exercise real WebGL, camera controls, stage navigation,
walkthrough playback, temperature, generation, training, resets, prompt validation,
long contexts, the diagram fallback, and mobile overflow. Results and screenshots
are written under `artifacts/stage3-*`. The old stage-2 browser script is retained
as `scripts/test_browser_v2.py` for historical reference, not the current UI.
