# main2 integration

This branch was created from local `frontend` commit `2c9253e`, in a separate
worktree, preserving the original checkout and its subsequent work. No history was
rewritten. The existing remote is `https://github.com/Ghaz-555/Hackathon`.

The later-provided backup at `/home/sawoo/Desktop/New Folder/wokring stuff`
was also reviewed. Its original complete embedding scene, styling, browser tests
and historical documentation were recovered into this integration.

## Integrated changes

- Copied the verified GlassworkV1 engine package, server, tests, training scripts,
  model checkpoint, documentation and verification artifacts into this repository.
- Kept the original `backend/` Python source intact. The running server imports
  the adapted `glasswork/team_engine/` package.
- Consolidated the web implementation under `frontend/`, with two HTML entry
  points and shared locked dependencies. Added navigation in both directions.
- Preserved the branch's embedding page logic in `frontend/src/brain/main.ts`,
  supplied the missing math/scene/style modules and completed the empty build files.
- Regenerated the deleted embedding vectors. The older binary recovered from Git
  did not match the newer manifest, so it was not used as if it did. The exporter
  now records the exact selected token IDs for reproducible regeneration.
- Served both entry points, JS/CSS assets and embeddings from the same FastAPI
  application. The default review port is 8001 to avoid the older local app on 8000.
- Removed tracked generated Python bytecode from this branch and added ignore rules
  for caches, dependencies and compiled frontend output.

## Architecture and limitations

The model UI uses session-isolated live NumPy calculations. GPT-2 exploration uses
static pretrained vectors and client-side cosine search. These two datasets are
intentionally distinct and clearly named. Nothing in the GPT-2 explorer changes
the small model's learned weights.

The team model has one attention head and only trains its 40 output-projection
weights. GPT-2 search is over the included 2,048-token subset. PCA is lossy; vector
arithmetic and rankings use all 768 dimensions. Gemini integration, public hosting
and domain configuration are not implemented in this branch.

## Verification

Run the commands in the root README. Current integration results are written to
`artifacts/main2-*`. Existing `stage2-*`, `team-integration-*` and `stage3-*` files
were imported as historical evidence; they do not substitute for testing main2.

The branch is intended for review before merging into `main`. No merge into main
is part of this integration.

## Animation and tensor architecture revision

The model explorer was redesigned after reviewing [Brendan Bycroft's LLM
visualization](https://bbycroft.net/llm). This is an independent implementation
using the team's actual single-head architecture. It retains Glasswork's dark
glass presentation, with connected tensor planes, separate Q/K/V weight branches,
a causal attention grid, final hidden state, logits and probability columns.

Trace schema 4 adds copied Q/K/V projection weights. Reconstruction tests verify
their products against the actual forward pass. Scanning rows and moving signals
are explanatory animations, enabled by default. The walkthrough changes stages
and smoothly moves the camera; reduced motion freezes the effects. Training
animates depth changes and highlights only genuinely changed output weights.

The CPU math is unchanged by these rendering effects. Long contexts use a bounded
16-position window in the scene, while the inspector retains full attention rows.

## Final verification result

- 36 Python engine/API tests passed in this checkout's isolated environment.
- Eight embedding numerical tests passed.
- 13 model-explorer browser scenarios and 12 combined-app browser scenarios passed.
- The animation test compared rendered frames and confirmed that reduced motion
  freezes the scene. Both browser reports contain zero uncaught page errors.
- TypeScript checking and the production build passed. Vite reports the expected
  large shared Three.js chunk; it is not a build failure.
- All nine original backend source hashes still match the imported source record.

Evidence: `artifacts/main2-python-tests.txt`, `main2-math-tests.txt`,
`main2-model-browser-results.json`, `main2-integrated-browser.json`, and
`main2-build.txt`.

## Publication history

The first integration push was rejected because an inherited commit contained a 147 MB downloaded model cache. The unpublished `main2` history was rebuilt to exclude `.cache/` files, preserving commit messages and attribution. The final application tree was verified identical before adding this note and the cache ignore rule. Original local `main` and `frontend` branches were left unchanged. Downloaded caches are regenerated locally and must not be committed.
