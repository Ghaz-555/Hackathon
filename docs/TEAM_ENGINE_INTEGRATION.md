# Team engine integrated into stage 2

The live Glasswork app now uses the team's `GlassBoxModel` from
`/home/sawoo/Downloads/Hackathon/backend` through the existing FastAPI session API.
The original Downloads project was read and copied; it was not edited.

## What runs now

```text
React lesson UI
   │ existing /api/sessions routes
FastAPI: isolated sessions, locks, revisions, request limits
   │
server/team_adapter.py
   │ forward(), train_step(), evaluate(), generate()
glasswork/team_engine/ — the copied team implementation
```

The copied model, attention, embeddings, positions, output, gradients, tokenizer,
loss and dataset modules are under `glasswork/team_engine/`. Its MIT license and
copyright are preserved in that folder. `SOURCE.json` records hashes of the files
as imported, allowing later team changes to be compared with this integration.

The source API's broad CORS/global session handling was not imported. The existing
stage-2 API provides the session isolation, request validation and revision locks.
The team's character model and gradient implementation perform the computations.

## Actual architecture

| Property | Active team engine |
| --- | --- |
| Tokenization | Characters, normalized to lowercase; source vocabulary order preserved |
| Vocabulary | 10 characters from `the cat sat on the mat` |
| Embedding width | 4 |
| Context | 128 characters |
| Attention | One causal attention head |
| Stored model parameters | 640 |
| Parameters updated by training | 40 output-projection weights |
| Optimizer | SGD, learning rate 0.1 |
| One UI training step | One sampled context/next-character example and one SGD update |
| Training sentence | `the cat sat on the mat` |
| Held-out sentence | `the mat sat on the cat` |

The source computes the gradient flowing into the final hidden state but does not
propagate it through attention or embeddings. Accordingly, those 600 parameters
stay fixed. There is no feed-forward block, layer normalization or Adam in this
engine. The UI now reports the actual architecture and training scope.

This is a working output-layer learning experiment. It is not complete transformer
backpropagation, and generated text remains mostly incoherent with this small fixed
feature extractor and one-sentence corpus. The original stage-1 full-transformer
implementation and artifacts remain available as historical work; the web server
uses the team engine and its separate checkpoint by default.

## Integration changes

- Converted loose imports to package-relative imports so the engine can be imported
  safely by the server without changing Python's global module lookup path.
- Added a local seeded RNG while preserving the source's initialization order and
  distributions. Creating a session cannot reseed or perturb another session.
- Made temperature zero exactly greedy. Negative/non-finite temperatures fail
  clearly; positive temperatures use stable softmax arithmetic.
- Normalized prompts and training targets consistently, including uppercase input.
  Empty inputs, unsupported characters and one-character evaluation corpora now
  fail clearly rather than raising incidental index/division errors.
- Computed model loss directly with log-sum-exp, avoiding the old probability floor
  that capped loss. The normal-range source calculations still match numerically.
- Added a committed update counter and checks before replacing output weights.
- Added versioned, validated, pickle-free checkpoints with atomic save, vocabulary,
  configuration and training RNG state for exact resume on the tested environment.
- Adapted ragged attention rows to `[batch, head, query, key]` by padding only future
  positions with zeros. Existing visible attention values are preserved exactly.
- Updated frontend metadata, vocabulary counts, context limits, head selectors,
  training explanations and model notes. No old 28,186-parameter or Adam claims are
  shown for this engine. Loss curves evaluate all examples after each update.

Trace schema **2** describes this engine explicitly. It contains actual combined
embeddings, attention weights, head outputs and next-token logits/probabilities.
It does not invent feed-forward or normalization activations absent from the model.

## Run or rebuild

```bash
# Run from GlassworkV1. The integrated checkpoint and frontend build already exist.
.venv/bin/python scripts/serve.py

# Rebuild the team's reproducible checkpoint if needed:
.venv/bin/python scripts/train_team.py --steps 2000

# Rebuild the frontend after edits:
cd frontend
npm run build
```

Open **http://127.0.0.1:8001**. Restart an already-running server after changing
backend code. Old browser session IDs are recovered when the page reloads.

Active checkpoint: `artifacts/team-glassbox.npz`. Its training report is
`artifacts/team-training-report.json`. This is a newly trained checkpoint using
your engine; the supplied Downloads project did not include pretrained weights.
The original `scripts/train.py` trains the earlier full transformer, not the
active team engine. Historical stage-1/stage-2 documents describe that earlier model.

## Verification

- Ran the untouched original backend with a fixed NumPy seed and captured an
  independent fixture in `tests/fixtures/team_source_reference.json`.
- Imported engine matches original embeddings, attention, logits, probabilities,
  one-step output update, hidden gradient and seeded generation to numerical precision.
- Checked all 40 output-weight gradients and 4 hidden-state gradients against
  central finite differences. Maximum output-gradient error: **2.602e-11**.
- Verified only declared output parameters change during training, causal masking,
  normalized attention, RNG isolation, text normalization, input limits, checkpoint
  corruption rejection, and exact checkpoint training resume.
- **34 Python tests passed:** 19 original-engine regressions, 8 team-engine checks
  and 7 stage-2 API integration checks. Output: `artifacts/team-integration-tests.txt`.
- The React TypeScript check and production build passed.

Browser verification uses `scripts/test_browser.py`. It checks real inference,
training, stop/reset/restore, temperature invariance, single-head selectors,
parameter counts, uppercase normalization, prompts longer than 32 characters,
model explanations, keyboard behavior and mobile layout. Results and screenshots
are saved as `artifacts/team-integration-*`.

Final browser result: **23 checks passed in Chromium**, with zero uncaught page
errors and external requests blocked. The source SHA-256 hashes were rechecked;
all nine original backend source files are unchanged.

The 2,000-update team checkpoint reduced training loss from **2.3010 to 2.2545**
and held-out loss from **2.3012 to 2.2571** nats. Improvement is modest because only
the output layer learns from fixed, randomly initialized features. These figures
belong to this engine; the earlier full-transformer scores do not apply here.
