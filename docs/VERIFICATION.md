# Stage 1 verification

Verified locally on **2026-09-26** using Python **3.14.7** and NumPy **2.5.3**.
This is measured evidence for the engine, not a guarantee of error-free software.

## Automated checks

Command: `.venv/bin/python -m unittest discover -s tests -v`

**19 tests passed.** The captured output is in `artifacts/test-results.txt`.

| Check | Observed result |
| --- | --- |
| Central finite differences for every scalar parameter in a small model | 223 checked; maximum absolute error `9.175e-08` |
| Random-direction gradient checks in a two-block model | Passed for every parameter array |
| Causal attention | Changing future tokens did not change prefix logits |
| Attention normalization/masking | Rows sum to one; future weights are exactly zero |
| Trace correctness | Head outputs and residual sums recompute from the recorded tensors |
| Trace isolation and strict JSON | Mutating trace copies cannot change model output; no non-finite JSON values |
| Extreme-logit loss | Correct finite loss/gradient for logits `[1000, 0, -1000]` |
| Layer normalization | Constant input finite; uniform shifts preserve normalized output |
| Temperature and sampling | Distributions normalize, higher tested temperature increases entropy, weights unchanged |
| Initialization, batch independence, sampling seeds | Reproducible on this environment |
| Tiny sequence overfit | Loss `0.695950 -> 0.001205`; exact alternating sequence generated greedily |
| Adam reference calculation | First update matches a hand-computed result |
| Clipping and failed updates | Expected norm/clip scale; non-finite gradient rejected without partial weight update |
| Checkpoint inference | Predictions exactly equal after save/load |
| Checkpoint training resume | Next minibatch metrics, parameters and Adam moments exactly equal |
| Invalid checkpoints | Non-finite parameters and unknown format versions rejected |
| Evaluation | Final partial chunk correctly weighted by its number of targets |
| Single-token and maximum-context backward | Finite gradients; weights remain unchanged before optimizer step |
| Dataset separation | No identical train/validation sentences; held-out characters known from training vocabulary |
| Input validation and Unicode tokenizer | Round trips and explicit invalid-input errors passed |

Finite-difference tolerances are `2e-6 + 2e-4 * abs(numerical_gradient)`, with a
central perturbation of `1e-5`. The exhaustive small model uses repeated token IDs,
two attention heads and two examples, exercising embedding accumulation and batch
averaging. These numerical checks validate the implementation independently of
whether a training curve happens to go down.

## Full default model run

Command: `.venv/bin/python scripts/train.py`

- **28,186** trainable parameters; 26-character vocabulary.
- Two blocks, two heads per block, width 32, feed-forward width 128, context 32.
- **1,200** Adam updates; minibatch size 8; model seed 42; training RNG seed 123.
- Training loss: **3.2536 -> 0.300649** nats per next-character target.
- Held-out loss: **3.2602 -> 0.675038** nats per next-character target.
- Recorded wall time: approximately **28.96 seconds**, including periodic evaluation.
- Median training-step time: approximately **22.72 ms** on this machine.

Raw measurements, every 50-step evaluation, corpus hashes, configuration,
environment and generated samples are in `artifacts/training-report.json`.
Performance is local evidence, not a promise for another CPU or a future server.

The lowest recorded held-out loss occurred before the final step (approximately
0.594 at step 550). Later training loss kept declining while held-out loss
fluctuated upward. This is a useful example of overfitting on a narrow corpus.
The supplied checkpoint is the **final step-1,200 state**, not a checkpoint selected
as best on validation. No general-language benchmark claim is made.

## Saved artifacts verified

- `artifacts/tiny-transformer.npz`: real trained weights, vocabulary, Adam state,
  update count and training RNG. Reloaded loss on both complete streams matched
  the final report **exactly** in this environment.
- `artifacts/example-trace.json`: real two-block forward pass for `the `, strict
  JSON with embeddings, Q/K/V, attention, feed-forward activations and logits.
- `artifacts/test-results.txt`: passing test output and numerical-check results.
- Python syntax compilation passed for the engine, scripts and tests.

Command: `.venv/bin/python scripts/demo.py --temperature 0`

Actual generated sample:

```text
the blue bird flies in the sky.
the green frog jumps in the pond.
the blue pond is blue.
the tree frog.
```

This includes familiar training phrases and imperfect continuations. It verifies
learned character generation; it does not demonstrate fluent general-purpose AI.

## Not part of this verification

No browser UI, FastAPI server, GPU path, Gemini integration or public deployment
has been implemented or tested yet. No cross-platform bitwise reproducibility or
adversarial server-load testing is claimed. These belong to subsequent stages.
