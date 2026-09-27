# Stage 1 architecture and future integration

## Boundaries

`glasswork/` imports NumPy and the Python standard library only. It knows nothing
about HTTP, graphics, Gemini or accounts. The scripts handle file paths and console
output. This lets the future backend import the engine without launching training
or reading files as an import side effect.

Proposed next-stage structure:

```text
React / TypeScript / Three.js / charting libraries
                |
          FastAPI routes
                |
   serialized model worker + session state
                |
     this NumPy engine + checkpoints
```

Use a worker process or a controlled task queue for training, not a long synchronous
call on an async event loop. Only one operation may access a model while its weights
are being updated. Use separate model instances for independent learners. Emit
metrics after each committed optimizer update, and support cancellation between
steps. Stage 1 does not implement these server responsibilities.

## Shapes and forward equations

`B` = batch size, `T` = sequence length, `C` = model width, `H` = attention heads,
`D = C/H`, `F` = feed-forward width, `V` = vocabulary size.

1. Input IDs `[B,T]` index token embeddings `[V,C]`. Add learned positions `[T,C]`.
2. Normalize each token over its `C` features with learned gain and bias.
3. Project to concatenated Q/K/V `[B,T,3C]`, split into `[B,H,T,D]`.
4. Compute `scores = Q @ K.transpose / sqrt(D)` -> `[B,H,T,T]`.
5. Mask future key positions. Apply row softmax to get attention weights `A`.
6. Compute `A @ V`, join heads back into `[B,T,C]`, project, and add the residual.
7. Normalize, project `C -> F`, apply tanh-approximate GELU, project `F -> C`,
   and add the second residual.
8. Repeat blocks; final normalization and linear projection produce `[B,T,V]` logits.

Loss is mean next-token negative log likelihood in nats. For training windows
`tokens[start:start+T]`, targets are `tokens[start+1:start+T+1]`.
Attention is causal even during training, so each position cannot see its target.

## Manual backward path

Every derivative is implemented explicitly; NumPy performs array arithmetic, not
autodifferentiation. Cross-entropy gives `(softmax(logits) - one_hot(target))/B/T`.
Linear layers use `dX = dY @ W.T`, `dW = X.T @ dY`, `db = sum(dY)`.

Residual addition sends the upstream gradient along both branches, and their input
gradients add. Layer norm uses the centered, normalized input and inverse standard
deviation from the same forward pass. GELU uses the derivative of the exact tanh
approximation used in forward.

For attention, `dA = dHeads @ V.T`, `dV = A.T @ dHeads`. The softmax Jacobian product
is `A * (dA - sum(dA * A))`. Masked probabilities are zero, making their derivatives
zero too. Apply the `1/sqrt(D)` scale before deriving `dQ` and `dK`.

Token embedding gradients use `np.add.at`: repeated character IDs must accumulate
rather than overwrite. Position gradients sum over batch. Training caches are
local to a single loss/backward call and never retained on the model.

## Trace schema v1

`forward(ids, capture_trace=True)` returns `ForwardResult(logits, trace)`.
Use `trace_to_json` at the API boundary; arrays remain NumPy arrays internally.

| Field | Shape / meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `input_ids` | `[B,T]` |
| `token_embeddings`, `embedding_sum` | `[B,T,C]` |
| `position_embeddings` | `[T,C]` |
| `causal_mask` | `[T,T]`; `true` means the connection is allowed |
| `blocks[i].normalized_attention_input` | `[B,T,C]` |
| `blocks[i].query`, `key`, `value` | `[B,H,T,D]` |
| `blocks[i].scaled_scores` | `[B,H,T,T]`, **before masking** |
| `blocks[i].attention` | `[B,H,T,T]`, after masking and softmax |
| `blocks[i].head_outputs` | `[B,H,T,D]` |
| `blocks[i].attention_projection`, `attention_residual` | `[B,T,C]` |
| `blocks[i].normalized_ff_input` | `[B,T,C]` |
| `blocks[i].ff_pre_activation`, `ff_activation` | `[B,T,F]` |
| `blocks[i].ff_projection`, `output` | `[B,T,C]` |
| `final_normalized` | `[B,T,C]` |
| `logits`, `probabilities` | `[B,T,V]`; probabilities use temperature 1 |

The mask is separate from scores to avoid exporting non-finite `-Infinity` to JSON.
The visualizer must honor that mask rather than displaying raw future-position
scores as usable connections. Head weights show information mixing, not the
model's complete reasoning or a human-readable explanation of causality.

Traces are opt-in and detached copies. Request one selected example/step for the UI
instead of exporting every minibatch. A future server should include checkpoint
identity, optimizer step, vocabulary, experiment parameters and sampling seed in
its response envelope. Use the recorded values, never re-created placeholder math.

## Checkpoint v1

Compressed `.npz` file with scalar JSON `header` plus float64 arrays named
`param/<name>`, `adam_m/<name>` and `adam_v/<name>`. The header stores config,
vocabulary order, Adam hyperparameters, update count, NumPy RNG state and metadata.
Loading checks shapes, dtypes, finite values, nonnegative second moments, version
and vocabulary size. Pickle is disabled. Saves replace the destination atomically.

Inference resumes exactly. Training resumes exactly on the tested environment when
given the same token stream and batch size. Different hardware/BLAS implementations
can produce small floating-point differences; cross-platform bitwise identity is
not promised. RNG state is local, and sampling does not advance the training RNG.

## Next-stage educational experiments

- **Temperature:** use `distribution` on fixed logits. No training call is involved.
- **Context:** forward the original and shortened prompt; compare measured outputs.
  Position shifts are part of this intervention and must be explained.
- **Learning:** capture logits before an optimizer step and afterwards on the same
  held-fixed input. Distinguish the current minibatch loss from full training and
  held-out loss. The Adam learning rate is independent of sampling temperature.
- **Tutor:** send Gemini measured results and the learner's answer. Gemini does not
  supply logits, gradients, scores or the engine's actual computations.

Head ablation, parameter editing, live gradient visualization, lesson grading and
server cancellation are future features, not silently simulated stage-1 behavior.
