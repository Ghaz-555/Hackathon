"""A small decoder-only, pre-normalization transformer with manual backprop.

Shape convention: B=batch, T=time, C=model width, H=heads, D=C/H.
The forward path is shared by training, evaluation and visualization. Traces
are copies of actual activations, never a separately simulated animation.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .math import (cross_entropy, gelu, gelu_backward, layer_norm,
                   layer_norm_backward, linear, linear_backward, softmax,
                   softmax_backward)


@dataclass(frozen=True)
class Config:
    vocab_size: int
    context_length: int = 32
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 128
    seed: int = 42

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
        if self.vocab_size < 2 or self.d_model < 2 or self.d_model % self.n_heads:
            raise ValueError("Need vocabulary >=2 and model width >=2 divisible by head count")
        # Educational-engine bounds also prevent accidental huge checkpoint loads.
        if (self.vocab_size > 4096 or self.context_length > 256 or self.d_model > 256
                or self.n_layers > 8 or self.d_ff > 1024):
            raise ValueError("Configuration exceeds this educational engine's size limits")


@dataclass
class ForwardResult:
    logits: np.ndarray
    trace: dict | None = None


def distribution(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Sampling transformation only: it cannot update any model weights.

    At temperature zero use deterministic argmax (first index breaks ties).
    Subtract the maximum BEFORE division for stability at tiny temperatures.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 1 or logits.shape[-1] < 2 or not np.isfinite(logits).all():
        raise ValueError("Need finite logits with vocabulary size >=2")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if temperature == 0:
        out = np.zeros_like(logits)
        np.put_along_axis(out, logits.argmax(axis=-1, keepdims=True), 1, axis=-1)
        return out
    with np.errstate(over="ignore", under="ignore"):
        exp = np.exp((logits - logits.max(axis=-1, keepdims=True)) / temperature)
    return exp / exp.sum(axis=-1, keepdims=True)


class TinyTransformer:
    def __init__(self, config: Config):
        self.config = config
        self.params: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(config.seed)
        c = config.d_model

        def weight(name, shape, std=0.02):
            self.params[name] = rng.normal(0, std, shape).astype(np.float64)

        def norm(name):
            self.params[name + ".gain"] = np.ones(c)
            self.params[name + ".bias"] = np.zeros(c)

        def dense(name, inputs, outputs, residual=False):
            # Scale residual projections to limit variance growth through depth.
            std = 0.02 / np.sqrt(2 * config.n_layers) if residual else 0.02
            weight(name + ".weight", (inputs, outputs), std)
            self.params[name + ".bias"] = np.zeros(outputs)

        weight("token_embedding", (config.vocab_size, c))
        weight("position_embedding", (config.context_length, c))
        for i in range(config.n_layers):
            p = f"blocks.{i}"
            norm(p + ".norm1")
            dense(p + ".qkv", c, 3 * c)
            dense(p + ".attention_output", c, c, residual=True)
            norm(p + ".norm2")
            dense(p + ".ff_in", c, config.d_ff)
            dense(p + ".ff_out", config.d_ff, c, residual=True)
        norm("final_norm")
        dense("lm_head", c, config.vocab_size)

    @property
    def parameter_count(self):
        return sum(p.size for p in self.params.values())

    def _validate_ids(self, ids):
        ids = np.asarray(ids)
        if (ids.ndim != 2 or not ids.shape[0] or not 1 <= ids.shape[1] <= self.config.context_length
                or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.config.vocab_size)):
            raise ValueError("Expected valid integer IDs [batch,time], 1 <= time <= context_length")
        return ids

    def _norm(self, x, name):
        return layer_norm(x, self.params[name + ".gain"], self.params[name + ".bias"])

    def _linear(self, x, name):
        return linear(x, self.params[name + ".weight"], self.params[name + ".bias"])

    def _forward(self, ids, capture_trace=False, retain_cache=False):
        ids = self._validate_ids(ids)
        b, t = ids.shape
        c, h = self.config.d_model, self.config.n_heads
        d = c // h
        token = self.params["token_embedding"][ids]
        position = self.params["position_embedding"][:t]
        x = token + position
        # True means the key position is visible; every row includes itself.
        visible = np.tril(np.ones((t, t), dtype=bool))
        trace = None
        if capture_trace:
            trace = {"schema_version": 1, "input_ids": ids.copy(),
                     "token_embeddings": token.copy(), "position_embeddings": position.copy(),
                     "embedding_sum": x.copy(), "causal_mask": visible.copy(), "blocks": []}
        caches = []
        for i in range(self.config.n_layers):
            p = f"blocks.{i}"
            n1, norm1_cache = self._norm(x, p + ".norm1")
            qkv = self._linear(n1, p + ".qkv")
            # Each head owns a contiguous C/H slice; transpose separates H and T.
            q, k, v = [part.reshape(b, t, h, d).transpose(0, 2, 1, 3)
                       for part in np.split(qkv, 3, axis=-1)]
            scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(d)
            attention = softmax(np.where(visible, scores, -np.inf))
            heads = attention @ v
            joined = heads.transpose(0, 2, 1, 3).reshape(b, t, c)
            projected = self._linear(joined, p + ".attention_output")
            residual = x + projected
            n2, norm2_cache = self._norm(residual, p + ".norm2")
            hidden = self._linear(n2, p + ".ff_in")
            activated = gelu(hidden)
            ff = self._linear(activated, p + ".ff_out")
            x = residual + ff
            if retain_cache:
                caches.append((n1, norm1_cache, q, k, v, attention, joined,
                               n2, norm2_cache, hidden, activated))
            if capture_trace:
                # Scores are pre-mask so the trace remains valid strict JSON.
                # The separately supplied boolean mask determines allowed edges.
                trace["blocks"].append({
                    "normalized_attention_input": n1.copy(), "query": q.copy(),
                    "key": k.copy(), "value": v.copy(), "scaled_scores": scores.copy(),
                    "attention": attention.copy(), "head_outputs": heads.copy(),
                    "attention_projection": projected.copy(), "attention_residual": residual.copy(),
                    "normalized_ff_input": n2.copy(), "ff_pre_activation": hidden.copy(),
                    "ff_activation": activated.copy(), "ff_projection": ff.copy(), "output": x.copy()})
        final, final_cache = self._norm(x, "final_norm")
        logits = self._linear(final, "lm_head")
        if not np.isfinite(logits).all():
            raise FloatingPointError("Model produced non-finite logits")
        if capture_trace:
            trace.update(final_normalized=final.copy(), logits=logits.copy(),
                         probabilities=softmax(logits))
        cache = (ids.copy(), caches, final, final_cache) if retain_cache else None
        return ForwardResult(logits, trace), cache

    def forward(self, ids, *, capture_trace=False) -> ForwardResult:
        """Pure inference: no optimizer or random state is touched."""
        return self._forward(ids, capture_trace=capture_trace)[0]

    def loss(self, ids, targets) -> float:
        return cross_entropy(self.forward(ids).logits, np.asarray(targets))[0]

    def loss_and_gradients(self, ids, targets):
        """One explicit reverse pass. Returns fresh gradients, never updates weights."""
        output, (ids, caches, final, final_cache) = self._forward(ids, retain_cache=True)
        loss, dy = cross_entropy(output.logits, np.asarray(targets))
        grads = {}

        def dense_backward(dout, x, name):
            dx, dw, db = linear_backward(dout, x, self.params[name + ".weight"])
            grads[name + ".weight"], grads[name + ".bias"] = dw, db
            return dx

        def norm_backward(dout, cache, name):
            dx, dg, db = layer_norm_backward(dout, cache, self.params[name + ".gain"])
            grads[name + ".gain"], grads[name + ".bias"] = dg, db
            return dx

        dx = dense_backward(dy, final, "lm_head")
        dx = norm_backward(dx, final_cache, "final_norm")
        b, t = ids.shape
        c, h = self.config.d_model, self.config.n_heads
        d = c // h
        for i in reversed(range(self.config.n_layers)):
            p = f"blocks.{i}"
            n1, nc1, q, k, v, attention, joined, n2, nc2, hidden, activated = caches[i]
            # Residual addition sends the SAME upstream gradient down both paths.
            dff = dense_backward(dx, activated, p + ".ff_out")
            dff = gelu_backward(dff, hidden)
            dn2 = dense_backward(dff, n2, p + ".ff_in")
            dr = dx + norm_backward(dn2, nc2, p + ".norm2")
            djoined = dense_backward(dr, joined, p + ".attention_output")
            dheads = djoined.reshape(b, t, h, d).transpose(0, 2, 1, 3)
            da = dheads @ v.swapaxes(-1, -2)
            dv = attention.swapaxes(-1, -2) @ dheads
            dscores = softmax_backward(da, attention) / np.sqrt(d)
            # Masked probabilities are exactly zero, so masked score gradients are too.
            dq = dscores @ k
            dk = dscores.swapaxes(-1, -2) @ q
            dqkv = np.concatenate([part.transpose(0, 2, 1, 3).reshape(b, t, c)
                                   for part in (dq, dk, dv)], axis=-1)
            dn1 = dense_backward(dqkv, n1, p + ".qkv")
            dx = dr + norm_backward(dn1, nc1, p + ".norm1")
        grads["token_embedding"] = np.zeros_like(self.params["token_embedding"])
        # Fancy-index += would lose repeats. add.at accumulates every occurrence.
        np.add.at(grads["token_embedding"], ids, dx)
        grads["position_embedding"] = np.zeros_like(self.params["position_embedding"])
        grads["position_embedding"][:t] = dx.sum(axis=0)
        if not all(np.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError("Non-finite gradient")
        return loss, grads

    def generate(self, prompt_ids, *, max_new_tokens=64, temperature=0.8, seed=7):
        """Autoregressive sampling with a local RNG and sliding context window.

        Positions restart at zero when the window slides. No KV cache is used:
        deliberately recomputing the tiny model makes every step inspectable.
        """
        ids = np.asarray(prompt_ids)
        if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= self.config.vocab_size)):
            raise ValueError("Need a nonempty prompt of valid integer token IDs")
        if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be an integer between 0 and 4096")
        distribution(np.zeros(self.config.vocab_size), temperature)  # Validate even for zero tokens.
        rng = np.random.default_rng(seed)
        output = ids.tolist()
        for _ in range(max_new_tokens):
            window = np.array([output[-self.config.context_length:]], dtype=np.int64)
            logits = self.forward(window).logits[0, -1]
            probabilities = distribution(logits, temperature)
            output.append(int(rng.choice(self.config.vocab_size, p=probabilities)))
        return np.array(output, dtype=np.int64)


def trace_to_json(trace: dict) -> dict:
    """Convert copied traces to plain JSON data for a future FastAPI/WebGL client."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return convert(trace)
