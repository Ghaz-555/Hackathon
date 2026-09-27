"""Adam, gradient clipping and deterministic minibatch selection, from scratch."""

from dataclasses import asdict, dataclass

import numpy as np

from .math import cross_entropy


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: float = 0.003
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Optimizer settings must be finite")
        if (self.learning_rate <= 0 or self.epsilon <= 0 or self.max_grad_norm <= 0
                or not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1):
            raise ValueError("Invalid Adam configuration")


class Adam:
    def __init__(self, params, config=None):
        self.config = config or AdamConfig()
        self.step_count = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, gradients):
        if params.keys() != gradients.keys() or params.keys() != self.m.keys():
            raise ValueError("Parameter and gradient names must match")
        for name, p in params.items():
            g = gradients[name]
            if g.shape != p.shape:
                raise ValueError(f"Gradient shape mismatch: {name}")
            if not np.isfinite(g).all() or not np.isfinite(p).all():
                raise FloatingPointError(f"Non-finite parameter/gradient: {name}")
        # Stable norm: scale before squaring to avoid overflow on large gradients.
        peak = max(float(np.max(np.abs(g))) for g in gradients.values())
        norm = 0.0 if peak == 0 else peak * np.sqrt(sum(float(np.sum((g / peak)**2)) for g in gradients.values()))
        if not np.isfinite(norm):
            raise FloatingPointError("Gradient norm overflow")
        cfg = self.config
        clip = min(1.0, cfg.max_grad_norm / max(norm, 1e-300))
        step = self.step_count + 1
        staged = {}
        for name, p in params.items():
            g = gradients[name] * clip
            m = cfg.beta1 * self.m[name] + (1 - cfg.beta1) * g
            v = cfg.beta2 * self.v[name] + (1 - cfg.beta2) * g * g
            # Bias corrections matter especially for the first few educational steps.
            mhat, vhat = m / (1 - cfg.beta1**step), v / (1 - cfg.beta2**step)
            updated = p - cfg.learning_rate * mhat / (np.sqrt(vhat) + cfg.epsilon)
            if not all(np.isfinite(a).all() for a in (m, v, updated)):
                raise FloatingPointError("Non-finite Adam update; no weights were changed")
            staged[name] = (updated, m, v)
        # Commit only once every update is valid: no partially changed model on error.
        for name, (updated, m, v) in staged.items():
            params[name][...] = updated
            self.m[name], self.v[name] = m, v
        self.step_count = step
        return {"gradient_norm": float(norm), "clip_scale": float(clip)}


def validate_stream(tokens, vocab_size, context_length):
    tokens = np.asarray(tokens)
    if (tokens.ndim != 1 or len(tokens) <= context_length or tokens.dtype.kind not in "iu"
            or np.any(tokens < 0) or np.any(tokens >= vocab_size)):
        raise ValueError("Token stream needs context_length + 1 valid integer IDs")
    return tokens


class Trainer:
    """Synchronous steps are easy to run in a dedicated server worker later.

    Inference is read-only, but callers must serialize it with training updates
    on the same model. This class is not a shared-state concurrency framework.
    """
    def __init__(self, model, optimizer=None, *, seed=123):
        self.model = model
        self.optimizer = optimizer or Adam(model.params)
        self.rng = np.random.default_rng(seed)

    def train_step(self, tokens, *, batch_size=8):
        cfg = self.model.config
        tokens = validate_stream(tokens, cfg.vocab_size, cfg.context_length)
        if type(batch_size) is not int or not 1 <= batch_size <= 1024:
            raise ValueError("Batch size must be between 1 and 1024")
        # The upper bound is exclusive; final valid start is N - T - 1.
        starts = self.rng.integers(0, len(tokens) - cfg.context_length, size=batch_size)
        indices = starts[:, None] + np.arange(cfg.context_length)
        loss, grads = self.model.loss_and_gradients(tokens[indices], tokens[indices + 1])
        metrics = self.optimizer.step(self.model.params, grads)
        return {"step": self.optimizer.step_count, "loss": loss, **metrics}


def evaluate(model, tokens):
    """Deterministic loss over every target exactly once; never trains on it.

    Windows do not cross the supplied stream's boundaries. Context resets at
    each chunk, including the last partial chunk. No RNG state is consumed.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError("Evaluation needs at least two tokens")
    total, count = 0.0, 0
    for start in range(0, len(tokens) - 1, model.config.context_length):
        chunk = tokens[start:start + model.config.context_length + 1]
        loss, _ = cross_entropy(model.forward(chunk[:-1][None, :]).logits, chunk[1:][None, :])
        total += loss * (len(chunk) - 1)
        count += len(chunk) - 1
    return total / count
