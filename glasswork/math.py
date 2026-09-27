"""Explicit forward/backward building blocks. No automatic differentiation.

Arrays end in a feature dimension; leading dimensions are batch and sequence.
Float64 is intentional for teaching and finite-difference gradient checks.
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Normalize the last axis without exponentiating large positive values."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def softmax_backward(dy: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    # Jacobian-vector product avoids allocating a vocabulary-by-vocabulary matrix.
    return probabilities * (dy - (dy * probabilities).sum(axis=-1, keepdims=True))


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight + bias


def linear_backward(dy: np.ndarray, x: np.ndarray, weight: np.ndarray):
    flat_x, flat_dy = x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1])
    return dy @ weight.T, flat_x.T @ flat_dy, flat_dy.sum(axis=0)


def layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps=1e-5):
    centered = x - x.mean(axis=-1, keepdims=True)
    inverse_std = 1.0 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    normalized = centered * inverse_std
    return normalized * gain + bias, (normalized, inverse_std)


def layer_norm_backward(dy: np.ndarray, cache, gain: np.ndarray):
    normalized, inverse_std = cache
    dxhat = dy * gain
    dx = inverse_std * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                       - normalized * (dxhat * normalized).mean(axis=-1, keepdims=True))
    axes = tuple(range(dy.ndim - 1))
    return dx, (dy * normalized).sum(axis=axes), dy.sum(axis=axes)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh approximation, used consistently in both forward and derivative."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_backward(dy: np.ndarray, x: np.ndarray) -> np.ndarray:
    k, c = np.sqrt(2 / np.pi), 0.044715
    t = np.tanh(k * (x + c * x**3))
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * c * x * x))


def cross_entropy(logits: np.ndarray, targets: np.ndarray):
    """Mean negative log likelihood and dLoss/dLogits over every batch token.

    Use log-sum-exp directly: -log(softmax) can hit log(0) for unlikely targets.
    This loss is measured in nats, since NumPy log is the natural logarithm.
    """
    if logits.ndim != 3 or targets.shape != logits.shape[:-1]:
        raise ValueError("Expected logits [batch,time,vocabulary] and targets [batch,time]")
    if targets.dtype.kind not in "iu" or np.any(targets < 0) or np.any(targets >= logits.shape[-1]):
        raise ValueError("Target IDs are outside the vocabulary")
    if not np.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits")
    flat = logits.reshape(-1, logits.shape[-1])
    y = targets.reshape(-1)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1))
    loss = np.mean(log_normalizer - shifted[np.arange(len(y)), y])
    grad = softmax(flat)
    grad[np.arange(len(y)), y] -= 1
    grad /= len(y)
    return float(loss), grad.reshape(logits.shape)
