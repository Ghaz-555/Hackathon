import numpy as np


def create_output_weights(embedding_size, vocab_size, rng=None):
    rng = np.random if rng is None else rng
    return rng.uniform(
        -0.1,
        0.1,
        size=(embedding_size, vocab_size)
    )


def calculate_logits(hidden_state, output_weights):
    return hidden_state @ output_weights


def softmax_with_temperature(logits, temperature=1.0):
    """Stable sampling; zero means exact greedy selection, not a small temperature."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("Expected finite, one-dimensional logits")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if temperature == 0:
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1
        return result
    with np.errstate(over="ignore", under="ignore"):
        shifted = (logits - np.max(logits)) / temperature
        exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
