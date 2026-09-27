import numpy as np


def create_output_weights(embedding_size, vocab_size):
    return np.random.uniform(
        -0.1,
        0.1,
        size=(embedding_size, vocab_size)
    )


def calculate_logits(hidden_state, output_weights):
    return hidden_state @ output_weights


def softmax_with_temperature(logits, temperature=1.0):
    # Prevent division by zero
    temperature = max(temperature, 0.01)

    scaled_logits = logits / temperature

    # Numerical stability
    shifted_logits = scaled_logits - np.max(scaled_logits)

    exponentials = np.exp(shifted_logits)

    return exponentials / np.sum(exponentials)