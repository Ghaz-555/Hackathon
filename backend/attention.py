import numpy as np


def softmax(values):
    values = np.array(values)

    shifted_values = values - np.max(values)
    exp_values = np.exp(shifted_values)

    return exp_values / np.sum(exp_values)


def create_matrix(size):
    return np.random.uniform(
        -0.1,
        0.1,
        size=(size, size)
    )


def causal_self_attention(inputs, W_q, W_k, W_v):
    embedding_size = inputs.shape[1]

    # Create queries, keys and values
    queries = inputs @ W_q
    keys = inputs @ W_k
    values = inputs @ W_v

    attention_weights = []
    outputs = []

    for i in range(len(inputs)):

        # Only positions 0 through i are visible.
        # This is our causal mask.
        visible_keys = keys[:i + 1]

        scores = (
            queries[i] @ visible_keys.T
        ) / np.sqrt(embedding_size)

        weights = softmax(scores)

        visible_values = values[:i + 1]

        output = weights @ visible_values

        attention_weights.append(weights)
        outputs.append(output)

    return (
        np.array(outputs),
        attention_weights
    )