import numpy as np


def output_layer_gradients(
    hidden_state,
    probabilities,
    correct_index
):
    """
    Calculate gradients for the output layer.
    """

    # For softmax + cross-entropy:
    # gradient of loss with respect to logits
    d_logits = probabilities.copy()
    d_logits[correct_index] -= 1

    # Gradient for output weights
    d_output_weights = np.outer(
        hidden_state,
        d_logits
    )

    return d_logits, d_output_weights