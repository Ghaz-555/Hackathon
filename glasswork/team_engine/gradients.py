import numpy as np


def output_layer_gradients(
    hidden_state,
    probabilities,
    correct_index,
    output_weights
):
    """
    Calculate gradients for the output layer
    and the gradient flowing back into the transformer.
    """

    # Softmax + cross-entropy gradient
    d_logits = probabilities.copy()
    d_logits[correct_index] -= 1

    # Gradient of the output weight matrix
    d_output_weights = np.outer(
        hidden_state,
        d_logits
    )

    # Gradient passed backward into the transformer
    d_hidden_state = d_logits @ output_weights.T

    return (
        d_logits,
        d_output_weights,
        d_hidden_state
    )
