import math


def cross_entropy_loss(probabilities, correct_character):
    """
    Measures how wrong the model's prediction was.

    Lower loss = better prediction.
    Higher loss = worse prediction.
    """

    correct_probability = probabilities[correct_character]

    # Avoid log(0)
    correct_probability = max(correct_probability, 1e-12)

    loss = -math.log(correct_probability)

    return loss