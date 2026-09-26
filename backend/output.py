import math
import random


def create_output_weights(embedding_size, vocab_size):
    return [
        [
            random.uniform(-0.1, 0.1)
            for _ in range(vocab_size)
        ]
        for _ in range(embedding_size)
    ]


def calculate_logits(hidden_state, output_weights):
    logits = []

    vocab_size = len(output_weights[0])

    for vocab_index in range(vocab_size):
        value = 0

        for i in range(len(hidden_state)):
            value += hidden_state[i] * output_weights[i][vocab_index]

        logits.append(value)

    return logits


def softmax_with_temperature(logits, temperature=1.0):
    # Prevent division by zero
    temperature = max(temperature, 0.01)

    scaled_logits = [
        value / temperature
        for value in logits
    ]

    max_logit = max(scaled_logits)

    exponentials = [
        math.exp(value - max_logit)
        for value in scaled_logits
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]