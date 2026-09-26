import numpy as np


def create_positional_embeddings(context_length, embedding_size):
    return np.random.uniform(
        -1,
        1,
        size=(context_length, embedding_size)
    )


def add_position_embeddings(tokens, embeddings, positional_embeddings):
    combined = []

    for position, token in enumerate(tokens):
        token_vector = embeddings[token]
        position_vector = positional_embeddings[position]

        combined.append(
            token_vector + position_vector
        )

    return np.array(combined)