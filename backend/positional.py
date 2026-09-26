import random


def create_positional_embeddings(context_length, embedding_size):
    positions = []

    for position in range(context_length):
        vector = [
            random.uniform(-1, 1)
            for _ in range(embedding_size)
        ]

        positions.append(vector)

    return positions


def add_position_embeddings(tokens, embeddings, positional_embeddings):
    combined = []

    for position, token in enumerate(tokens):
        token_vector = embeddings[token]
        position_vector = positional_embeddings[position]

        new_vector = []

        for i in range(len(token_vector)):
            new_vector.append(
                token_vector[i] + position_vector[i]
            )

        combined.append(new_vector)

    return combined