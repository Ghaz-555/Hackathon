import numpy as np


def create_embeddings(vocabulary, embedding_size=4, rng=None):
    rng = np.random if rng is None else rng
    embeddings = {}

    for character in vocabulary:
        embeddings[character] = rng.uniform(
            -1,
            1,
            size=embedding_size
        )

    return embeddings
