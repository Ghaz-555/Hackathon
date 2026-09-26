import numpy as np


def create_embeddings(vocabulary, embedding_size=4):
    embeddings = {}

    for character in vocabulary:
        embeddings[character] = np.random.uniform(
            -1,
            1,
            size=embedding_size
        )

    return embeddings