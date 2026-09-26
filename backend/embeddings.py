import random


def create_embeddings(vocabulary, embedding_size=4):
    embeddings = {}

    for character in vocabulary:
        embeddings[character] = [
            random.uniform(-1, 1)
            for _ in range(embedding_size)
        ]

    return embeddings