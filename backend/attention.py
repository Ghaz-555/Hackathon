import math


def dot_product(vector1, vector2):
    total = 0

    for i in range(len(vector1)):
        total += vector1[i] * vector2[i]

    return total


def softmax(numbers):
    exponentials = [math.exp(number) for number in numbers]
    total = sum(exponentials)

    return [number / total for number in exponentials]


def calculate_attention(tokens, embeddings):
    attention = {}

    for current_word in tokens:
        scores = []

        for other_word in tokens:
            score = dot_product(
                embeddings[current_word],
                embeddings[other_word]
            )
            scores.append(score)

        attention[current_word] = softmax(scores)

    return attention