import math
import random


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(values):
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def create_matrix(size):
    return [
        [random.uniform(-0.1, 0.1) for _ in range(size)]
        for _ in range(size)
    ]


def matrix_vector(vector, matrix):
    result = []

    for column in range(len(matrix[0])):
        value = 0

        for row in range(len(vector)):
            value += vector[row] * matrix[row][column]

        result.append(value)

    return result


def causal_self_attention(inputs, W_q, W_k, W_v):
    embedding_size = len(inputs[0])

    queries = [matrix_vector(x, W_q) for x in inputs]
    keys = [matrix_vector(x, W_k) for x in inputs]
    values = [matrix_vector(x, W_v) for x in inputs]

    attention_weights = []
    outputs = []

    for i in range(len(inputs)):
        scores = []

        # Causal attention:
        # position i can only look at positions <= i
        for j in range(i + 1):
            score = dot_product(queries[i], keys[j])
            score /= math.sqrt(embedding_size)
            scores.append(score)

        weights = softmax(scores)
        attention_weights.append(weights)

        output = [0.0] * embedding_size

        for j, weight in enumerate(weights):
            for k in range(embedding_size):
                output[k] += weight * values[j][k]

        outputs.append(output)

    return outputs, attention_weights