import random
from tokenizer import tokenize, build_vocabulary


def create_embeddings(vocabulary, embedding_size=4):
    embeddings = {}

    for word in vocabulary:
        embeddings[word] = [
            random.uniform(-1, 1)
            for _ in range(embedding_size)
        ]

    return embeddings


sentence = "the cat sat on the mat"

tokens = tokenize(sentence)
vocabulary = build_vocabulary(tokens)
embeddings = create_embeddings(vocabulary)

print("Tokens:", tokens)
print("Vocabulary:", vocabulary)

for word, vector in embeddings.items():
    print(word, "->", vector)



from attention import calculate_attention

attention = calculate_attention(tokens, embeddings)

print("\nAttention:")

for word, scores in attention.items():
    print(word, "->", scores)