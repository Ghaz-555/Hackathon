def tokenize(text):
    # Character-level tokenizer
    return list(text.lower())


def build_vocabulary(tokens):
    vocabulary = {}

    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode(tokens, vocabulary):
    return [vocabulary[token] for token in tokens]
