def create_training_examples(text, context_length=128):
    examples = []

    for i in range(1, len(text)):
        start = max(0, i - context_length)

        input_text = text[start:i]
        target_character = text[i]

        examples.append(
            (input_text, target_character)
        )

    return examples
