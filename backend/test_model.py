from model import GlassBoxModel


training_text = "the cat sat on the mat"

model = GlassBoxModel(
    training_text,
    embedding_size=4,
    context_length=128
)

prompt = "the ca"

temperatures = [0.2, 1.0, 2.0]

for temperature in temperatures:

    result = model.forward(
        prompt,
        temperature=temperature
    )

    probabilities = result["probabilities"]

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print(f"\n--- Temperature: {temperature} ---")

    for character, probability in sorted_probabilities:
        display_character = (
            "[SPACE]" if character == " " else character
        )

        print(
            f"{display_character}: "
            f"{probability * 100:.2f}%"
        )


print("\n--- Loss Test ---")

loss = model.calculate_loss(
    "the ca",
    correct_character="t"
)

print("Correct next character: 't'")
print("Loss:", round(loss, 4))


print("\n--- Training Test ---")

training_prompt = "the ca"
correct_character = "t"

before_loss = model.calculate_loss(
    training_prompt,
    correct_character
)

print("Loss before training:", round(before_loss, 4))

for step in range(10):
    training_result = model.train_step(
        training_prompt,
        correct_character,
        learning_rate=0.1
    )

    print(
        f"Step {step + 1}: "
        f"loss = {training_result['loss']:.4f}"
    )

after_loss = model.calculate_loss(
    training_prompt,
    correct_character
)

print("Loss after training:", round(after_loss, 4))