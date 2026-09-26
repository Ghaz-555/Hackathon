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