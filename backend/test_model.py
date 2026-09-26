from model import GlassBoxModel


training_text = "the cat sat on the mat"

model = GlassBoxModel(
    training_text,
    embedding_size=4,
    context_length=128
)

result = model.forward("the cat")

print("Tokens:")
print(result["tokens"])

print("\nAttention:")
for position, weights in enumerate(result["attention_weights"]):
    print(
        f"Position {position} ({result['tokens'][position]!r}) ->",
        [round(weight, 3) for weight in weights]
    )