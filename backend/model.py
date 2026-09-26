from tokenizer import tokenize, build_vocabulary
from embeddings import create_embeddings
from positional import (
    create_positional_embeddings,
    add_position_embeddings
)
from attention import causal_self_attention


class GlassBoxModel:

    def __init__(self, text, embedding_size=4, context_length=128):
        self.embedding_size = embedding_size
        self.context_length = context_length

        # Build character vocabulary
        self.tokens = tokenize(text)
        self.vocabulary = build_vocabulary(self.tokens)

        # Character embeddings
        self.embeddings = create_embeddings(
            self.vocabulary,
            embedding_size
        )

        # Position embeddings
        self.positional_embeddings = create_positional_embeddings(
            context_length,
            embedding_size
        )

    def forward(self, text):
        tokens = tokenize(text)

        # Don't allow inputs longer than our context window
        tokens = tokens[-self.context_length:]

        # Combine character + position information
        combined = add_position_embeddings(
            tokens,
            self.embeddings,
            self.positional_embeddings
        )

        # Transformer attention
        outputs, attention_weights = causal_self_attention(combined)

        return {
            "tokens": tokens,
            "combined_embeddings": combined,
            "attention_weights": attention_weights,
            "attention_output": outputs
        }