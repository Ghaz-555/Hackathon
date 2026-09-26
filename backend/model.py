from tokenizer import tokenize, build_vocabulary
from embeddings import create_embeddings
from positional import (
    create_positional_embeddings,
    add_position_embeddings
)
from attention import causal_self_attention, create_matrix
from output import (
    create_output_weights,
    calculate_logits,
    softmax_with_temperature
)


class GlassBoxModel:

    def __init__(self, text, embedding_size=4, context_length=128):
        self.embedding_size = embedding_size
        self.context_length = context_length

        self.W_q = create_matrix(embedding_size)
        self.W_k = create_matrix(embedding_size)
        self.W_v = create_matrix(embedding_size)

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
        # Output layer for next-character prediction
        self.output_weights = create_output_weights(
            embedding_size,
            len(self.vocabulary)
        )

        # Reverse vocabulary lets us convert IDs back into characters
        self.id_to_character = {
            index: character
            for character, index in self.vocabulary.items()
        }

    def forward(self, text, temperature=1.0):
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
        outputs, attention_weights = causal_self_attention(
            combined,
            self.W_q,
            self.W_k,
            self.W_v
        )

        
        
        # Use the final character's transformer output
        # to predict what character comes next
        last_hidden_state = outputs[-1]

        logits = calculate_logits(
            last_hidden_state,
            self.output_weights
        )

        probabilities = softmax_with_temperature(
            logits,
            temperature
        )

        next_character_probabilities = {
            self.id_to_character[i]: probabilities[i]
            for i in range(len(probabilities))
        }
        return {
            "tokens": tokens,
            "combined_embeddings": combined,
            "attention_weights": attention_weights,
            "attention_output": outputs,
            "logits": logits,
            "probabilities": next_character_probabilities,
            "temperature": temperature
        }