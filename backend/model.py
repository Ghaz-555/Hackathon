import numpy as np
from dataset import create_training_examples
from gradients import output_layer_gradients
from loss import cross_entropy_loss
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

    def calculate_loss(self, text, correct_character, temperature=1.0):
        result = self.forward(
            text,
            temperature=temperature
        )

        loss = cross_entropy_loss(
            result["probabilities"],
            correct_character
        )

        return loss

    def train_step(self, text, correct_character, learning_rate=0.1):
        # Forward pass
        result = self.forward(text, temperature=1.0)

        probabilities_dict = result["probabilities"]

        # Convert probability dictionary back into an array
        probabilities = np.array([
            probabilities_dict[self.id_to_character[i]]
            for i in range(len(self.vocabulary))
        ])

        correct_index = self.vocabulary[correct_character]

        # Final hidden state used for next-character prediction
        hidden_state = result["attention_output"][-1]

        # Calculate gradients
        d_logits, d_output_weights, d_hidden_state = output_layer_gradients(
            hidden_state,
            probabilities,
            correct_index,
            self.output_weights
        )

        # Update output weights
        self.output_weights -= learning_rate * d_output_weights

        # Calculate loss AFTER the update
        new_loss = self.calculate_loss(
            text,
            correct_character
        )

        return {
            "loss": new_loss,
            "d_logits": d_logits,
            "d_output_weights": d_output_weights,
            "d_hidden_state": d_hidden_state
        }
    
    
    def train_epoch(self, text, learning_rate=0.1):
        examples = create_training_examples(
            text,
            self.context_length
        )

        total_loss = 0.0

        for input_text, correct_character in examples:
            result = self.train_step(
                input_text,
                correct_character,
                learning_rate
            )

            total_loss += result["loss"]

        average_loss = total_loss / len(examples)

        return {
            "average_loss": average_loss,
            "examples_seen": len(examples)
        }
    
    
    def evaluate(self, text):
        examples = create_training_examples(
            text,
            self.context_length
        )

        total_loss = 0.0

        for input_text, correct_character in examples:
            loss = self.calculate_loss(
                input_text,
                correct_character
            )

            total_loss += loss

        average_loss = total_loss / len(examples)

        return {
            "average_loss": average_loss,
            "examples_seen": len(examples)
        }
    def generate(
        self,
        prompt,
        length=20,
        temperature=1.0,
        seed=42
    ):
        rng = np.random.default_rng(seed)

        generated_text = prompt

        characters = [
            self.id_to_character[i]
            for i in range(len(self.vocabulary))
        ]

        for _ in range(length):
            result = self.forward(
                generated_text,
                temperature=temperature
            )

            probabilities = np.array([
                result["probabilities"][character]
                for character in characters
            ])

            next_character = rng.choice(
                characters,
                p=probabilities
            )

            generated_text += next_character

        return generated_text