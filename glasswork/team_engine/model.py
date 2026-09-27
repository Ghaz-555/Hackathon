import numpy as np
from .dataset import create_training_examples
from .gradients import output_layer_gradients
from .tokenizer import tokenize, build_vocabulary
from .embeddings import create_embeddings
from .positional import (
    create_positional_embeddings,
    add_position_embeddings
)
from .attention import causal_self_attention, create_matrix
from .output import (
    create_output_weights,
    calculate_logits,
    softmax_with_temperature
)


class GlassBoxModel:

    def __init__(self, text, embedding_size=4, context_length=128, seed=42):
        if not isinstance(text, str) or len(set(tokenize(text))) < 2:
            raise ValueError("Training text needs at least two distinct characters")
        if type(embedding_size) is not int or not 1 <= embedding_size <= 64:
            raise ValueError("embedding_size must be between 1 and 64")
        if type(context_length) is not int or not 1 <= context_length <= 128:
            raise ValueError("context_length must be between 1 and 128")
        self.seed = seed
        self.training_text = ''.join(tokenize(text))
        self.step_count = 0
        rng = np.random.RandomState(seed)
        self.embedding_size = embedding_size
        self.context_length = context_length

        self.W_q = create_matrix(embedding_size, rng=rng)
        self.W_k = create_matrix(embedding_size, rng=rng)
        self.W_v = create_matrix(embedding_size, rng=rng)

        # Build character vocabulary
        self.tokens = tokenize(text)
        self.vocabulary = build_vocabulary(self.tokens)

        # Character embeddings
        self.embeddings = create_embeddings(
            self.vocabulary,
            embedding_size, rng=rng
        )

        # Position embeddings
        self.positional_embeddings = create_positional_embeddings(
            context_length,
            embedding_size, rng=rng
        )
        # Output layer for next-character prediction
        self.output_weights = create_output_weights(
            embedding_size,
            len(self.vocabulary), rng=rng
        )

        # Reverse vocabulary lets us convert IDs back into characters
        self.id_to_character = {
            index: character
            for character, index in self.vocabulary.items()
        }

    def forward(self, text, temperature=1.0, capture_trace=False):
        tokens = self.validate_text(text)

        # Don't allow inputs longer than our context window
        tokens = tokens[-self.context_length:]

        # Combine character + position information
        combined = add_position_embeddings(
            tokens,
            self.embeddings,
            self.positional_embeddings
        )

        # Transformer attention
        attention_result = causal_self_attention(
            combined,
            self.W_q,
            self.W_k,
            self.W_v, capture_trace=capture_trace
        )
        outputs, attention_weights = attention_result[:2]



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
        result = {
            "tokens": tokens,
            "combined_embeddings": combined,
            "attention_weights": attention_weights,
            "attention_output": outputs,
            "logits": logits,
            "probabilities": next_character_probabilities,
            "temperature": temperature
        }
        if capture_trace:
            result["internals"] = {**attention_result[2],
                "token_embeddings": np.array([self.embeddings[c] for c in tokens]),
                "position_embeddings": self.positional_embeddings[:len(tokens)].copy(),
                "query_weights": self.W_q.copy(),
                "key_weights": self.W_k.copy(),
                "value_weights": self.W_v.copy(),
                "output_weights": self.output_weights.copy()}
        return result

    def calculate_loss(self, text, correct_character, temperature=1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Loss requires a finite positive temperature")
        target = self.validate_text(correct_character)
        if len(target) != 1:
            raise ValueError("Target must normalize to one character")
        result = self.forward(text, temperature=temperature)
        # Log-sum-exp avoids the source probability floor that capped loss at ~27.6.
        logits = result["logits"] / temperature
        shifted = logits - np.max(logits)
        return float(np.log(np.exp(shifted).sum()) - shifted[self.vocabulary[target[0]]])

    def train_step(self, text, correct_character, learning_rate=0.1):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        target = self.validate_text(correct_character)
        if len(target) != 1:
            raise ValueError("Target must normalize to one character")
        correct_character = target[0]
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
        updated = self.output_weights - learning_rate * d_output_weights
        if not np.isfinite(updated).all():
            raise FloatingPointError("Non-finite output update; weights were not changed")
        self.output_weights[...] = updated
        self.step_count += 1

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
        normalized = ''.join(self.validate_text(text))
        examples = create_training_examples(normalized, self.context_length)
        if not examples:
            raise ValueError("Training/evaluation needs at least two characters")

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
        normalized = ''.join(self.validate_text(text))
        examples = create_training_examples(normalized, self.context_length)
        if not examples:
            raise ValueError("Training/evaluation needs at least two characters")

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
        if type(length) is not int or not 0 <= length <= 4096:
            raise ValueError("Generation length must be between 0 and 4096")
        softmax_with_temperature(np.zeros(len(self.vocabulary)), temperature)
        rng = np.random.default_rng(seed)
        generated_text = ''.join(self.validate_text(prompt))

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

    def validate_text(self, text):
        """Normalize once for prompts and targets; reject unknown characters clearly."""
        if not isinstance(text, str) or not text:
            raise ValueError("Text cannot be empty")
        tokens = tokenize(text)
        unknown = sorted(set(tokens) - self.vocabulary.keys())
        if unknown:
            raise ValueError(f"Unsupported characters: {unknown!r}")
        return tokens

    @property
    def params(self):
        """Actual parameter arrays, including frozen feature-extraction weights."""
        parameters = {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v,
                      "positions": self.positional_embeddings, "output_weights": self.output_weights}
        parameters.update({f"embedding.{index}": self.embeddings[character]
                           for character, index in self.vocabulary.items()})
        return parameters
