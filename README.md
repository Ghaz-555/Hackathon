# GlassBox

**See inside a language model.**

GlassBox is an interactive learning lab built for Hack the Hill III 2026. It makes language models easier to understand by letting users experiment with a small transformer-style model built from scratch and see how its predictions and internal calculations change in real time.

## What learners can do

- Change **temperature** and compare next-character probabilities to see how sampling changes without changing the model's weights.
- Train the model and observe how learning changes its predictions.
- Compare **training loss** with **held-out loss** to explore the difference between learning and generalization.
- Visualize **attention weights** to see how characters attend to earlier characters.
- Generate text using a repeatable sampling seed and experiment with different temperatures.

## Architecture

### 1. Frontend — React + TypeScript
- Interactive learning dashboard
- Live next-character probability visualization
- Attention visualization
- Training and held-out loss graphs using Recharts
- Temperature, training, prompt, seed, and generation controls

### 2. API — FastAPI
- Creates isolated learner sessions
- Connects the frontend to the model
- Validates requests
- Returns predictions, probabilities, attention weights, training metrics, and generated text

### 3. Model — Python + NumPy
- Character-level tokenization
- Token and positional embeddings
- Causal self-attention
- Next-character prediction
- Temperature-based softmax sampling
- Cross-entropy loss
- Gradient-based output-layer training
- Seeded text generation

## Why GlassBox?

Large language models can feel like black boxes. GlassBox turns some of their core ideas into experiments that learners can interact with directly.

Instead of only reading about concepts like attention, probability distributions, temperature, and training, users can change them and immediately see what happens.

## Current Scope

GlassBox intentionally uses a very small character-level, transformer-style model so its behavior can be inspected and updated quickly in real time.

It is an educational demonstration, not a general-purpose chatbot. The model is intentionally simplified, and training currently updates the output layer rather than every parameter in the model.

## Built With

- Python
- NumPy
- FastAPI
- React
- TypeScript
- Recharts

## Future Improvements

- Interactive 3D model architecture exploration
- Gemini-powered explanations and learning assistance
- Larger datasets and additional model experiments