# Glasswork — project overview

## Goal

Glasswork is an interactive learning lab that makes language models easier to
understand. Learners experiment with a small attention model built from scratch by the team and
see how its real predictions and internal calculations change. The project is
designed for the MathemaTech educational challenge.

## What learners can do

- Explore six model operations in 3D, follow a character, and inspect the actual
  vectors, attention weights, equations and probabilities behind a prediction.
- Change **temperature** and compare next-character probabilities to learn how
  sampling changes without changing the model's weights.
- Train the model from random weights and compare its predictions before and
  after learning.
- Compare training loss with held-out loss to see how practice and generalization
  can differ.
- Inspect attention weights and generate text with a repeatable sampling seed.

## Simple architecture

```text
React + TypeScript website
  ├─ Recharts: training and held-out loss graphs
  ├─ Three.js: interactive 3D model and animated forward-pass walkthrough
  └─ Stage inspector, prompts, sampling and learning drawer
             │ HTTP / JSON
             ▼
FastAPI server
  ├─ Creates isolated learner sessions
  ├─ Validates requests and serializes model updates
  └─ Returns real model predictions, traces and training metrics
             │
             ▼
Python + NumPy team attention model
  ├─ Handwritten causal attention and output-layer gradients
  ├─ SGD output-layer training and text generation
  └─ Saved weights, vocabulary and training state
```

The model math is implemented with NumPy. Training currently updates the output
projection while embeddings and attention weights stay fixed. The website uses standard UI,
charting, and web-server libraries. Visualizations use values captured from the
model's actual computations.

## Current scope

The active team model has 640 parameters, of which 40 output weights are trained.
It has one attention head, four-dimensional embeddings, and a 128-character context.
It learns from “the cat sat on the mat” and evaluates on a separate sentence. It is an educational example, not a general-purpose chatbot. The current
site includes interactive 3D exploration, temperature experiments, and live
training. Its spatial layout is an educational diagram; its numerical values come
from the actual model. Gemini-powered explanations remain a future stage.

## Run locally

From the `GlassworkV1` project directory:

```bash
.venv/bin/python scripts/serve.py
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001). Setup and implementation
details are in the [team engine integration guide](TEAM_ENGINE_INTEGRATION.md)
and the [stage 3 guide](STAGE3.md).
