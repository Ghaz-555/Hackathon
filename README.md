# Hackathon
Project for hack the hill III 2026.
## Goal

Glasswork is basically an interactive learning lab, that makes language models easier to understand. Our users experiment with a small transformer built from scratch and see how its real predictions and internal calculations change.

## What learners can do?
- Change **Temperature** and compare next-character probabilties to learn how sampling changes without chaninging hte model's weights.
- Train the model from random weights and compare its predictions before and after learning
- Compare training loss with held-out loss to see how practice and generalization can differ.
- Inspect attention and weights and generate text with a repeatble sampling seed.

## Simple Overview of the architecture
1. The website UI/UX (React + Typescript)
    - Recharts: training and held-out loss graphs
    - Live probability bars and attention heatmap
    - Lesson controls, prompts and generated text
2. FastAPI server
    - Creates isolated learner sessions
    - Validates requests and serializes model updates
    - return real model predictions, traces and training matricies
3. Backend (Python + Numpy transformer)
    - Forward pass modules and backpropagation functions
    - Model weights, tokenizer, and training state are maintained for each learner session.


## current scope
The model is intentionally small and learns from a character-level dataset so learners can inspect its behavior and training in real time. It is an educational example, NOT A GENERAL-PURPOSE CHATBOT. The site includes temperature and learning lessons. Its architecture illustration is conceptual. Interactive 3D model exploration and GeminiAPI-powered explanations are planned for later stages.

