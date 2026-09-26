import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import GlassBoxModel


app = FastAPI(title="GlassBox API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_text = "the cat sat on the mat"

sessions = {}


def create_model():
    return GlassBoxModel(
        training_text,
        embedding_size=4,
        context_length=128
    )

def get_model(session_id):
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return sessions[session_id]


def validate_text(model, text):
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    unknown = [
        character
        for character in text.lower()
        if character not in model.vocabulary
    ]

    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported characters: {sorted(set(unknown))}"
        )


class PredictRequest(BaseModel):
    session_id: str
    text: str
    temperature: float = 1.0


class GenerateRequest(BaseModel):
    session_id: str
    prompt: str
    length: int = 20
    temperature: float = 1.0
    seed: int = 42


class TrainRequest(BaseModel):
    session_id: str
    text: str
    epochs: int = 1
    learning_rate: float = 0.1


@app.get("/")
def root():
    return {
        "message": "GlassBox API is running"
    }


@app.post("/predict")
def predict(request: PredictRequest):
    model = get_model(request.session_id)
    validate_text(model, request.text)
    result = model.forward(
        request.text,
        temperature=request.temperature
    )

    return {
        "tokens": result["tokens"],
        "probabilities": result["probabilities"],
        "temperature": result["temperature"],
        "attention_weights": [
            weights.tolist()
            for weights in result["attention_weights"]
        ]
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    model = get_model(request.session_id)
    validate_text(model, request.prompt)
    generated_text = model.generate(
        request.prompt,
        length=request.length,
        temperature=request.temperature,
        seed=request.seed
    )

    return {
        "generated_text": generated_text
    }

@app.post("/train")
def train(request: TrainRequest):
    model = get_model(request.session_id)
    validate_text(model, request.text)
    training_losses = []
    held_out_losses = []

    held_out_text = "the mat sat on the cat"

    for _ in range(request.epochs):
        train_result = model.train_epoch(
            request.text,
            learning_rate=request.learning_rate
        )

        held_out_result = model.evaluate(
            held_out_text
        )

        training_losses.append(
            train_result["average_loss"]
        )

        held_out_losses.append(
            held_out_result["average_loss"]
        )

    return {
        "training_losses": training_losses,
        "held_out_losses": held_out_losses,
        "epochs": request.epochs
    }

@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())

    sessions[session_id] = create_model()

    return {
        "session_id": session_id
    }