"""Session-isolated lesson API. NumPy work runs in FastAPI's sync thread pool.

Each session has a lock; inference, reset and training on that model serialize.
Mutations require the displayed revision, preventing accidental stale updates.
Use a single Uvicorn process: these short-lived sessions live in process memory.
"""
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import monotonic
import secrets
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .team_adapter import TeamEngine, DEFAULT_CHECKPOINT, ENGINE_ID

ROOT = Path(__file__).resolve().parents[1]


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Inspect(Request):
    prompt: str = Field(default='the ', min_length=1, max_length=128)
    temperature: float = Field(default=1, ge=0, le=2)


class Sample(Inspect):
    seed: int = Field(default=7, ge=0, le=2**32-1)
    count: int = Field(default=64, ge=1, le=80)


class Train(Inspect):
    revision: int = Field(ge=0)
    steps: int = Field(default=1, ge=1, le=5)


class Reset(Request):
    revision: int = Field(ge=0)
    mode: Literal['trained', 'random']


@dataclass
class Session:
    engine: TeamEngine
    mode: str = 'trained'
    revision: int = 0
    lock: Lock = field(default_factory=Lock)
    touched: float = field(default_factory=monotonic)
    history: list = field(default_factory=list)


def create_app(checkpoint=DEFAULT_CHECKPOINT, max_sessions=16, ttl=3600):
    app = FastAPI(title='Glasswork lesson API', version='0.2.0')
    sessions, registry_lock = {}, Lock()

    def get_session(sid):
        with registry_lock:
            session = sessions.get(sid)
            if not session or monotonic() - session.touched > ttl:
                sessions.pop(sid, None)
                raise HTTPException(404, 'Your lab session expired. Start a new session to continue.')
            session.touched = monotonic()
            return session

    def metrics(session):
        return session.engine.metrics()

    def state(session):
        return {'revision': session.revision, 'mode': session.mode,
                **session.engine.metadata(), 'history': [dict(point) for point in session.history]}

    def inspect(session, request):
        try:
            result = session.engine.inspect(request.prompt, request.temperature)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {**result, **state(session)}

    def require_revision(session, revision):
        if revision != session.revision:
            raise HTTPException(409, 'The model changed. Refresh the experiment before changing it again.')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'engine': 'numpy', 'implementation': ENGINE_ID, 'checkpoint_available': Path(checkpoint).is_file()}

    @app.post('/api/sessions', status_code=201)
    def create_session():
        # Registry lock also makes capacity checks atomic under simultaneous arrivals.
        with registry_lock:
            for sid in list(sessions):
                if monotonic() - sessions[sid].touched > ttl:
                    del sessions[sid]
            if len(sessions) >= max_sessions:
                raise HTTPException(503, 'The lab is full. Close an unused session or try again later.')
            try:
                engine = TeamEngine.load(checkpoint)
            except (OSError, ValueError) as error:
                raise HTTPException(503, 'The trained checkpoint is unavailable. Run scripts/train_team.py first.') from error
            session = Session(engine)
            session.history = [metrics(session)]
            sid = secrets.token_urlsafe(24)
            sessions[sid] = session
            return {'session_id': sid, **state(session)}

    @app.get('/api/sessions/{sid}')
    def session_state(sid: str):
        session = get_session(sid)
        with session.lock:
            return state(session)

    @app.delete('/api/sessions/{sid}', status_code=204)
    def delete_session(sid: str):
        with registry_lock:
            sessions.pop(sid, None)

    @app.post('/api/sessions/{sid}/inspect')
    def inspect_model(sid: str, request: Inspect):
        session = get_session(sid)
        with session.lock:
            return inspect(session, request)

    @app.post('/api/sessions/{sid}/sample')
    def sample(sid: str, request: Sample):
        session = get_session(sid)
        with session.lock:
            try:
                output = session.engine.generate(request.prompt, request.count, request.temperature, request.seed)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            return {'text': output, 'prompt': request.prompt.lower(),
                    'seed': request.seed, 'temperature': request.temperature, **state(session)}

    @app.post('/api/sessions/{sid}/train')
    def train(sid: str, request: Train):
        session = get_session(sid)
        with session.lock:
            require_revision(session, request.revision)
            before = inspect(session, request)  # Validate BEFORE changing any weights.
            updates = []
            # Consume the revision before updates, even if a later operation fails.
            session.revision += 1
            for _ in range(request.steps):
                update = session.engine.train_step()
                point = metrics(session)
                session.history.append(point)
                updates.append({**update, **point})
            session.history = session.history[-501:]
            return {'before': {'probabilities': before['probabilities'], 'fingerprint': before['fingerprint'],
                               'step': before['step']}, 'updates': updates, 'after': inspect(session, request)}

    @app.post('/api/sessions/{sid}/reset')
    def reset(sid: str, request: Reset):
        session = get_session(sid)
        with session.lock:
            require_revision(session, request.revision)
            if request.mode == 'random':
                session.engine = session.engine.reset()
            else:
                session.engine = TeamEngine.load(checkpoint)
            session.mode = request.mode
            session.revision += 1
            session.history = [metrics(session)]
            return state(session)

    # One origin in production; Vite proxies /api during development. No broad CORS.
    dist = ROOT / 'frontend/dist'
    if dist.is_dir():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')
        if (dist / 'embeddings').is_dir():
            app.mount('/embeddings', StaticFiles(directory=dist / 'embeddings'), name='embeddings')

        @app.get('/brain.html', include_in_schema=False)
        def brain():
            return FileResponse(dist / 'brain.html')

        @app.get('/', include_in_schema=False)
        def index():
            return FileResponse(dist / 'index.html')

    return app


app = create_app()
