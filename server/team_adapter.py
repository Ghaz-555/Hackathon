"""Translate the team's GlassBoxModel into the stage-2 lesson contract.

This layer owns session training selection, persistence, and JSON shapes. Model
forward passes and SGD updates run through the copied team implementation.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from glasswork.team_engine import GlassBoxModel
from glasswork.team_engine.dataset import create_training_examples

TRAIN_TEXT = 'the cat sat on the mat'
VALIDATION_TEXT = 'the mat sat on the cat'
ENGINE_ID = 'team-glassbox-v1'
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / 'artifacts/team-glassbox.npz'


class TeamEngine:
    def __init__(self, model=None, *, learning_rate=.1):
        self.model = model or GlassBoxModel(TRAIN_TEXT, embedding_size=4, context_length=128, seed=42)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Learning rate must be positive and finite')
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(123)
        self.examples = create_training_examples(self.model.training_text, self.model.context_length)
        if not self.examples:
            raise ValueError('Training corpus needs at least two characters')
        self.model.validate_text(VALIDATION_TEXT)

    @property
    def characters(self):
        return [self.model.id_to_character[i] for i in range(len(self.model.vocabulary))]

    @property
    def step(self):
        return self.model.step_count

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in self.model.params.items():
            digest.update(name.encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def metadata(self):
        m = self.model
        return {'engine_id': ENGINE_ID, 'model_name': 'GlassBox Attention',
                'parameter_count': sum(p.size for p in m.params.values()),
                'trainable_parameter_count': m.output_weights.size,
                'training_scope': 'output_projection_only', 'optimizer': 'SGD',
                'learning_rate': self.learning_rate, 'examples_per_step': 1,
                'training_text': m.training_text, 'validation_text': VALIDATION_TEXT,
                'config': {'context_length': m.context_length, 'd_model': m.embedding_size,
                           'n_layers': 1, 'n_heads': 1, 'vocab_size': len(m.vocabulary)},
                'characters': self.characters, 'step': self.step, 'fingerprint': self.fingerprint()}

    def inspect(self, prompt, temperature):
        normalized = ''.join(self.model.validate_text(prompt))
        if len(normalized) > self.model.context_length:
            raise ValueError(f'Prompt must fit within {self.model.context_length} normalized characters')
        result = self.model.forward(normalized, temperature=temperature, capture_trace=True)
        probabilities = np.array([result['probabilities'][c] for c in self.characters])
        positive = probabilities[probabilities > 0]
        # Source attention rows have lengths 1..T. Pad only the masked future cells
        # with zeros, preserving every original visible weight in [B,H,T,T] form.
        t = len(result['tokens'])
        attention = np.zeros((t, t), dtype=np.float64)
        for r, row in enumerate(result['attention_weights']):
            attention[r, :r+1] = row
        trace = {
            'schema_version': 4, 'engine_id': ENGINE_ID,
            'input_ids': [[self.model.vocabulary[c] for c in result['tokens']]],
            'tokens': result['tokens'], 'causal_mask': np.tril(np.ones((t,t), dtype=bool)).tolist(),
            'embedding_sum': result['combined_embeddings'][None, ...].tolist(),
            **{name: value.tolist() for name, value in result['internals'].items()},
            'blocks': [{'kind': 'single_head_attention', 'attention': attention[None,None,...].tolist(),
                        'head_outputs': result['attention_output'][None,None,...].tolist()}],
            'next_token_logits': result['logits'].tolist(),
            'next_token_probabilities': probabilities.tolist(),
        }
        return {'prompt': normalized, 'temperature': temperature, 'logits': result['logits'].tolist(),
                'probabilities': probabilities.tolist(), 'entropy': float(-np.sum(positive*np.log(positive))),
                'trace': trace}

    def train_step(self):
        # One displayed optimizer step is exactly one team train_step, not an epoch.
        prompt, target = self.examples[int(self.rng.integers(len(self.examples)))]
        result = self.model.train_step(prompt, target, self.learning_rate)
        return {'example_loss_after': result['loss'], 'training_prompt': prompt, 'target': target}

    def metrics(self):
        return {'step': self.step, 'train': self.model.evaluate(self.model.training_text)['average_loss'],
                'validation': self.model.evaluate(VALIDATION_TEXT)['average_loss']}

    def generate(self, prompt, count, temperature, seed):
        # Shared validation makes display, generation and training use one vocabulary.
        self.inspect(prompt, temperature)
        return self.model.generate(prompt, length=count, temperature=temperature, seed=seed)

    def reset(self):
        return TeamEngine(GlassBoxModel(self.model.training_text, self.model.embedding_size,
                                        self.model.context_length, self.model.seed), learning_rate=self.learning_rate)

    def save(self, path):
        header = {'version': 1, 'engine_id': ENGINE_ID, 'training_text': self.model.training_text,
                  'embedding_size': self.model.embedding_size, 'context_length': self.model.context_length,
                  'seed': self.model.seed, 'step': self.step, 'learning_rate': self.learning_rate,
                  'characters': self.characters, 'rng': self.rng.bit_generator.state}
        arrays = dict(self.model.params)
        if not all(np.isfinite(p).all() for p in arrays.values()):
            raise ValueError('Cannot save non-finite weights')
        arrays['header'] = np.array(json.dumps(header, allow_nan=False))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
                tmp = Path(file.name)
                np.savez_compressed(file, **arrays)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, path)
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path=DEFAULT_CHECKPOINT):
        with zipfile.ZipFile(path) as archive:
            items = archive.infolist()
            if (sum(item.file_size for item in items) > 16 * 1024 * 1024
                    or len(items) != len({item.filename for item in items})):
                raise ValueError('Invalid checkpoint size or duplicate entries')
        with np.load(path, allow_pickle=False) as arrays:
            header = json.loads(str(arrays['header'].item()))
            if header['version'] != 1 or header['engine_id'] != ENGINE_ID:
                raise ValueError('Wrong engine or checkpoint version')
            model = GlassBoxModel(header['training_text'], header['embedding_size'], header['context_length'], header['seed'])
            engine = cls(model, learning_rate=header['learning_rate'])
            if engine.characters != header['characters']:
                raise ValueError('Checkpoint vocabulary mismatch')
            if set(arrays.files) != set(model.params) | {'header'}:
                raise ValueError('Checkpoint parameter names do not match')
            for name, target in model.params.items():
                source = arrays[name]
                if source.dtype != np.float64 or source.shape != target.shape or not np.isfinite(source).all():
                    raise ValueError(f'Invalid parameter: {name}')
                target[...] = source
            if type(header['step']) is not int or header['step'] < 0:
                raise ValueError('Invalid training step')
            model.step_count = header['step']
            engine.rng.bit_generator.state = header['rng']
        return engine
