"""Connect the existing manually differentiated NumPy transformer to the lab."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from glasswork.checkpoint import load_checkpoint
from glasswork.model import TinyTransformer, distribution, trace_to_json
from glasswork.training import Trainer, evaluate

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'artifacts/tiny-transformer.npz'
ENGINE_ID = 'numpy-transformer-28k'


class TransformerEngine:
    def __init__(self, trainer, tokenizer):
        self.trainer, self.tokenizer = trainer, tokenizer
        self.model = trainer.model
        self.train_text = (ROOT / 'data/train.txt').read_text()
        self.validation_text = (ROOT / 'data/validation.txt').read_text()
        self.train_tokens = tokenizer.encode(self.train_text)
        self.validation_tokens = tokenizer.encode(self.validation_text)

    @classmethod
    def load(cls, path=CHECKPOINT):
        trainer, tokenizer, _ = load_checkpoint(path)
        return cls(trainer, tokenizer)

    @property
    def step(self):
        return self.trainer.optimizer.step_count

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in self.model.params.items():
            digest.update(name.encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def metadata(self):
        return dict(engine_id=ENGINE_ID, model_name='Glasswork Transformer',
                    parameter_count=self.model.parameter_count,
                    trainable_parameter_count=self.model.parameter_count,
                    training_scope='all_parameters', optimizer='Adam',
                    learning_rate=self.trainer.optimizer.config.learning_rate,
                    examples_per_step=8, training_text=self.train_text,
                    validation_text=self.validation_text,
                    config=asdict(self.model.config), characters=list(self.tokenizer.characters),
                    step=self.step, fingerprint=self.fingerprint())

    def inspect(self, prompt, temperature):
        prompt = prompt.lower()
        if not 1 <= len(prompt) <= self.model.config.context_length:
            raise ValueError(f'Use 1–{self.model.config.context_length} characters for this model.')
        ids = self.tokenizer.encode(prompt)
        result = self.model.forward(ids[None, :], capture_trace=True)
        trace = trace_to_json(result.trace)
        params, width = self.model.params, self.model.config.d_model
        # Preserve every layer/head. The client chooses which one to inspect.
        for i, block in enumerate(trace['blocks']):
            prefix = f'blocks.{i}'
            weights = params[prefix + '.qkv.weight']
            for j, name in enumerate(('query_weights', 'key_weights', 'value_weights')):
                block[name] = weights[:, j*width:(j+1)*width].tolist()
            for name in ('qkv.bias', 'attention_output.weight', 'attention_output.bias',
                         'ff_in.weight', 'ff_in.bias', 'ff_out.weight', 'ff_out.bias'):
                block[name.replace('.', '_')] = params[prefix+'.'+name].tolist()
        block = trace['blocks'][0]
        trace.update(schema_version=5, engine_id=ENGINE_ID, tokens=list(prompt),
                     token_embeddings=trace['token_embeddings'][0],
                     queries=block['query'][0][0], keys=block['key'][0][0], values=block['value'][0][0],
                     query_weights=block['query_weights'], key_weights=block['key_weights'], value_weights=block['value_weights'],
                     output_weights=params['lm_head.weight'].tolist(),
                     output_bias=params['lm_head.bias'].tolist(), final_hidden=trace['final_normalized'][0][-1])
        logits = result.logits[0, -1]
        probabilities = distribution(logits, temperature)
        positive = probabilities[probabilities > 0]
        return dict(prompt=prompt, temperature=temperature, logits=logits.tolist(),
                    probabilities=probabilities.tolist(), entropy=float(-np.sum(positive*np.log(positive))), trace=trace)

    def train_step(self):
        return self.trainer.train_step(self.train_tokens, batch_size=8)

    def metrics(self):
        return dict(step=self.step, train=evaluate(self.model, self.train_tokens),
                    validation=evaluate(self.model, self.validation_tokens))

    def generate(self, prompt, count, temperature, seed):
        self.inspect(prompt, temperature)
        return self.tokenizer.decode(self.model.generate(self.tokenizer.encode(prompt.lower()),
                   max_new_tokens=count, temperature=temperature, seed=seed))

    def reset(self):
        return TransformerEngine(Trainer(TinyTransformer(self.model.config)), self.tokenizer)
