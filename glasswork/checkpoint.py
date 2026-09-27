"""Versioned NumPy archives: weights + tokenizer + Adam + training RNG.

No pickle or executable objects. Saves use a same-directory temporary file
and atomic replacement, so a failed write cannot truncate a good checkpoint.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .model import Config, TinyTransformer
from .tokenizer import CharacterTokenizer
from .training import Adam, AdamConfig, Trainer

FORMAT_VERSION = 1
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def save_checkpoint(path, trainer: Trainer, tokenizer: CharacterTokenizer, *, metadata=None):
    if len(tokenizer) != trainer.model.config.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    metadata = metadata or {}
    header = {
        "format_version": FORMAT_VERSION,
        "config": asdict(trainer.model.config), "characters": list(tokenizer.characters),
        "optimizer": asdict(trainer.optimizer.config), "step": trainer.optimizer.step_count,
        "rng_state": trainer.rng.bit_generator.state, "metadata": metadata,
    }
    # allow_nan=False prevents writing a checkpoint we cannot reliably validate.
    arrays = {"header": np.array(json.dumps(header, allow_nan=False))}
    for prefix, source in (("param", trainer.model.params), ("adam_m", trainer.optimizer.m),
                           ("adam_v", trainer.optimizer.v)):
        for name, values in source.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Cannot save non-finite array {prefix}/{name}")
            arrays[f"{prefix}/{name}"] = values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_checkpoint(path):
    """Return (trainer, tokenizer, metadata); validate before exposing any state."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Checkpoint is too large")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES
                or len(entries) != len({e.filename for e in entries})):
            raise ValueError("Invalid or oversized checkpoint archive")
    with np.load(path, allow_pickle=False) as arrays:
        header = json.loads(str(arrays["header"].item()))
        if header["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint version")
        model = TinyTransformer(Config(**header["config"]))
        tokenizer = CharacterTokenizer(header["characters"])
        if len(tokenizer) != model.config.vocab_size:
            raise ValueError("Tokenizer/model vocabulary mismatch")
        optimizer = Adam(model.params, AdamConfig(**header["optimizer"]))
        if type(header["step"]) is not int or header["step"] < 0:
            raise ValueError("Invalid optimizer step")
        optimizer.step_count = header["step"]
        expected = {"header"}
        for prefix, destination in (("param", model.params), ("adam_m", optimizer.m), ("adam_v", optimizer.v)):
            for name, target in destination.items():
                key = f"{prefix}/{name}"
                expected.add(key)
                source = arrays[key]
                if source.dtype != np.float64 or source.shape != target.shape or not np.isfinite(source).all():
                    raise ValueError(f"Invalid checkpoint array {key}")
                if prefix == "adam_v" and np.any(source < 0):
                    raise ValueError("Adam second moments cannot be negative")
                target[...] = source
        if set(arrays.files) != expected:
            raise ValueError("Unexpected checkpoint entries")
        trainer = Trainer(model, optimizer)
        trainer.rng.bit_generator.state = header["rng_state"]
        return trainer, tokenizer, header["metadata"]
