"""Train on original toy sentences; report held-out loss and save real artifacts."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from glasswork import CharacterTokenizer, Config, TinyTransformer, Trainer, save_checkpoint
from glasswork.training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.batch_size <= 1024:
        parser.error("steps must be positive and batch-size must be between 1 and 1024")
    train_text = (ROOT / "data/train.txt").read_text()
    validation_text = (ROOT / "data/validation.txt").read_text()
    # Build vocabulary ONLY from training text; held-out characters must be known.
    tokenizer = CharacterTokenizer.from_text(train_text)
    train, validation = tokenizer.encode(train_text), tokenizer.encode(validation_text)
    model = TinyTransformer(Config(vocab_size=len(tokenizer)))
    trainer = Trainer(model)
    initial = {"train": evaluate(model, train), "validation": evaluate(model, validation)}
    history = [{"step": 0, **initial}]
    print(f"Parameters: {model.parameter_count:,}; vocabulary: {len(tokenizer)}; NumPy {np.__version__}", flush=True)
    print(f"Initial loss: train={initial['train']:.4f}, validation={initial['validation']:.4f}", flush=True)
    start, step_times = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        tick = time.perf_counter()
        metrics = trainer.train_step(train, batch_size=args.batch_size)
        step_times.append(time.perf_counter() - tick)
        if step % 50 == 0 or step == args.steps:
            record = {"step": step, "train": evaluate(model, train), "validation": evaluate(model, validation),
                      "minibatch_loss": metrics["loss"], "gradient_norm": metrics["gradient_norm"],
                      "clip_scale": metrics["clip_scale"]}
            history.append(record)
            print(f"Step {step:4}: train={record['train']:.4f}, validation={record['validation']:.4f}", flush=True)
    elapsed = time.perf_counter() - start
    report = {
        "config": asdict(model.config), "parameters": model.parameter_count,
        "steps": args.steps, "batch_size": args.batch_size, "history": history,
        "elapsed_seconds": elapsed, "median_step_ms": float(np.median(step_times) * 1000),
        "tokens_per_second_training_only": args.steps * args.batch_size * model.config.context_length / sum(step_times),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "data_sha256": {"train": hashlib.sha256(train_text.encode()).hexdigest(),
                        "validation": hashlib.sha256(validation_text.encode()).hexdigest()},
        "notes": "Original narrow toy corpus. Held-out loss measures this corpus, not general language ability.",
    }
    for temperature in (0, .8):
        ids = model.generate(tokenizer.encode("the "), max_new_tokens=100, temperature=temperature)
        report[f"sample_temperature_{temperature}"] = tokenizer.decode(ids)
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output / "tiny-transformer.npz", trainer, tokenizer,
                    metadata={"dataset_hashes": report["data_sha256"], "training_steps": args.steps})
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved checkpoint and report to {args.output}; {elapsed:.2f}s; median step {report['median_step_ms']:.2f}ms")
    print("Greedy sample:", repr(report["sample_temperature_0"]))
    print("Sample at T=0.8:", repr(report["sample_temperature_0.8"]))


if __name__ == "__main__":
    main()
