"""Load the trained model, generate text, and export a visualization trace."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glasswork import load_checkpoint
from glasswork.model import trace_to_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/tiny-transformer.npz")
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", type=Path, default=ROOT / "artifacts/example-trace.json")
    args = parser.parse_args()
    trainer, tokenizer, _ = load_checkpoint(args.checkpoint)
    model = trainer.model
    ids = tokenizer.encode(args.prompt)
    result = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature, seed=args.seed)
    print(f"Loaded {model.parameter_count:,} parameters after {trainer.optimizer.step_count} updates.")
    print(tokenizer.decode(result))
    trace = model.forward(ids[-model.config.context_length:][None, :], capture_trace=True).trace
    payload = {"characters": list(tokenizer.characters), "trace": trace_to_json(trace)}
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(f"Actual computation trace saved to {args.trace}")


if __name__ == "__main__":
    main()
