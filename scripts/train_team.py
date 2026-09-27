"""Build a reproducible checkpoint using the team's output-projection SGD engine."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.team_adapter import TeamEngine, DEFAULT_CHECKPOINT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    engine = TeamEngine()
    history = [engine.metrics()]
    start = time.perf_counter()
    for step in range(args.steps):
        engine.train_step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            history.append(engine.metrics())
    engine.save(DEFAULT_CHECKPOINT)
    report = {'engine': engine.metadata(), 'history': history, 'elapsed_seconds': time.perf_counter()-start,
              'training_scope': 'Only output_weights are updated; embeddings and Q/K/V stay fixed.',
              'sample': engine.generate('the ', 64, .8, 7)}
    (ROOT/'artifacts/team-training-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'parameters':report['engine']['parameter_count'], 'trainable':report['engine']['trainable_parameter_count'],
                      'initial':history[0], 'final':history[-1], 'sample':report['sample']}, indent=2))
