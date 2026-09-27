# Original educational corpus

`train.txt` and `validation.txt` were authored for GlassworkV1 during this build.
They were not downloaded from an existing language-model dataset. The sentences
describe a deliberately small world of colors, animals and places.

The files are separate before tokenization or training. Vocabulary is derived
only from `train.txt`; validation contains no unknown characters. Validation
sentences are distinct from training sentences but reuse the same grammar and
words. This is a compositional toy split, not evidence of general language skill.

Training samples contiguous character windows within the training file. Windows
may span a newline between sentences, but never cross into validation. Evaluation
uses deterministic non-overlapping target chunks and includes each next-character
target once, weighted by token count. The final partial chunk is included.

Training reports include SHA-256 hashes of both files to identify the exact data
used by a checkpoint. A resumed run must deliberately supply the same stream and
batch size to reproduce uninterrupted training.
