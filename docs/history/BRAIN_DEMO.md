> Historical standalone-demo record. For the integrated branch, follow the root README and docs/MAIN2_INTEGRATION.md.

# The brain of an LLM

A standalone Three.js explorer of **real GPT-2 input embeddings**, added to the
original Hackathon frontend at `/brain.html`. It is separate from the tiny NumPy
model that learners train in GlassworkV1. No GPT-2 inference or training runs in
this demo; it explores exported pretrained vectors.

## Run

```bash
cd /home/sawoo/Downloads/Hackathon/frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5174/brain.html**. The exported data is included, so opening
the demo needs no model download, Python server, Gemini key or other external API.
If a review server is already using that port, use its existing page.

Production build: `npm run build`. Local production preview: `npm run preview`.
The empty pre-existing frontend entry point and components are unchanged; visit
`/brain.html`, not `/`.

## Try it

1. Search for `king`, `woman`, `cat`, `computer`, or another suggested token.
2. Read its eight nearest neighbors. Click one to move through the vocabulary.
3. Run `king − man + woman`. The current exported data ranks `queen` first with
   cosine similarity approximately **0.7085**. This is computed, not a hardcoded
   answer. Results are ranked only among the 2,048 exported candidates.
4. Edit A, B and C. The browser recomputes the full 768-dimensional query and its
   nearest matches. The three inputs are omitted from the result list.
5. Compare the B → C arrow to the same displacement starting at A. The resulting
   point is the projected query; the dotted segment leads to the nearest token.
6. Drag to orbit, scroll to zoom, or focus the selected token. Change depth and
   thickness to view a band of the third principal component.

Selected tokens and neighbors stay visible when slicing. Sparse labels reduce
clutter. Reduce motion stops automatic rotation and the arithmetic animation.
A WebGL failure leaves numerical search and arithmetic available in the inspector.

## What the math means

The source is the `wte.weight` input embedding matrix from
[`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e),
pinned at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The original
[GPT-2 license](https://github.com/openai/gpt-2/blob/master/LICENSE) is included
beside the data. The full matrix has 50,257 rows and 768 dimensions.

The export takes preset educational examples, then leading-space alphabetic
tokens in token-ID order until it contains 2,048 tokens. The preset examples get
no special coordinates or ranking advantage. Some alphabetic tokens are word
fragments. Uppercase and lowercase tokens are distinct.

The preparation script centers those raw vectors and computes three principal
components with NumPy. Together they retain approximately **5.97%** of the variance
in this subset. This is a lossy projection, not a faithful map of all semantic
distances. The origin of the scene is the subset mean. The colors follow a
projected coordinate; they do not represent labeled semantic categories.

For any query vector `q`, the browser ranks candidates using:

```text
cosine(q, token) = dot(q, token) / (norm(q) * norm(token))
q = vector(A) - vector(B) + vector(C)
position(q) = (q - subset_mean) @ PCA_basis * display_scale
```

Ranking uses all 768 original coordinates. Because the coefficients in A−B+C sum
to one, its projected position also equals `position(A)-position(B)+position(C)`.
That is why the transferred arrow is mathematically consistent with the query.
Nearest-match analogies are approximate observations, not exact equalities or
universal laws about language. These are static input embeddings, not contextual
activations or a complete picture of an LLM's internal computation.

## Files

- `frontend/scripts/prepare_embeddings.py`: downloads only the safetensors header
  and input-embedding byte range, then exports the subset and PCA metadata.
- `frontend/public/embeddings/manifest.json`: token IDs, labels, projection,
  source revision, reference results and checksum.
- `frontend/public/embeddings/vectors.f32`: about 6.3 MB of raw float32 vectors.
- `frontend/src/brain/math.ts`: data loading, validation, cosine search, arithmetic
  and projection. This module has no UI dependency.
- `frontend/src/brain/scene.ts`: Three.js scene, labels, camera and arrows.
- `frontend/src/brain/main.ts`: page structure and event handlers.
- `frontend/src/brain/style.css`: layout and styling.
- `frontend/tests/math.test.ts`: independent export/reference and math checks.
- `frontend/scripts/test_browser.py`: real WebGL/browser interaction checks.

The full 154 MB embedding download is cached under `frontend/.cache/`, which is
ignored by Git. Regenerate from the repository root with a Python environment
containing NumPy:

```bash
/home/sawoo/GlassworkV1/.venv/bin/python frontend/scripts/prepare_embeddings.py
```

That preparation step needs network access; the running demo does not.

## Verification

From `frontend/`:

```bash
npm test
npm run build
```

With the demo server running and Playwright/Chromium installed:

```bash
/home/sawoo/playwright-env/bin/python scripts/test_browser.py
```

Eight numerical tests cover export provenance/checksum, PCA reconstruction for
all tokens, affine arithmetic, cosine self-similarity, exclusion rules, independent
NumPy analogy results, malformed data and orthonormal PCA directions. Browser
checks cover the scene, editable arithmetic, search, slicing, camera controls,
mobile layout and fallback. Logs and screenshots are local `frontend/artifacts/`
outputs and are ignored by Git.

## Future integration

The demo currently lives only in the original Hackathon project, as requested.
To connect it to GlassworkV1 later, `math.ts` can be reused directly, and a React
wrapper can own a `BrainScene` instance and call `dispose()` on unmount. The data
files must accompany the imported view. Keep GPT-2's pretrained embedding
explorer labeled separately from the four-dimensional team model that learners
train. No source files need to be represented as having a different author.

For the detailed change inventory and manual Git commands, see
[WORKLOG_AND_COMMITS.md](WORKLOG_AND_COMMITS.md).
