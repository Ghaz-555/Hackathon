import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { EmbeddingSpace, type Manifest } from "../src/brain/math.ts";
const dir = new URL("../public/embeddings/", import.meta.url);
const manifest = JSON.parse(
  readFileSync(new URL("manifest.json", dir), "utf8"),
) as Manifest;
const bytes = readFileSync(new URL("vectors.f32", dir));
const vectors = new Float32Array(
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
);
const space = new EmbeddingSpace(manifest, vectors),
  index = (s: string) => space.byLabel.get(s)!;
const close = (a: number, b: number, t = 1e-6) =>
  assert.ok(Math.abs(a - b) < t, `${a} != ${b}`);
test("export shape and checksum", () => {
  assert.equal(manifest.count, 2048);
  assert.equal(manifest.dimensions, 768);
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    manifest.sha256,
  );
});
test("PCA matches every exported position", () => {
  manifest.tokens.forEach((t, i) =>
    space
      .project(space.vector(i))
      .forEach((v, j) => close(v, t.position[j], 1e-5)),
  );
});
test("affine arithmetic matches projected displacement", () => {
  const [a, b, c] = ["king", "man", "woman"].map(index);
  space
    .project(space.analogy(a, b, c))
    .forEach((v, i) =>
      close(
        v,
        manifest.tokens[a].position[i] -
          manifest.tokens[b].position[i] +
          manifest.tokens[c].position[i],
      ),
    );
});
test("self cosine is one and exclusions apply", () => {
  const i = index("king");
  close(space.nearest(space.vector(i), new Set(), 1)[0].score, 1);
  assert.ok(
    space.nearest(space.vector(i), new Set([i])).every((n) => n.index !== i),
  );
});
test("analogy matches independent NumPy ranking", () => {
  const [a, b, c] = ["king", "man", "woman"].map(index);
  space
    .nearest(space.analogy(a, b, c), new Set([a, b, c]), 10)
    .forEach((n, i) => {
      assert.equal(
        manifest.tokens[n.index].label,
        manifest.referenceAnalogy.neighbors[i].label,
      );
      close(n.score, manifest.referenceAnalogy.neighbors[i].cosine);
    });
});
test("cancellation returns the remaining vector", () => {
  const a = index("king"),
    c = index("cat");
  space.analogy(a, a, c).forEach((v, i) => close(v, space.vector(c)[i]));
});
test("invalid vectors fail", () => {
  assert.throws(() => space.nearest(new Float64Array(768)), /zero/);
  assert.throws(
    () => new EmbeddingSpace(manifest, new Float32Array(7)),
    /do not match/,
  );
});
test("PCA basis is orthonormal", () => {
  const b = manifest.projection.basis;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      close(
        b[i].reduce((s, v, k) => s + v * b[j][k], 0),
        i === j ? 1 : 0,
      );
});

test("weighted sums project from the projected zero, not the plot origin", () => {
  const ids = ["king", "man", "woman"].map(index),
    scales = [2, -0.5, 0.25];
  const query = space.combination(ids, scales);
  query.forEach((v, i) =>
    close(
      v,
      ids.reduce((sum, id, j) => sum + scales[j] * space.vector(id)[i], 0),
    ),
  );
  const zero = space.project(new Float64Array(768));
  space
    .project(query)
    .forEach((v, i) =>
      close(
        v,
        zero[i] +
          ids.reduce(
            (sum, id, j) =>
              sum + scales[j] * (manifest.tokens[id].position[i] - zero[i]),
            0,
          ),
        1e-5,
      ),
    );
});
test("weighted sums reject invalid input and cancellation has no cosine direction", () => {
  assert.throws(() => space.combination([0], [Infinity]), /Invalid/);
  assert.throws(() => space.combination([-1], [1]), /Invalid/);
  assert.throws(() => space.combination([0, 1], [1]), /Invalid/);
  assert.throws(
    () => space.nearest(space.combination([0, 0], [1, -1])),
    /zero/,
  );
  space
    .combination([0, 1], [0, 1])
    .forEach((v, i) => close(v, space.vector(1)[i]));
});
