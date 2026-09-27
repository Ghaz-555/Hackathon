import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { EmbeddingSpace, type Manifest } from "../src/brain/math.ts";

const directory = new URL("../public/embeddings/", import.meta.url);
const manifest = JSON.parse(
  readFileSync(new URL("manifest.json", directory), "utf8"),
) as Manifest;
const bytes = readFileSync(new URL("vectors.f32", directory));
const vectors = new Float32Array(
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
);
const space = new EmbeddingSpace(manifest, vectors);
const index = (label: string) => space.byLabel.get(label)!;
const close = (a: number, b: number, tolerance = 1e-6) =>
  assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

test("export has the pinned GPT-2 shape, token IDs and checksum", () => {
  assert.equal(manifest.dimensions, 768);
  assert.equal(manifest.count, 2048);
  assert.equal(new Set(manifest.tokens.map((token) => token.id)).size, 2048);
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    manifest.sha256,
  );
  assert.equal(manifest.revision, "607a30d783dfa663caf39e06633721c8d4cfcd7e");
});
test("PCA coordinates reproduce the Python export for every token", () => {
  manifest.tokens.forEach((token, i) =>
    space
      .project(space.vector(i))
      .forEach((value, axis) => close(value, token.position[axis], 1e-5)),
  );
});
test("projection is affine for A - B + C", () => {
  const a = index("king"),
    b = index("man"),
    c = index("woman");
  const actual = space.project(space.analogy(a, b, c));
  const expected = manifest.tokens[a].position.map(
    (value, i) =>
      value - manifest.tokens[b].position[i] + manifest.tokens[c].position[i],
  );
  actual.forEach((value, i) => close(value, expected[i]));
});
test("self similarity is one and exclusion removes self", () => {
  const king = index("king");
  const match = space.nearest(space.vector(king), new Set(), 1)[0];
  assert.equal(match.index, king);
  close(match.score, 1);
  assert.ok(
    space
      .nearest(space.vector(king), new Set([king]))
      .every((n) => n.index !== king),
  );
});
test("analogy matches independent NumPy reference ranking and scores", () => {
  const a = index("king"),
    b = index("man"),
    c = index("woman");
  const matches = space.nearest(space.analogy(a, b, c), new Set([a, b, c]), 10);
  matches.forEach((match, i) => {
    assert.equal(
      manifest.tokens[match.index].label,
      manifest.referenceAnalogy.neighbors[i].label,
    );
    close(match.score, manifest.referenceAnalogy.neighbors[i].cosine);
  });
});
test("subtracting a vector from itself leaves the remaining vector", () => {
  const a = index("king"),
    c = index("cat");
  const query = space.analogy(a, a, c);
  space.vector(c).forEach((value, i) => close(query[i], value));
  assert.equal(space.nearest(query, new Set(), 1)[0].index, c);
});
test("zero-vector cosine and corrupt vector files fail clearly", () => {
  assert.throws(() => space.nearest(new Float64Array(768)), /zero vector/);
  assert.throws(
    () => new EmbeddingSpace(manifest, new Float32Array(7)),
    /do not match/,
  );
});
test("PCA basis is orthonormal and reported variance is bounded", () => {
  const { basis, variance } = manifest.projection;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      close(
        basis[i].reduce((sum, v, k) => sum + v * basis[j][k], 0),
        i === j ? 1 : 0,
      );
  assert.ok(variance.every((v) => v > 0));
  assert.ok(variance.reduce((a, b) => a + b, 0) < 1);
});
