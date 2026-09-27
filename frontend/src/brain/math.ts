export type Token = {
  id: number;
  label: string;
  token: string;
  position: [number, number, number];
};
export type Manifest = {
  schemaVersion: number;
  model: string;
  repository: string;
  revision: string;
  source: string;
  tensor: string;
  dimensions: number;
  fullVocabularySize: number;
  count: number;
  binary: string;
  sha256: string;
  selection: string;
  projection: {
    method: string;
    variance: number[];
    scale: number;
    mean: number[];
    basis: number[][];
  };
  tokens: Token[];
  referenceAnalogy: {
    expression: string;
    neighbors: { label: string; cosine: number }[];
    queenRank: number;
  };
};
export type Neighbor = { index: number; score: number };

export class EmbeddingSpace {
  readonly norms: Float64Array;
  readonly byLabel: Map<string, number>;
  constructor(
    readonly manifest: Manifest,
    readonly vectors: Float32Array,
  ) {
    if (
      manifest.schemaVersion !== 1 ||
      vectors.length !== manifest.count * manifest.dimensions
    )
      throw new Error(
        "The embedding files do not match. Regenerate the dataset.",
      );
    if (!vectors.every(Number.isFinite))
      throw new Error("The embedding file contains invalid values.");
    this.norms = new Float64Array(manifest.count);
    this.byLabel = new Map(manifest.tokens.map((token, i) => [token.label, i]));
    for (let i = 0; i < manifest.count; i++)
      this.norms[i] = Math.hypot(...this.vector(i));
  }
  vector(index: number): Float32Array {
    const d = this.manifest.dimensions;
    return this.vectors.subarray(index * d, (index + 1) * d);
  }
  analogy(a: number, b: number, c: number): Float64Array {
    const av = this.vector(a),
      bv = this.vector(b),
      cv = this.vector(c);
    return Float64Array.from(av, (v, i) => v - bv[i] + cv[i]);
  }
  nearest(
    query: ArrayLike<number>,
    excluded = new Set<number>(),
    count = 8,
  ): Neighbor[] {
    const norm = Math.hypot(...Array.from(query));
    if (norm < 1e-12)
      throw new Error(
        "A zero vector has no cosine direction. Choose another expression.",
      );
    const matches: Neighbor[] = [];
    for (let i = 0; i < this.manifest.count; i++) {
      if (excluded.has(i) || this.norms[i] === 0) continue;
      const vector = this.vector(i);
      let dot = 0;
      for (let j = 0; j < vector.length; j++) dot += vector[j] * query[j];
      matches.push({
        index: i,
        score: Math.max(-1, Math.min(1, dot / (norm * this.norms[i]))),
      });
    }
    return matches
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .slice(0, count);
  }
  project(vector: ArrayLike<number>): [number, number, number] {
    const { mean, basis, scale } = this.manifest.projection;
    // Center once, after arithmetic. A - B + C still has coefficients summing to one.
    return basis.map(
      (axis) =>
        axis.reduce((sum, value, i) => sum + value * (vector[i] - mean[i]), 0) *
        scale,
    ) as [number, number, number];
  }
}

export async function loadSpace(
  base = "/embeddings/",
): Promise<EmbeddingSpace> {
  const response = await fetch(base + "manifest.json");
  if (!response.ok)
    throw new Error(
      "Embedding data is missing. Run scripts/prepare_embeddings.py.",
    );
  const manifest = (await response.json()) as Manifest;
  const binary = await fetch(base + manifest.binary);
  if (!binary.ok) throw new Error("The vector file could not be loaded.");
  const buffer = await binary.arrayBuffer();
  if (globalThis.crypto?.subtle) {
    const hash = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
      (v) => v.toString(16).padStart(2, "0"),
    ).join("");
    if (hash !== manifest.sha256)
      throw new Error("The vector file failed its checksum. Regenerate it.");
  }
  return new EmbeddingSpace(manifest, new Float32Array(buffer));
}
