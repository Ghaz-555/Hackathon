export type Token = {
  id: number;
  label: string;
  token: string;
  position: [number, number, number];
};
export type Manifest = {
  schemaVersion: number;
  count: number;
  dimensions: number;
  fullVocabularySize: number;
  binary: string;
  sha256: string;
  source: string;
  revision: string;
  tokens: Token[];
  projection: {
    mean: number[];
    basis: number[][];
    scale: number;
    variance: number[];
  };
  referenceAnalogy: {
    queenRank: number;
    neighbors: { label: string; cosine: number }[];
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
      vectors.length !== manifest.count * manifest.dimensions ||
      manifest.tokens.length !== manifest.count
    )
      throw new Error("Embedding files do not match.");
    if (!vectors.every(Number.isFinite))
      throw new Error("Embedding values must be finite.");
    this.byLabel = new Map(manifest.tokens.map((t, i) => [t.label, i]));
    this.norms = Float64Array.from(manifest.tokens, (_, i) =>
      Math.hypot(...this.vector(i)),
    );
  }
  vector(i: number) {
    const d = this.manifest.dimensions;
    return this.vectors.subarray(i * d, (i + 1) * d);
  }
  combination(indices: number[], coefficients: number[]) {
    if (
      indices.length !== coefficients.length ||
      !indices.length ||
      indices.some(
        (i) => !Number.isInteger(i) || i < 0 || i >= this.manifest.count,
      ) ||
      coefficients.some((c) => !Number.isFinite(c))
    )
      throw new Error("Invalid vector coefficients.");
    const query = new Float64Array(this.manifest.dimensions);
    indices.forEach((index, j) =>
      this.vector(index).forEach((v, i) => {
        query[i] += coefficients[j] * v;
      }),
    );
    return query;
  }
  analogy(a: number, b: number, c: number) {
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
    if (query.length !== this.manifest.dimensions)
      throw new Error("Wrong query dimension.");
    const norm = Math.hypot(...Array.from(query));
    if (!Number.isFinite(norm) || norm < 1e-12)
      throw new Error("A zero or invalid vector has no cosine direction.");
    const result: Neighbor[] = [];
    for (let i = 0; i < this.manifest.count; i++) {
      if (excluded.has(i) || this.norms[i] === 0) continue;
      const vector = this.vector(i);
      let dot = 0;
      for (let j = 0; j < vector.length; j++) dot += vector[j] * query[j];
      result.push({
        index: i,
        score: Math.max(-1, Math.min(1, dot / (norm * this.norms[i]))),
      });
    }
    return result
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .slice(0, count);
  }
  project(query: ArrayLike<number>): [number, number, number] {
    const { basis, mean, scale } = this.manifest.projection;
    // Center after arithmetic so the displayed displacement agrees with A - B + C.
    return basis.map(
      (axis) =>
        axis.reduce((sum, v, i) => sum + v * (query[i] - mean[i]), 0) * scale,
    ) as [number, number, number];
  }
}
export async function loadSpace(): Promise<EmbeddingSpace> {
  const metadata = await fetch("/embeddings/manifest.json");
  if (!metadata.ok) throw new Error("Could not load the embedding manifest.");
  const manifest = (await metadata.json()) as Manifest;
  const response = await fetch("/embeddings/" + manifest.binary);
  if (!response.ok) throw new Error("Could not load the embedding vectors.");
  const bytes = await response.arrayBuffer();
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (v) => v.toString(16).padStart(2, "0"),
  ).join("");
  if (hash !== manifest.sha256) throw new Error("Embedding checksum mismatch.");
  return new EmbeddingSpace(manifest, new Float32Array(bytes));
}
