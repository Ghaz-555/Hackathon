export type Point = { step: number; train: number; validation: number };
export type ModelState = {
  revision: number;
  mode: "random" | "trained";
  step: number;
  fingerprint: string;
  parameter_count: number;
  trainable_parameter_count: number;
  engine_id: string;
  model_name: string;
  training_scope: string;
  optimizer: string;
  learning_rate: number;
  examples_per_step: number;
  training_text: string;
  validation_text: string;
  characters: string[];
  history: Point[];
  config: {
    n_layers: number;
    n_heads: number;
    d_model: number;
    context_length: number;
  };
};
export type Inspection = ModelState & {
  prompt: string;
  temperature: number;
  logits: number[];
  probabilities: number[];
  entropy: number;
  trace: {
    schema_version: number;
    tokens: string[];
    token_embeddings: number[][];
    position_embeddings: number[][];
    embedding_sum: number[][][];
    queries: number[][];
    keys: number[][];
    values: number[][];
    output_weights: number[][];
    query_weights: number[][];
    key_weights: number[][];
    value_weights: number[][];
    blocks: { attention: number[][][][]; head_outputs: number[][][][] }[];
  };
};
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const response = await fetch("/api" + path, {
    method: method ?? (body === undefined ? "GET" : "POST"),
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      typeof data.detail === "string"
        ? data.detail
        : "Check your input and try again.",
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}
export const tokenLabel = (s: string) =>
  s === " " ? "␣" : s === "\n" ? "↵" : s;
