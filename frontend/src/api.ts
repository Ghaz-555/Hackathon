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
    d_ff?: number;
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
    output_bias?: number[];
    final_hidden?: number[];
    query_weights: number[][];
    key_weights: number[][];
    value_weights: number[][];
    blocks: BlockTrace[];
    selected_block?: BlockTrace;
    layer_index?: number;
    head_index?: number;
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

export type BlockTrace = {
  attention: number[][][][]; head_outputs: number[][][][];
  query?: number[][][][]; key?: number[][][][]; value?: number[][][][];
  query_weights?: number[][]; key_weights?: number[][]; value_weights?: number[][];
  normalized_attention_input?: number[][][]; attention_projection?: number[][][];
  attention_residual?: number[][][]; normalized_ff_input?: number[][][];
  ff_pre_activation?: number[][][]; ff_activation?: number[][][];
  ff_projection?: number[][][]; output?: number[][][];
  qkv_bias?: number[];
};
// Keep the API trace intact; choose one layer/head for the close-up view.
export function selectTrace(data: Inspection, layer: number, head: number): Inspection {
  if (data.training_scope !== "all_parameters") return data;
  const block = data.trace.blocks[layer] ?? data.trace.blocks[0];
  const d = data.config.d_model / data.config.n_heads;
  const slice = (m: number[][]) => m.map(r => r.slice(head*d, (head+1)*d));
  return { ...data, trace: { ...data.trace,
    queries: block.query![0][head], keys: block.key![0][head], values: block.value![0][head],
    query_weights: slice(block.query_weights!), key_weights: slice(block.key_weights!), value_weights: slice(block.value_weights!),
    blocks: [{ attention: [[block.attention[0][head]]], head_outputs: [[block.head_outputs[0][head]]] }],
    selected_block: block, layer_index: layer, head_index: head,
  }};
}
