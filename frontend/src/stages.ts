export const stages = [
  {
    name: "Embedding",
    short: "Characters → vectors",
    color: "#86efc2",
    formula: "E = lookup(character)",
    description:
      "Each character gets a vector of four numbers. These numbers are the starting representation the model works with.",
    detail:
      "One tile is one actual value. Green is positive; blue is negative. Tile depth shows magnitude, capped at 1.5 for readability. Hover or use the inspector for exact numbers.",
  },
  {
    name: "Position",
    short: "Add a sense of order",
    color: "#8bdad9",
    formula: "Xᵢ = Eᵢ + Pᵢ",
    description:
      "A position vector is added to each character vector. This gives the same character a different representation at each position.",
    detail:
      "The inspector shows the addition for your selected character, using the actual stored vectors.",
  },
  {
    name: "Q · K · V",
    short: "Three views of the input",
    color: "#b3a0ff",
    formula: "Q = XWq   K = XWk   V = XWv",
    description:
      "The input is projected three ways: queries and keys determine attention scores; values carry the information that gets mixed.",
    detail:
      "The three branches show real 4 × 4 projection weights beside the resulting query, key and value tensors. These parameters remain fixed during output-layer training.",
  },
  {
    name: "Attention",
    short: "Mix the visible context",
    color: "#7bcaff",
    formula: "A = softmax(QKᵀ / √4 + mask)",
    description:
      "Each position can attend to itself and earlier characters. The weighted values combine into a new four-number representation.",
    detail:
      "Bright cells carry more attention. Empty future cells are masked. Attention weights are not a complete explanation of reasoning.",
  },
  {
    name: "Projection",
    short: "Turn context into scores",
    color: "#f6c888",
    formula: "z = hlast Wout",
    description:
      "The final position’s attention output is multiplied by a 4 × 10 matrix to produce one score for each possible next character.",
    detail:
      "These 40 output weights are the only parameters updated by this engine’s SGD training. Amber highlights mark actual changed weights.",
  },
  {
    name: "Probabilities",
    short: "Choose what comes next",
    color: "#b8f188",
    formula: "pᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)",
    description:
      "Softmax turns the scores into probabilities. Temperature changes this distribution; it does not change model weights.",
    detail:
      "Column heights show actual probabilities, on a shared 0–100% scale. At temperature zero, the largest score is selected directly.",
  },
];
