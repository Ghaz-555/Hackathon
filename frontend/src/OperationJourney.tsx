import { useMemo, useRef, type ComponentType, type ReactNode } from "react";
import { useFrame } from "@react-three/fiber";
import { Line, RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import { tokenLabel, type Inspection } from "./api";

type Point = [number, number, number];
type LabelProps = {
  position: Point;
  center?: boolean;
  children: ReactNode;
  style?: React.CSSProperties;
};
type Props = {
  data: Inspection;
  stage: number;
  token: number;
  reduced: boolean;
  paused: boolean;
  progress: number | null;
  dimension: number;
  Label: ComponentType<LabelProps>;
};
const ease = (t: number) => {
  t = Math.max(0, Math.min(1, t));
  return t * t * (3 - 2 * t);
};
const fmt = (n: number) => n.toFixed(3);

/** One operation, enlarged. Motion interpolates positions; stored values stay exact. */
export default function OperationJourney({
  data,
  stage,
  token,
  reduced,
  paused,
  progress,
  dimension,
  Label,
}: Props) {
  const elapsed = useRef(0),
    moving = useRef<THREE.Group>(null),
    source = useRef<THREE.Group>(null);
  const ribbon = useRef<HTMLDivElement>(null);
  const movingLabels = useRef<(HTMLSpanElement | null)[]>([]);
  const t = data.trace,
    expanded = data.training_scope === "all_parameters";
  const d = Math.min(dimension, data.config.d_model - 1);
  const embedding = t.token_embeddings[token],
    position = t.position_embeddings[token];
  const weights = t.blocks[0].attention[0][0][token];
  const hidden = t.final_hidden ?? t.blocks[0].head_outputs[0][0].at(-1)!;
  const best = data.probabilities.indexOf(Math.max(...data.probabilities));
  const headWidth = data.config.d_model / data.config.n_heads;
  const headDim = Math.min(d, headWidth - 1);
  const colors = [
    "#86efc2",
    "#8bdad9",
    "#b3a0ff",
    "#7bcaff",
    "#f6c888",
    "#b8f188",
  ];
  const color = colors[stage];
  const rows = useMemo(() => {
    const count = Math.min(8, data.config.d_model - d);
    if (stage === 0)
      return Array.from({ length: count }, (_, j) => ({
        label: `e${d + j}`,
        value: embedding[d + j],
        from: [-3.7, 0, 0] as Point,
        to: [-0.8 + j * 0.63, 0.25, 0] as Point,
      }));
    if (stage === 1)
      return Array.from({ length: count }, (_, j) => ({
        label: `p${d + j}`,
        value: position[d + j],
        from: [-3 + j * 0.82, 1.2, -0.8] as Point,
        to: [-3 + j * 0.82, -0.4, 0] as Point,
      }));
    if (stage === 2)
      return [t.queries[token], t.keys[token], t.values[token]].map((v, j) => ({
        label: ["Q", "K", "V"][j],
        value: v[headDim],
        from: [-3, 0, 0] as Point,
        to: [1.4, (1 - j) * 1.5, (j - 1) * 0.7] as Point,
      }));
    if (stage === 3)
      return weights
        .slice(0, token + 1)
        .map((w, j) => ({ w, j }))
        .sort((a, b) => b.w - a.w)
        .slice(0, 8)
        .sort((a, b) => a.j - b.j)
        .map(({ w, j }, i) => ({
          label: `${j + 1}:${tokenLabel(t.tokens[j])}`,
          value: w,
          from: [-3.2 + i * 0.85, 0.6, -0.3] as Point,
          to: [0.5, -0.7, 0] as Point,
        }));
    if (stage === 4)
      return hidden.slice(d, d + 8).map((v, j) => ({
        label: `h${d + j}`,
        value: v,
        from: [-3.5, j * 0.4 - 1.5, 0] as Point,
        to: [0.4, 0, 0] as Point,
      }));
    return data.characters.map((c, j) => ({
      label: tokenLabel(c),
      value: data.probabilities[j],
      from: [-3.7 + (j * 7.4) / data.characters.length, 0.7, 0] as Point,
      to: [
        -3.7 + (j * 7.4) / data.characters.length,
        -1.3 + data.probabilities[j] * 3,
        0,
      ] as Point,
    }));
  }, [data, stage, token, d]);
  const phases = [
    ["Read character", "Look up its row", "Unpack the learned vector"],
    [
      "Character + position",
      "Add matching dimensions",
      "Pass the combined vector forward",
    ],
    [
      expanded ? "Normalize the block input" : "Read the combined input",
      "Multiply by three learned matrices",
      "Separate query, key and value",
    ],
    [
      "Read only visible characters",
      "Weight and sum their value vectors",
      expanded
        ? "Join heads → residual → GELU → residual"
        : "A new contextual representation",
    ],
    [
      expanded
        ? "Read the final normalized vector"
        : "Read the final attention vector",
      "Multiply, sum and add bias",
      "One logit per possible character",
    ],
    [
      "Read the next-character logits",
      "Apply temperature and normalize",
      "A distribution, ready to sample",
    ],
  ][stage];
  useFrame((_, dt) => {
    if (!paused) elapsed.current += Math.min(dt, 0.05);
    const p =
      progress !== null
        ? progress / 100
        : reduced
          ? 1
          : (elapsed.current % 6.4) / 6.4;
    const u = ease((p - 0.16) / 0.65);
    moving.current?.children.forEach((child, i) => {
      const r = rows[i];
      if (!r) return;
      const local = ease((p - 0.12 - i * 0.008) / 0.62);
      child.position.set(...r.from).lerp(new THREE.Vector3(...r.to), local);
      child.position.z += Math.sin(local * Math.PI) * 0.7;
      child.scale.setScalar(
        stage === 3
          ? (0.4 + Math.sqrt(r.value) * 1.3) * (1 - local * 0.85)
          : 0.85 + local * 0.15,
      );
      if (movingLabels.current[i])
        movingLabels.current[i]!.style.opacity =
          stage === 0
            ? String(ease((local - 0.2) / 0.35))
            : stage === 1 || stage === 3 || stage === 4
              ? String(Math.max(0, 1 - local * 2))
              : "1";
    });
    if (source.current)
      source.current.scale.setScalar(stage === 0 ? 1 - u * 0.28 : 1);
    if (ribbon.current) {
      ribbon.current.style.setProperty("--journey-progress", `${p * 100}%`);
      ribbon.current.dataset.phase = String(Math.min(2, Math.floor(p * 3)));
    }
  }, -2); // Move objects before Html projects labels, including demand-rendered frames.
  const caption = (text: string, at: Point, className = "journey-caption") => (
    <Label position={at} center style={{ pointerEvents: "none" }}>
      <span className={className}>{text}</span>
    </Label>
  );
  return (
    <group>
      <RoundedBox
        args={[10.4, 6.1, 0.08]}
        radius={0.16}
        position={[0, 0, -1.8]}
      >
        <meshStandardMaterial
          color="#11232d"
          metalness={0.5}
          roughness={0.45}
        />
      </RoundedBox>
      <Label position={[0, 2.5, 0]} center style={{ pointerEvents: "none" }}>
        <div className="journey-title">
          <span>
            {stage >= 4 ? "FINAL POSITION" : "CHARACTER"}{" "}
            {stage >= 4 ? t.tokens.length : token + 1} ·{" "}
            {tokenLabel(t.tokens[stage >= 4 ? t.tokens.length - 1 : token])}
            {expanded && stage >= 2 && stage <= 3
              ? ` / LAYER ${(t.layer_index ?? 0) + 1} · HEAD ${(t.head_index ?? 0) + 1}`
              : ""}
          </span>
          <strong>
            {
              [
                "A character becomes a vector.",
                "Give the character its place.",
                "One input. Three perspectives.",
                "Gather context. Transform it.",
                "Turn context into a prediction.",
                "From scores to possibilities.",
              ][stage]
            }
          </strong>
        </div>
      </Label>
      {stage === 0 && (
        <>
          <group ref={source} position={[-3.7, 0, 0]}>
            <RoundedBox args={[1.1, 1.3, 0.4]} radius={0.1}>
              <meshStandardMaterial
                color="#235346"
                emissive="#86efc2"
                emissiveIntensity={0.2}
              />
            </RoundedBox>
            {caption(
              tokenLabel(t.tokens[token]),
              [0, 0, 0.3],
              "journey-character",
            )}
          </group>
          <Line
            points={[
              [-3.1, 0, 0],
              [-1.3, 0, 0],
            ]}
            color={color}
            dashed
            dashSize={0.12}
            gapSize={0.1}
          />
          {caption(
            `token ID ${data.characters.indexOf(t.tokens[token])} → embedding lookup`,
            [-2.6, -1.35, 0],
          )}
          {caption(
            `Dimensions ${d}–${d + rows.length - 1} of ${embedding.length}`,
            [1.5, -1.35, 0],
          )}
        </>
      )}
      {stage === 1 && (
        <>
          {embedding.slice(d, d + 8).map((v, j) => (
            <group key={j} position={[-3 + j * 0.82, -0.4, -0.12]}>
              <RoundedBox args={[0.62, 0.7, 0.17]} radius={0.04}>
                <meshStandardMaterial color="#26594c" />
              </RoundedBox>
              {caption(fmt(v), [0, 0, 0.15], "journey-number")}
              {caption(
                `= ${fmt(v + position[d + j])}`,
                [0, -0.85, 0],
                "journey-number",
              )}
            </group>
          ))}
          {caption(
            "P moves into E. Addition is component by component.",
            [0, 1.25, 0],
          )}
        </>
      )}
      {stage === 2 && (
        <>
          {caption(
            expanded ? "LayerNorm(X)" : "X = E + P",
            [-3, 0, 0],
            "journey-operator",
          )}
          {rows.map((r, j) => (
            <group key={j}>
              <Line
                points={[[-2.3, 0, 0], [-0.6, (1 - j) * 1.5, 0], r.to]}
                color={["#b3a0ff", "#75cfeb", "#f3bd8d"][j]}
                lineWidth={2}
              />
              {caption(
                ["× Wq + bq", "× Wk + bk", "× Wv + bv"][j].replace(
                  expanded ? "!" : / \+ b./g,
                  "",
                ),
                [-0.55, (1 - j) * 1.5, 0.1],
              )}
            </group>
          ))}
          {caption(
            `One displayed component per branch · head dimension ${headDim}`,
            [0, -2, 0],
          )}
        </>
      )}
      {stage === 3 && (
        <>
          {caption(`Σⱼ attention[${token},j] × V[j,${headDim}]`, [0, 1.25, 0])}
          {rows.map((r, i) => (
            <group key={i}>
              <Line
                points={[r.from, [0.5, -0.7, 0]]}
                color="#5394ac"
                transparent
                opacity={0.25}
                lineWidth={1 + r.value * 4}
              />
              {caption(
                `${r.label} · ${(r.value * 100).toFixed(1)}%`,
                [r.from[0], 0.72, -0.3],
                "journey-number",
              )}
            </group>
          ))}
          <Line
            points={[
              [-3, -0.7, 0],
              [3.6, -0.7, 0],
            ]}
            color={color}
            lineWidth={2}
          />
          {caption(
            `Σ = ${fmt(t.blocks[0].head_outputs[0][0][token][headDim])}`,
            [0.5, -0.7, 0.3],
            "journey-operator",
          )}
          {token >= 8 &&
            caption(
              `Showing 8 strongest of ${token + 1} visible positions · sum uses all`,
              [0, 1.65, 0],
            )}
          {expanded && (
            <>
              {caption("concat 2 heads → 32", [2.8, -0.65, 0])}
              {caption(
                "+ residual → LN → 128 GELU → 32 → + residual",
                [0, -1.6, 0],
              )}
            </>
          )}
        </>
      )}
      {stage === 4 && (
        <>
          {caption(`Wout[:, ${best}]`, [0.4, 0.95, 0], "journey-operator")}
          <Line
            points={[
              [0.4, 0, 0],
              [3.2, 0, 0],
            ]}
            color={color}
            lineWidth={2}
          />
          {caption(
            `${tokenLabel(data.characters[best])}: ${fmt(data.logits[best])}`,
            [3, 0, 0.2],
            "journey-operator",
          )}
          {caption(
            `${hidden.length} products (showing ${rows.length}) + ${fmt(t.output_bias?.[best] ?? 0)} bias`,
            [0, -1.9, 0],
          )}
        </>
      )}
      {stage === 5 && (
        <>
          {caption(
            data.temperature === 0
              ? "T = 0 · choose the highest score"
              : `pᵢ = exp(zᵢ / ${data.temperature.toFixed(2)}) / Σⱼ exp(zⱼ / T)`,
            [0, 1.6, 0],
          )}
          {rows.map((r, j) => (
            <group key={j} position={[r.to[0], -1.5, 0]}>
              <mesh position={[0, r.value * 1.5, 0]}>
                <boxGeometry
                  args={[
                    Math.min(0.2, 6 / rows.length),
                    Math.max(0.012, r.value * 3),
                    0.18,
                  ]}
                />
                <meshStandardMaterial
                  color={color}
                  emissive={color}
                  emissiveIntensity={0.2}
                />
              </mesh>
              {caption(r.label, [0, -0.3, 0], "journey-number")}
            </group>
          ))}
        </>
      )}
      <group ref={moving}>
        {rows.map((r, i) => (
          <group key={i} position={r.from}>
            <RoundedBox
              args={[stage === 5 ? 0.17 : 0.52, stage === 5 ? 0.17 : 0.5, 0.25]}
              radius={0.04}
            >
              <meshStandardMaterial
                color={stage === 3 ? "#3f769b" : color}
                emissive={color}
                emissiveIntensity={0.16}
                metalness={0.35}
              />
            </RoundedBox>
            {stage !== 5 && (
              <Label
                position={[0, 0, 0.2]}
                center
                style={{ pointerEvents: "none" }}
              >
                <span
                  ref={(el) => {
                    movingLabels.current[i] = el;
                  }}
                  className="journey-number"
                >
                  {stage === 0 || stage === 1
                    ? fmt(r.value)
                    : stage === 3
                      ? ""
                      : `${r.label} ${fmt(r.value)}`}
                </span>
              </Label>
            )}
          </group>
        ))}
      </group>
      <Label position={[0, -2.55, 0]} center style={{ pointerEvents: "none" }}>
        <div ref={ribbon} className="journey-phases" data-phase="0">
          {phases.map((p, i) => (
            <span key={p}>
              <b>0{i + 1}</b>
              {p}
            </span>
          ))}
          <i />
        </div>
      </Label>
    </group>
  );
}
