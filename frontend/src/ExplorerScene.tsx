import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  type ComponentProps,
  type RefObject,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  OrbitControls,
  Html as DreiHtml,
  Edges,
  Line,
  RoundedBox,
} from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import type { Inspection } from "./api";
import { tokenLabel } from "./api";
import OperationJourney from "./OperationJourney";
import { lessonStages } from "./stages";

type Point = [number, number, number];
type Props = {
  data: Inspection;
  stage: number;
  token: number;
  playing: boolean;
  paused: boolean;
  progress: number | null;
  dimension: number;
  reduced: boolean;
  onStage: (i: number) => void;
  onValue: (s: string) => void;
  reset: number;
  focus: boolean;
  changed: number[][] | null;
  onFailure: () => void;
};
const centers: Point[] = [
  [-6, 2.3, 0],
  [-2.6, 2.3, 0],
  [1.6, 2.3, 0],
  [5.5, -0.1, 0],
  [1, -3.4, 0],
  [-4.6, -3.4, 0],
];
const portalContext = createContext<
  RefObject<HTMLDivElement | null> | undefined
>(undefined);
function Html(props: ComponentProps<typeof DreiHtml>) {
  const portal = useContext(portalContext);
  return (
    <DreiHtml
      {...props}
      portal={portal?.current ? { current: portal.current } : undefined}
      zIndexRange={[20, 0]}
    />
  );
}

// Cell geometry comes from real arrays. Activity lighting illustrates the pass;
// it does not alter the stored values or pretend to measure execution time.
function Tensor({
  values,
  color,
  label,
  active,
  motion,
  selected = -1,
  offset = 0,
  onValue,
  changed,
  size = 1.9,
  causal = false,
}: {
  values: number[][];
  color: string;
  label: string;
  active: boolean;
  motion: boolean;
  selected?: number;
  offset?: number;
  onValue: Props["onValue"];
  changed?: number[][] | null;
  size?: number;
  causal?: boolean;
}) {
  const mesh = useRef<THREE.InstancedMesh>(null),
    scan = useRef<THREE.Mesh>(null),
    time = useRef(0);
  const rows = values.length,
    cols = values[0].length,
    pitch = size / Math.max(rows, cols),
    width = cols * pitch,
    height = rows * pitch;
  const depths = useRef<number[]>([]),
    matrix = useMemo(() => new THREE.Matrix4(), []),
    cellPosition = useMemo(() => new THREE.Vector3(), []),
    cellScale = useMemo(() => new THREE.Vector3(), []),
    cellRotation = useMemo(() => new THREE.Quaternion(), []),
    base = useMemo(() => new THREE.Color(color), [color]),
    shade = useMemo(() => new THREE.Color(), []);
  const revision = values.flat().join(",");
  const update = (dt: number) => {
    if (!mesh.current) return;
    if (motion) time.current += Math.min(dt, 0.05);
    const rowPhase = (time.current * 0.65) % 1;
    values.forEach((row, r) =>
      row.forEach((value, c) => {
        const i = r * cols + c,
          masked = causal && c > r,
          target = 0.07 + Math.min(Math.abs(value), 1.5) * 0.24;
        const previous = depths.current[i] ?? target;
        const depth = motion
          ? THREE.MathUtils.lerp(previous, target, 1 - Math.exp(-dt * 9))
          : target;
        depths.current[i] = depth;
        matrix.compose(
          cellPosition.set(
            (c - (cols - 1) / 2) * pitch,
            ((rows - 1) / 2 - r) * pitch,
            depth / 2,
          ),
          cellRotation,
          cellScale.set(pitch * 0.83, pitch * 0.83, masked ? 0.018 : depth),
        );
        mesh.current!.setMatrixAt(i, matrix);
        shade.copy(
          masked
            ? new THREE.Color("#182733")
            : value < 0
              ? new THREE.Color("#4c87bd")
              : base,
        );
        const scanStrength =
          active && motion
            ? Math.max(0, 1 - Math.abs((r + 0.5) / rows - rowPhase) * rows) *
              0.65
            : 0;
        shade.lerp(
          new THREE.Color("#e5fff6"),
          masked ? 0 : scanStrength + (r === selected ? 0.16 : 0),
        );
        if (changed && Math.abs(changed[r]?.[c] ?? 0) > 1e-12)
          shade
            .set("#ffc578")
            .multiplyScalar(
              motion ? 1.05 + 0.2 * Math.sin(time.current * 5) : 1.1,
            );
        mesh.current!.setColorAt(i, shade);
      }),
    );
    mesh.current.instanceMatrix.needsUpdate = true;
    if (mesh.current.instanceColor)
      mesh.current.instanceColor.needsUpdate = true;
    if (scan.current) {
      scan.current.visible = motion && active;
      scan.current.position.y = height / 2 - rowPhase * height;
    }
  };
  useEffect(() => {
    update(0.1);
  }, [revision, active, motion, selected, changed]);
  useFrame((_, dt) => update(dt));
  return (
    <group>
      <RoundedBox
        args={[width + 0.15, height + 0.15, 0.06]}
        radius={0.045}
        smoothness={2}
        position={[0, 0, -0.075]}
      >
        <meshStandardMaterial
          color="#152832"
          metalness={0.6}
          roughness={0.24}
        />
        <Edges color={color} transparent opacity={active ? 0.8 : 0.3} />
      </RoundedBox>
      <instancedMesh
        ref={mesh}
        args={[undefined, undefined, rows * cols]}
        frustumCulled={false}
        onPointerMove={(e) => {
          if (e.instanceId === undefined) return;
          e.stopPropagation();
          const r = Math.floor(e.instanceId / cols),
            c = e.instanceId % cols;
          onValue(
            `${label} [${r + offset}, ${c + (causal ? offset : 0)}] = ${values[r][c].toFixed(6)}${causal && c > r ? " · masked" : ""}`,
          );
        }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          metalness={0.35}
          roughness={0.3}
          emissive={color}
          emissiveIntensity={active ? 0.13 : 0.035}
        />
      </instancedMesh>
      {selected >= 0 && selected < rows && (
        <mesh position={[0, ((rows - 1) / 2 - selected) * pitch, 0.015]}>
          <boxGeometry args={[width + 0.12, pitch, 0.055]} />
          <meshBasicMaterial
            color={color}
            wireframe
            transparent
            opacity={0.6}
          />
        </mesh>
      )}
      <mesh ref={scan} position={[0, 0, 0.47]}>
        <planeGeometry args={[width + 0.13, 0.027]} />
        <meshBasicMaterial
          color="#c6ffeb"
          transparent
          opacity={0.65}
          depthWrite={false}
        />
      </mesh>
      <Html
        position={[0, -height / 2 - 0.23, 0]}
        center
        style={{ pointerEvents: "none" }}
      >
        <span className="tensor-caption">
          {label}
          <b>
            {rows} × {cols}
          </b>
        </span>
      </Html>
    </group>
  );
}

type Connection = {
  from: Point;
  to: Point;
  color: string;
  stage: number;
  bend?: number;
};
function Conduit({
  from,
  to,
  color,
  stage,
  active,
  motion,
  bend = 0,
}: { active: number; motion: boolean } & Connection) {
  const dots = useRef<THREE.Group>(null),
    clock = useRef(0);
  const curve = useMemo(() => {
    const a = new THREE.Vector3(...from),
      b = new THREE.Vector3(...to),
      delta = b.clone().sub(a);
    return new THREE.CubicBezierCurve3(
      a,
      a
        .clone()
        .addScaledVector(delta, 0.28)
        .add(new THREE.Vector3(0, 0.35, bend)),
      a
        .clone()
        .addScaledVector(delta, 0.72)
        .add(new THREE.Vector3(0, 0.35, bend)),
      b,
    );
  }, [...from, ...to, bend]);
  const points = useMemo(() => curve.getPoints(32), [curve]);
  useFrame((_, dt) => {
    if (!motion || !dots.current) return;
    clock.current += Math.min(dt, 0.05);
    dots.current.children.forEach((dot, i) => {
      const t = (clock.current * 0.3 + i / 4) % 1;
      dot.position.copy(curve.getPoint(t));
    });
  });
  return (
    <group>
      <Line
        points={points}
        color={color}
        transparent
        opacity={active === stage || active + 1 === stage ? 0.9 : 0.42}
        lineWidth={active === stage || active + 1 === stage ? 2.1 : 1.2}
      />
      <group ref={dots} visible={motion}>
        {Array.from({ length: 4 }, (_, i) => (
          <mesh key={i} position={curve.getPoint(i / 4)}>
            <sphereGeometry
              args={[
                active === stage || active + 1 === stage ? 0.065 : 0.04,
                8,
                8,
              ]}
            />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={active === stage ? 1 : 0.45}
            />
          </mesh>
        ))}
      </group>
    </group>
  );
}
function OutputBars({
  data,
  active,
  motion,
  onValue,
}: {
  data: Inspection;
  active: boolean;
  motion: boolean;
  onValue: Props["onValue"];
}) {
  const group = useRef<THREE.Group>(null);
  const current = useRef(data.probabilities.map((p) => p * 2.5));
  const pitch = 2.8 / data.characters.length;
  useFrame((_, dt) => {
    group.current?.children.forEach((child, i) => {
      const target = data.probabilities[i] * 2.5;
      current.current[i] = motion
        ? THREE.MathUtils.lerp(
            current.current[i] ?? target,
            target,
            1 - Math.exp(-dt * 8),
          )
        : target;
      child.scale.y = Math.max(current.current[i], 0.00001);
      child.position.y = current.current[i] / 2;
      child.visible = target > 0;
    });
  });
  return (
    <group>
      <group ref={group}>
        {data.probabilities.map((p, i) => (
          <mesh
            key={i}
            position={[
              (i - (data.characters.length - 1) / 2) * pitch,
              p * 1.25,
              0,
            ]}
            scale={[1, Math.max(p * 2.5, 0.00001), 1]}
            visible={p > 0}
            onPointerMove={(e) => {
              e.stopPropagation();
              onValue(
                `${tokenLabel(data.characters[i])}: ${(p * 100).toFixed(4)}%`,
              );
            }}
          >
            <boxGeometry args={[pitch * 0.74, 1, 0.23]} />
            <meshStandardMaterial
              color="#b8f188"
              emissive="#94d076"
              emissiveIntensity={active ? 0.4 : 0.1}
              metalness={0.3}
              roughness={0.25}
            />
          </mesh>
        ))}
      </group>
      {data.characters.map((c, i) => (
        <Html
          key={i}
          position={[(i - (data.characters.length - 1) / 2) * pitch, -0.22, 0]}
          center
          style={{ pointerEvents: "none" }}
        >
          <span className="world-token">{tokenLabel(c)}</span>
        </Html>
      ))}
      <Line
        points={[
          [-1.5, 0, 0],
          [1.5, 0, 0],
        ]}
        color="#7bab74"
        transparent
        opacity={0.5}
      />
      <Html position={[0, -0.62, 0]} center style={{ pointerEvents: "none" }}>
        <span className="tensor-caption">
          NEXT CHARACTER<b>T = {data.temperature.toFixed(2)}</b>
        </span>
      </Html>
    </group>
  );
}
function CameraRig({
  props,
  controls,
}: {
  props: Props;
  controls: RefObject<OrbitControlsImpl | null>;
}) {
  const { camera, size, invalidate } = useThree();
  const destination = useRef({
    eye: new THREE.Vector3(9, 6, 21),
    target: new THREE.Vector3(0, -0.7, 0),
    zoom: 45,
    moving: false,
  });
  const initialized = useRef(false);
  const follow = props.focus || props.playing;
  useEffect(() => {
    const center = props.focus
      ? new THREE.Vector3(0, 0, 0)
      : props.playing
        ? new THREE.Vector3(...centers[props.stage])
        : new THREE.Vector3(0, -0.8, 0);
    destination.current = {
      eye: center
        .clone()
        .add(
          props.focus
            ? new THREE.Vector3(1.5, 1.2, 16)
            : props.playing
              ? new THREE.Vector3(5, 3, 14)
              : new THREE.Vector3(8, 6, 21),
        ),
      target: center,
      zoom: props.focus
        ? Math.min(size.width / 12, size.height / 7.2)
        : props.playing
          ? Math.min(size.width / 6, size.height / 4.8)
          : Math.min(size.width / 18, size.height / 11),
      moving: true,
    };
    if (props.reduced || !initialized.current) {
      camera.position.copy(destination.current.eye);
      controls.current?.target.copy(center);
      if (camera instanceof THREE.OrthographicCamera) {
        camera.zoom = destination.current.zoom;
        camera.updateProjectionMatrix();
      }
      controls.current?.update();
      destination.current.moving = false;
      initialized.current = true;
    }
    invalidate();
  }, [
    camera,
    size.width,
    size.height,
    props.reset,
    follow,
    props.focus,
    props.playing,
    follow ? props.stage : -1,
    props.reduced,
  ]);
  useEffect(() => {
    const c = controls.current;
    const stop = () => {
      destination.current.moving = false;
    };
    c?.addEventListener("start", stop);
    return () => c?.removeEventListener("start", stop);
  }, [controls]);
  useFrame((_, dt) => {
    const d = destination.current;
    if (!d.moving) return;
    const t = 1 - Math.exp(-dt * 4.5);
    camera.position.lerp(d.eye, t);
    controls.current?.target.lerp(d.target, t);
    if (camera instanceof THREE.OrthographicCamera) {
      camera.zoom = THREE.MathUtils.lerp(camera.zoom, d.zoom, t);
      camera.updateProjectionMatrix();
    }
    controls.current?.update();
    if (camera.position.distanceTo(d.eye) < 0.005) d.moving = false;
    invalidate();
  }, -1);
  return null;
}
function World(props: Props) {
  const { data, stage, token, onStage, onValue } = props;
  const { gl } = useThree();
  const expanded = data.training_scope === "all_parameters";
  const stages = lessonStages(expanded);
  const controls = useRef<OrbitControlsImpl>(null);
  const motion = !props.reduced && !props.paused;
  const start = Math.max(0, Math.min(token - 7, data.trace.tokens.length - 16)),
    end = Math.min(start + 16, data.trace.tokens.length),
    rows = (v: number[][]) => v.slice(start, end);
  useEffect(() => {
    const lost = (e: Event) => {
      e.preventDefault();
      props.onFailure();
    };
    gl.domElement.addEventListener("webglcontextlost", lost);
    return () => {
      gl.domElement.removeEventListener("webglcontextlost", lost);
      document.body.style.cursor = "auto";
    };
  }, [gl, props.onFailure]);
  const common = { onValue, motion };
  const tensor = (
    values: number[][],
    color: string,
    label: string,
    i: number,
    extra: Partial<ComponentProps<typeof Tensor>> = {},
  ) => (
    <Tensor
      values={values}
      color={color}
      label={label}
      active={stage === i}
      {...common}
      {...extra}
    />
  );
  const connections: Connection[] = [
    { from: [-5.1, 2.3, 0], to: [-3.5, 2.3, 0], color: "#86efc2", stage: 1 },
    ...[-1.35, 0, 1.35].map((z, j) => ({
      from: [-1.6, 2.3, 0] as Point,
      to: [0.35, 2.3, z] as Point,
      color: ["#b9a0ff", "#75cfeb", "#f3bd8d"][j],
      stage: 2,
      bend: z,
    })),
    ...[-1.35, 0, 1.35].map((z, j) => ({
      from: [2.85, 2.3, z] as Point,
      to: [5.5, j === 2 ? -1.95 : 0.95, 0] as Point,
      color: ["#b9a0ff", "#75cfeb", "#f3bd8d"][j],
      stage: 3,
      bend: z,
    })),
    { from: [5.5, -1.3, 0], to: [5.5, -2.05, 0], color: "#7bcaff", stage: 3 },
    {
      from: [4.7, -2.1, 0],
      to: [2.4, -3.4, 0],
      color: "#f6c888",
      stage: 4,
      bend: 1,
    },
    { from: [-0.2, -3.4, 0], to: [-3.1, -3.4, 0], color: "#b8f188", stage: 5 },
  ];
  return (
    <>
      <color attach="background" args={["#0c141b"]} />
      <fog attach="fog" args={["#0c141b", 32, 75]} />
      <ambientLight intensity={0.8} />
      <hemisphereLight args={["#c1ede1", "#102037", 1.6]} />
      <directionalLight position={[3, 8, 12]} intensity={2.6} color="#d7f6eb" />
      <pointLight
        position={[-6, 4, 4]}
        intensity={22}
        color="#82e6c4"
        distance={20}
      />
      <pointLight
        position={[5, 3, -2]}
        intensity={30}
        color="#9694ff"
        distance={20}
      />
      <gridHelper
        args={[65, 65, "#234039", "#14242b"]}
        position={[0, -5.3, 0]}
      />
      <mesh position={[0, -5.34, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#0b141b" metalness={0.2} roughness={0.8} />
      </mesh>
      {props.focus ? (
        <OperationJourney
          key={`${stage}-${token}-${props.reset}-${props.dimension}-${data.trace.layer_index}-${data.trace.head_index}`}
          data={data}
          stage={stage}
          token={token}
          reduced={props.reduced}
          paused={props.paused}
          progress={props.progress}
          dimension={props.dimension}
          Label={Html}
        />
      ) : (
        <>
          {connections.map((edge, i) => (
            <Conduit key={i} {...edge} active={stage} motion={motion} />
          ))}
          {stages.map((item, i) => (
            <group
              key={i}
              position={centers[i]}
              onClick={(e) => {
                e.stopPropagation();
                onStage(i);
              }}
              onPointerOver={() => {
                document.body.style.cursor = "pointer";
              }}
              onPointerOut={() => {
                document.body.style.cursor = "auto";
              }}
            >
              {/* A glass backplate groups actual operations; it is not a fictitious model layer. */}
              <RoundedBox
                args={[
                  i === 2 ? 3.7 : i === 3 ? 3.2 : 3,
                  i === 3 ? 4.7 : 3.4,
                  0.06,
                ]}
                radius={0.12}
                smoothness={3}
                position={[0, i === 3 ? -0.6 : 0, -1.8]}
              >
                <meshPhysicalMaterial
                  color={item.color}
                  transparent
                  opacity={i === stage ? 0.065 : 0.025}
                  roughness={0.25}
                  depthWrite={false}
                />
                <Edges
                  color={item.color}
                  transparent
                  opacity={i === stage ? 0.38 : 0.1}
                />
              </RoundedBox>
              <Html
                position={[0, i === 3 ? 2 : i === 5 ? 3.1 : 2.15, 0]}
                center
              >
                <button
                  className={"world-label " + (i === stage ? "active" : "")}
                  onClick={() => onStage(i)}
                >
                  <span>0{i + 1}</span>
                  {item.name}
                </button>
              </Html>
              {i === 0 && (
                <>
                  {tensor(
                    rows(data.trace.token_embeddings),
                    item.color,
                    "TOKEN EMBEDDINGS",
                    i,
                    { selected: token - start, offset: start, size: 2.2 },
                  )}
                  <Html
                    position={[0, 1.5, 0]}
                    center
                    style={{ pointerEvents: "none" }}
                  >
                    <span className="input-token-strip">
                      {data.trace.tokens.slice(start, end).map((c, j) => (
                        <b
                          className={j === token - start ? "selected" : ""}
                          key={j}
                        >
                          {tokenLabel(c)}
                        </b>
                      ))}
                    </span>
                  </Html>
                </>
              )}
              {i === 1 && (
                <>
                  <group position={[0, 0, -0.55]}>
                    {tensor(
                      rows(data.trace.position_embeddings),
                      "#568a9d",
                      "POSITIONS P",
                      i,
                      { selected: token - start, offset: start, size: 1.95 },
                    )}
                  </group>
                  <group position={[0.45, 0.15, 0.7]}>
                    {tensor(
                      rows(data.trace.embedding_sum[0]),
                      item.color,
                      "X = E + P",
                      i,
                      { selected: token - start, offset: start, size: 1.95 },
                    )}
                  </group>
                </>
              )}
              {i === 2 &&
                [data.trace.queries, data.trace.keys, data.trace.values].map(
                  (matrix, j) => (
                    <group key={j} position={[0, 0, (j - 1) * 1.35]}>
                      <group position={[-0.92, 0, 0]}>
                        {tensor(
                          [
                            data.trace.query_weights,
                            data.trace.key_weights,
                            data.trace.value_weights,
                          ][j],
                          ["#b9a0ff", "#75cfeb", "#f3bd8d"][j],
                          ["Wq", "Wk", "Wv"][j],
                          i,
                          { size: 0.85 },
                        )}
                      </group>
                      <Line
                        points={[
                          [-0.35, 0, 0],
                          [0.08, 0, 0],
                        ]}
                        color={["#b9a0ff", "#75cfeb", "#f3bd8d"][j]}
                        transparent
                        opacity={0.6}
                      />
                      <group position={[0.85, 0, 0]}>
                        {tensor(
                          rows(matrix),
                          ["#b9a0ff", "#75cfeb", "#f3bd8d"][j],
                          (expanded
                            ? [
                                "Q = LN(X)Wq + bq",
                                "K = LN(X)Wk + bk",
                                "V = LN(X)Wv + bv",
                              ]
                            : ["Q = XWq", "K = XWk", "V = XWv"])[j],
                          i,
                          {
                            selected: token - start,
                            offset: start,
                            size: 1.75,
                          },
                        )}
                      </group>
                    </group>
                  ),
                )}
              {i === 3 && (
                <>
                  {tensor(
                    rows(data.trace.blocks[0].attention[0][0]).map((r) =>
                      r.slice(start, end),
                    ),
                    item.color,
                    "MASKED ATTENTION A",
                    i,
                    {
                      size: 2.2,
                      selected: token - start,
                      offset: start,
                      causal: true,
                    },
                  )}
                  {expanded && (
                    <group position={[0, 0, -2.7]}>
                      {tensor(
                        rows(data.trace.selected_block!.ff_activation![0]).map(
                          (r) => r.slice(0, 16),
                        ),
                        "#f3bd8d",
                        `LAYER ${(data.trace.layer_index ?? 0) + 1} · GELU (16 / 128 channels)`,
                        i,
                        { size: 2.2, selected: token - start, offset: start },
                      )}
                      <Line
                        points={[
                          [0, 1.5, 0],
                          [-1.8, 1.5, 0],
                          [-1.8, -1.5, 2.7],
                          [0, -1.5, 2.7],
                        ]}
                        color="#f3bd8d"
                        dashed
                        dashSize={0.12}
                        gapSize={0.1}
                      />
                    </group>
                  )}
                  <group position={[0, -2.25, 0]}>
                    {tensor(
                      [
                        data.trace.final_hidden ??
                          data.trace.blocks[0].head_outputs[0][0].at(-1)!,
                      ],
                      "#9bd5e5",
                      expanded
                        ? "FINAL LAYERNORM · BOTH BLOCKS"
                        : "FINAL h = (A × V)last",
                      i,
                      { size: 1.6 },
                    )}
                  </group>
                </>
              )}
              {i === 4 && (
                <>
                  <group position={[0, 0.35, 0]}>
                    {tensor(
                      data.trace.output_weights,
                      item.color,
                      `Wout · ${data.config.d_model} × ${data.characters.length}`,
                      i,
                      { size: 2.5, changed: props.changed },
                    )}
                  </group>
                  <group position={[0, -1, 0]}>
                    {tensor(
                      [data.logits],
                      "#e9d5a4",
                      expanded ? "LOGITS = h × Wout + b" : "LOGITS = h × Wout",
                      i,
                      { size: 2.5 },
                    )}
                  </group>
                </>
              )}
              {i === 5 && (
                <OutputBars
                  data={data}
                  active={stage === i}
                  motion={motion}
                  onValue={onValue}
                />
              )}
            </group>
          ))}
        </>
      )}
      <OrbitControls
        ref={controls}
        makeDefault
        enableDamping={!props.reduced && !props.paused}
        dampingFactor={0.08}
        minZoom={12}
        maxZoom={180}
        minPolarAngle={0.2}
        maxPolarAngle={Math.PI * 0.8}
        enablePan
      />
      <CameraRig props={props} controls={controls} />
    </>
  );
}
export default function ExplorerScene(props: Props) {
  const portal = useRef<HTMLDivElement>(null);
  return (
    <div ref={portal} style={{ height: "100%", position: "relative" }}>
      <portalContext.Provider value={portal}>
        <Canvas
          orthographic
          dpr={[1, 1.5]}
          camera={{ position: [8, 6, 21], zoom: 45, near: 0.1, far: 150 }}
          gl={{
            antialias: true,
            alpha: false,
            powerPreference: "high-performance",
          }}
          frameloop={props.reduced || props.paused ? "demand" : "always"}
          fallback={<div>Use Diagram view when WebGL is unavailable.</div>}
        >
          <World {...props} />
        </Canvas>
      </portalContext.Provider>
    </div>
  );
}
