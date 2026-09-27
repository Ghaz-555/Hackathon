import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  CSS2DObject,
  CSS2DRenderer,
} from "three/addons/renderers/CSS2DRenderer.js";
import type { EmbeddingSpace } from "./math";

export type AnalogyView = {
  a: number;
  b: number;
  c: number;
  result: [number, number, number];
  nearest: number;
  steps: [number, number, number][];
  expression: string;
};
export class BrainScene {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(48, 1, 0.1, 180);
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  private labels = new CSS2DRenderer();
  private controls: OrbitControls;
  private points: THREE.Points;
  private marks = new THREE.Group();
  private paths = new THREE.Group();
  private plane: THREE.Mesh;
  private raf = 0;
  private disposed = false;
  private resize: ResizeObserver;
  private selected = 0;
  private neighbors: number[] = [];
  private visibleIndices: number[] = [];
  private slice = 0;
  private thickness = 30;
  private showLabels = true;
  private analogy: AnalogyView | null = null;
  private motion = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  private pulse: THREE.Mesh;
  private animationStart = 0;
  private animationProgress: number | null = null;
  private animatedArrows: THREE.Group[] = [];
  private animationOutput: HTMLElement | null = null;
  private arithmeticChart: HTMLElement | null = null;
  private pointerStart: [number, number] = [0, 0];
  constructor(
    private host: HTMLElement,
    private space: EmbeddingSpace,
    private onSelect: (index: number) => void,
    private onFailure: () => void,
  ) {
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    this.renderer.setClearColor("#0b111b");
    this.renderer.domElement.setAttribute(
      "aria-label",
      "3D projection of GPT-2 token embeddings",
    );
    this.renderer.domElement.addEventListener(
      "webglcontextlost",
      this.contextLost,
    );
    this.host.append(this.renderer.domElement);
    this.labels.domElement.className = "brain-label-layer";
    this.host.append(this.labels.domElement);
    this.camera.position.set(12, 8, 17);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 65;
    this.controls.autoRotateSpeed = 0.5;
    this.scene.fog = new THREE.FogExp2("#0b111b", 0.012);
    const grid = new THREE.GridHelper(32, 32, "#253444", "#172331");
    grid.position.y = -7;
    this.scene.add(grid);
    const axes = new THREE.AxesHelper(7);
    (axes.material as THREE.LineBasicMaterial).transparent = true;
    (axes.material as THREE.LineBasicMaterial).opacity = 0.32;
    this.scene.add(axes);
    const geometry = new THREE.BufferGeometry();
    this.points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 0.075,
        vertexColors: true,
        transparent: true,
        opacity: 0.65,
        sizeAttenuation: true,
      }),
    );
    this.scene.add(this.points, this.marks, this.paths);
    this.plane = new THREE.Mesh(
      new THREE.PlaneGeometry(22, 18),
      new THREE.MeshBasicMaterial({
        color: "#72dfc6",
        transparent: true,
        opacity: 0.025,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scene.add(this.plane);
    this.pulse = new THREE.Mesh(
      new THREE.SphereGeometry(0.11, 12, 12),
      new THREE.MeshBasicMaterial({ color: "#ffe0a0" }),
    );
    this.pulse.visible = false;
    this.scene.add(this.pulse);
    this.renderer.domElement.addEventListener("pointerdown", this.pointerDown);
    this.renderer.domElement.addEventListener("pointerup", this.pointerUp);
    this.resize = new ResizeObserver(this.size);
    this.resize.observe(host);
    this.size();
    this.rebuild();
    this.tick();
  }
  private contextLost = (event: Event) => {
    event.preventDefault();
    this.onFailure();
  };
  private size = () => {
    const { width, height } = this.host.getBoundingClientRect();
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.labels.setSize(width, height);
  };
  private pointerDown = (e: PointerEvent) => {
    this.pointerStart = [e.clientX, e.clientY];
  };
  private pointerUp = (e: PointerEvent) => {
    if (
      Math.hypot(
        e.clientX - this.pointerStart[0],
        e.clientY - this.pointerStart[1],
      ) > 5
    )
      return;
    const rect = this.host.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.params.Points.threshold = 0.18;
    ray.setFromCamera(
      new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        (-(e.clientY - rect.top) / rect.height) * 2 + 1,
      ),
      this.camera,
    );
    const hit = ray.intersectObject(this.points)[0];
    if (hit?.index !== undefined) this.onSelect(this.visibleIndices[hit.index]);
  };
  private tick = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.tick);
    this.controls.update();
    if (this.analogy) {
      const t =
        this.animationProgress ??
        (this.motion
          ? Math.min(1, (performance.now() - this.animationStart) / 6400)
          : 1);
      const part = Math.min(2, Math.floor(t * 3));
      const fraction = Math.min(1, t * 3 - part);
      this.animatedArrows.forEach((arrow) =>
        arrow.scale.setScalar(
          Math.max(0.00001, Math.min(1, t * 3 - arrow.userData.step)),
        ),
      );
      this.pulse.visible = this.motion && t < 1;
      this.pulse.position.lerpVectors(
        new THREE.Vector3(...this.analogy.steps[part]),
        new THREE.Vector3(...this.analogy.steps[part + 1]),
        fraction,
      );
      if (this.arithmeticChart)
        this.arithmeticChart.dataset.phase = String(t >= 1 ? 3 : part);
      if (this.animationOutput) {
        this.animationOutput.textContent =
          t >= 1
            ? "Sum complete · nearest neighbors ranked in 768D"
            : [
                "1 / 3 · Scale A",
                "2 / 3 · Add scaled B",
                "3 / 3 · Add scaled C",
              ][part];
        this.animationOutput.style.setProperty(
          "--arithmetic-progress",
          `${t * 100}%`,
        );
      }
    } else this.pulse.visible = false;
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  };
  private clear(group: THREE.Group) {
    group.traverse((object) => {
      if (object instanceof CSS2DObject) object.element.remove();
      if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => material.dispose());
      }
    });
    group.clear();
  }
  private label(
    text: string,
    position: THREE.Vector3,
    color: string,
    important = false,
    offset = [0, 0],
  ) {
    if (!this.showLabels && !important) return;
    const wrapper = document.createElement("div");
    const element = document.createElement("span");
    element.className = "token-label" + (important ? " important" : "");
    element.textContent = text;
    element.style.setProperty("--token-color", color);
    element.style.display = "inline-block";
    element.style.transform = `translate(${offset[0]}px,${offset[1]}px)`;
    wrapper.append(element);
    const label = new CSS2DObject(wrapper);
    label.position.copy(position).add(new THREE.Vector3(0, 0.2, 0));
    this.marks.add(label);
  }
  private dot(position: THREE.Vector3, color: string, size = 0.11) {
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(size, 14, 14),
      new THREE.MeshBasicMaterial({ color }),
    );
    dot.position.copy(position);
    this.marks.add(dot);
  }
  private arrow(
    from: THREE.Vector3,
    to: THREE.Vector3,
    color: string,
    step: number,
  ) {
    const delta = to.clone().sub(from),
      length = delta.length();
    if (length < 1e-7) return;
    // Own the geometry so clearing one scene never disposes shared arrow buffers.
    const group = new THREE.Group();
    group.position.copy(from);
    group.userData.step = step;
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        delta.clone(),
      ]),
      new THREE.LineBasicMaterial({ color }),
    );
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(
        Math.min(0.09, length * 0.1),
        Math.min(0.26, length * 0.2),
        10,
      ),
      new THREE.MeshBasicMaterial({ color }),
    );
    cone.position.copy(delta);
    cone.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      delta.normalize(),
    );
    group.add(line, cone);
    this.paths.add(group);
    this.animatedArrows.push(group);
  }
  private rebuild() {
    const { tokens } = this.space.manifest;
    const force = new Set([this.selected, ...this.neighbors]);
    if (this.analogy)
      [this.analogy.a, this.analogy.b, this.analogy.c].forEach((i) =>
        force.add(i),
      );
    this.visibleIndices = tokens.flatMap((token, i) =>
      this.thickness >= 30 ||
      Math.abs(token.position[2] - this.slice) <= this.thickness / 2 ||
      force.has(i)
        ? [i]
        : [],
    );
    const positions: number[] = [],
      colors: number[] = [];
    this.visibleIndices.forEach((i) => {
      positions.push(...tokens[i].position);
      const color = new THREE.Color().setHSL(
        0.48 + Math.max(-0.13, Math.min(0.13, tokens[i].position[0] / 45)),
        0.5,
        0.6,
      );
      colors.push(color.r, color.g, color.b);
    });
    this.points.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    this.points.geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(colors, 3),
    );
    this.points.geometry.computeBoundingSphere();
    this.clear(this.marks);
    this.clear(this.paths);
    this.animatedArrows = [];
    const decorated = new Set<number>();
    const mark = (
      i: number,
      color: string,
      important = false,
      offset = [0, 0],
    ) => {
      if (decorated.has(i)) return;
      decorated.add(i);
      const position = new THREE.Vector3(...tokens[i].position);
      this.dot(position, color, important ? 0.15 : 0.085);
      this.label(tokens[i].label, position, color, important, offset);
    };
    if (this.analogy) {
      const { a, b, c, result, nearest } = this.analogy;
      mark(a, "#f2c17b", true, [-25, 25]);
      mark(b, "#e79097", true, [20, 15]);
      mark(c, "#b2a1fa", true, [10, -25]);
      mark(nearest, "#88ead0", true, [-30, -35]);
      const position = new THREE.Vector3(...result);
      this.dot(position, "#faf2d4", 0.14);
      this.label(this.analogy.expression, position, "#faf2d4", true, [35, -5]);
      const steps = this.analogy.steps;
      this.label(
        "zero vector",
        new THREE.Vector3(...steps[0]),
        "#8ea0b8",
        true,
        [-20, 20],
      );
      for (let i = 0; i < 3; i++) {
        this.arrow(
          new THREE.Vector3(...steps[i]),
          new THREE.Vector3(...steps[i + 1]),
          ["#f2c17b", "#e79097", "#b2a1fa"][i],
          i,
        );
        if (i < 2)
          this.label(
            ["αA", "αA + βB"][i],
            new THREE.Vector3(...steps[i + 1]),
            ["#f2c17b", "#e79097"][i],
            true,
            [15, -30],
          );
      }
      const geometry = new THREE.BufferGeometry().setFromPoints([
        position,
        new THREE.Vector3(...tokens[nearest].position),
      ]);
      const line = new THREE.Line(
        geometry,
        new THREE.LineDashedMaterial({
          color: "#8bd9c1",
          dashSize: 0.13,
          gapSize: 0.09,
          transparent: true,
          opacity: 0.7,
        }),
      );
      line.computeLineDistances();
      this.paths.add(line);
    }
    mark(this.selected, "#88ead0", true);
    this.neighbors.forEach((i) => mark(i, "#8fd4c7"));
    // Background labels are sparse; selecting a point reveals its real neighbors.
    this.visibleIndices
      .filter(
        (_, i) =>
          i % Math.max(1, Math.floor(this.visibleIndices.length / 22)) === 0,
      )
      .slice(0, 22)
      .forEach((i) => {
        if (!decorated.has(i))
          this.label(
            tokens[i].label,
            new THREE.Vector3(...tokens[i].position),
            "#6e8597",
          );
      });
    this.plane.position.z = this.slice;
    this.plane.visible = this.thickness < 25;
  }
  select(
    index: number,
    neighbors: number[],
    analogy: AnalogyView | null = null,
  ) {
    this.selected = index;
    this.neighbors = neighbors;
    this.analogy = analogy;
    this.animationStart = performance.now();
    this.animationProgress = null;
    this.rebuild();
  }
  replay() {
    this.animationStart = performance.now();
    this.animationProgress = null;
  }
  scrub(progress: number) {
    this.animationProgress = Math.max(0, Math.min(1, progress));
  }
  setAnimationOutput(element: HTMLElement, chart: HTMLElement) {
    this.animationOutput = element;
    this.arithmeticChart = chart;
  }
  setSlice(center: number, thickness: number) {
    this.slice = center;
    this.thickness = thickness;
    this.rebuild();
    return this.visibleIndices.length;
  }
  setLabels(value: boolean) {
    this.showLabels = value;
    this.rebuild();
  }
  setRotate(value: boolean) {
    this.controls.autoRotate = value && this.motion;
  }
  setMotion(value: boolean) {
    this.motion = value;
    if (!value) this.controls.autoRotate = false;
  }
  reset() {
    this.camera.position.set(12, 8, 17);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
  focus() {
    const p = new THREE.Vector3(
      ...this.space.manifest.tokens[this.selected].position,
    );
    this.controls.target.copy(p);
    this.camera.position.copy(p).add(new THREE.Vector3(4, 3, 7));
    this.controls.update();
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resize.disconnect();
    this.controls.dispose();
    this.renderer.domElement.removeEventListener(
      "webglcontextlost",
      this.contextLost,
    );
    this.renderer.domElement.removeEventListener(
      "pointerdown",
      this.pointerDown,
    );
    this.renderer.domElement.removeEventListener("pointerup", this.pointerUp);
    this.clear(this.marks);
    this.clear(this.paths);
    this.scene.traverse((object) => {
      if (
        object instanceof THREE.Mesh ||
        object instanceof THREE.Points ||
        object instanceof THREE.Line
      ) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((m) => m.dispose());
      }
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labels.domElement.remove();
  }
}
