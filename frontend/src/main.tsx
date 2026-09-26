import { loadSpace, type EmbeddingSpace, type Neighbor } from "./math";
import { BrainScene, type AnalogyView } from "./scene";
import "./style.css";

// building most of the page here instead of writing it separately in html
const root = document.querySelector<HTMLDivElement>("#brain-app")!;
root.innerHTML = `
<header class="brain-header"><a href="/brain.html" class="brand"><span>◈</span> glasswork.</a><span class="header-note">THE BRAIN OF AN LLM</span><span class="model-tag">GPT-2 · input embeddings</span></header>
<main class="brain-layout">
  <section class="universe" aria-label="Embedding explorer">
    <div class="scene-title"><span class="eyebrow">A SMALL WINDOW INTO 768 DIMENSIONS</span><h1>Words have coordinates.</h1><p>Explore a real embedding space. Follow a word.<br>Move through the relationships it learned.</p></div>
    <div id="space" class="space"><div id="loading" class="loading">Loading real GPT-2 vectors…</div></div>
    <div class="scene-status"><span id="point-count">Preparing the space</span><span>Drag to orbit · scroll to zoom · click a point</span></div>
    <div class="view-controls"><button id="reset-view">Reset view</button><button id="focus-token">Focus token</button><button id="rotate" aria-pressed="false">Rotate</button><button id="labels" aria-pressed="true">Labels</button><button id="motion" aria-pressed="false">Reduce motion</button></div>
    <div class="slice-controls"><div><span class="eyebrow">SLICE THE PROJECTION</span><p id="variance-note">Keep a band of the third principal component in view.</p></div><label for="depth">Depth <output id="depth-value" for="depth">0.0</output><input id="depth" type="range" min="-10" max="10" step=".1" value="0"></label><label for="width">Thickness <output id="width-value" for="width">All</output><input id="width" type="range" min=".5" max="30" step=".5" value="30"></label></div>
  </section>
  <aside class="brain-inspector"><span class="eyebrow">EXPLORE A TOKEN</span><form id="search-form"><label for="token-search" class="sr-only">Find a token</label><input id="token-search" list="token-options" value="king" autocomplete="off" spellcheck="false"><button aria-label="Find token">↗</button></form><datalist id="token-options"></datalist>
    <div class="token-summary"><h2 id="selected-name">king</h2><span id="token-id">Waiting for data</span><p id="token-description">Each point is one learned input vector—not a whole thought or the model’s reasoning.</p></div>
    <div class="neighbor-heading"><h3 id="neighbors-heading">Nearest in 768D</h3><span>COSINE</span></div><ol id="neighbors" class="neighbors"></ol>
    <section class="arithmetic"><span class="eyebrow">MOVE A RELATIONSHIP</span><h3>What happens if…</h3><form id="analogy-form"><div class="expression"><label>A<input id="a" aria-label="Starting token A" list="token-options" value="king"></label><b>−</b><label>B<input id="b" aria-label="Subtract token B" list="token-options" value="man"></label><b>+</b><label>C<input id="c" aria-label="Add token C" list="token-options" value="woman"></label></div><button class="calculate">Explore A − B + C <span>→</span></button></form><p id="analogy-note">Try king − man + woman. The answer comes from the vectors; it is not scripted.</p></section>
    <p id="error" role="alert" hidden></p>
    <details class="method"><summary>What am I actually seeing?</summary><p id="projection-note"></p><p>The scene uses PCA: three directions capturing the most variance in this 2,048-token subset. Closeness in this view can differ from closeness in the original vectors.</p><p>Search and analogy rankings use cosine similarity across all 768 original dimensions, among these 2,048 tokens. Inputs are excluded from analogy results. Arrows show the projected B → C displacement, and the same displacement applied to A.</p><p>All tokens here begin with a space in GPT-2’s tokenizer. Alphabetic tokens can still be word fragments. These are static input embeddings, not contextual representations. Selected tokens and their neighbors stay visible when slicing.</p><a id="data-source" target="_blank" rel="noreferrer">View pinned model source ↗</a></details>
  </aside>
</main><footer><span>REAL WEIGHTS. A PROJECTED VIEW.</span><span id="dataset-status">NumPy preparation · Three.js visualization</span></footer>`;


// just helpers so i dont have to keep casting everything
const element = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const input = (id: string) => element<HTMLInputElement>(id);

let space: EmbeddingSpace;
let scene: BrainScene | null = null;
let selected = 0;
let currentNeighbors: Neighbor[] = [];
let activeAnalogy: AnalogyView | null = null;

const reduced = matchMedia("(prefers-reduced-motion: reduce)");

const error = (message = "") => {
  element("error").textContent = message;
  element("error").hidden = !message;
};


// finds the token based on whatever was typed in the input
function tokenIndex(id: string) {
  const label = input(id).value.trim();
  const index = space.byLabel.get(label);

  if (index === undefined)
    throw new Error(
      `“${label}” is not in this token slice. Choose a token from the suggestions.`,
    );

  return index;
}


// rebuilding the list everytime since the neighbors change depending on the token
function showNeighbors(neighbors: Neighbor[]) {
  const list = element("neighbors");
  list.replaceChildren();

  neighbors.forEach(({ index, score }, rank) => {
    const li = document.createElement("li"),
      button = document.createElement("button");

    const number = document.createElement("span"),
      label = document.createElement("strong"),
      value = document.createElement("code");

    number.textContent = String(rank + 1).padStart(2, "0");
    label.textContent = space.manifest.tokens[index].label;
    value.textContent = score.toFixed(3);

    button.append(number, label, value);
    button.addEventListener("click", () => select(index));

    li.append(button);
    list.append(li);
  });
}


function showSelection(index: number) {
  selected = index;

  const token = space.manifest.tokens[index];

  input("token-search").value = token.label;
  element("selected-name").textContent = token.label;

  element("token-id").textContent =
    `TOKEN ${token.id} · “ ${token.label}” · 768 VALUES`;
}


// gets the real 768d neighbors then tells the scene what to highlight
function select(index: number) {
  error();
  showSelection(index);

  activeAnalogy = null;

  currentNeighbors = space.nearest(
    space.vector(index),
    new Set([index]),
  );

  showNeighbors(currentNeighbors);

  element("neighbors-heading").textContent = "Nearest in 768D";

  element("analogy-note").textContent =
    "Try king − man + woman. The answer comes from the vectors; it is not scripted.";

  scene?.select(
    index,
    currentNeighbors.map((n) => n.index),
  );
}


// doing the analogy in the full embedding space, not the 3d pca view
function runAnalogy(event: SubmitEvent) {
  event.preventDefault();

  if (!space) return;

  try {
    const a = tokenIndex("a"),
      b = tokenIndex("b"),
      c = tokenIndex("c");

    const query = space.analogy(a, b, c);

    currentNeighbors = space.nearest(
      query,
      new Set([a, b, c]),
    );

    showSelection(a);
    showNeighbors(currentNeighbors);
    error();

    element("neighbors-heading").textContent =
      "Nearest to A − B + C";

    activeAnalogy = {
      a,
      b,
      c,
      result: space.project(query),
      nearest: currentNeighbors[0].index,
    };

    scene?.select(
      a,
      currentNeighbors.map((n) => n.index),
      activeAnalogy,
    );

    const best =
      space.manifest.tokens[currentNeighbors[0].index].label;

    const labels = [a, b, c].map(
      (i) => space.manifest.tokens[i].label,
    );

    const queen =
      labels.join("|") === "king|man|woman"
        ? ` Queen ranks #${space.manifest.referenceAnalogy.queenRank} in this subset.`
        : "";

    element("analogy-note").textContent =
      `${labels[0]} − ${labels[1]} + ${labels[2]} → ${best} (cosine ${currentNeighbors[0].score.toFixed(3)}).${queen} This is a nearest match, not an exact equality.`;

  } catch (e) {
    error((e as Error).message);
  }
}


// if webgl doesnt work the vector stuff should still be usable
function fallback() {
  scene?.dispose();
  scene = null;

  const div = document.createElement("div");
  div.className = "webgl-fallback";

  const heading = document.createElement("h2");
  heading.textContent = "The vectors are still here.";

  const p = document.createElement("p");

  p.textContent =
    "3D rendering is unavailable. Search tokens and explore vector arithmetic in the inspector; those calculations still use the real data.";

  div.append(heading, p);

  element("space").replaceChildren(div);

  ["reset-view", "focus-token", "rotate", "labels", "depth", "width"].forEach(
    (id) => ((element(id) as HTMLButtonElement).disabled = true),
  );
}


// controls what part of the 3rd pca axis is visable
function slice() {
  const center = Number(input("depth").value),
    width = Number(input("width").value);

  const count =
    scene?.setSlice(center, width) ?? space.manifest.count;

  element("depth-value").textContent =
    center.toFixed(1);

  element("width-value").textContent =
    width >= 30 ? "All" : width.toFixed(1);

  element("point-count").textContent =
    `${count.toLocaleString()} / ${space.manifest.count.toLocaleString()} tokens visible`;
}


// shared logic for buttons that behave like on/off switches
function toggle(
  id: string,
  callback: (value: boolean) => void,
) {
  element(id).addEventListener("click", () => {
    const button = element(id);

    const value =
      button.getAttribute("aria-pressed") !== "true";

    button.setAttribute(
      "aria-pressed",
      String(value),
    );

    callback(value);
  });
}


element("search-form").addEventListener(
  "submit",
  (event) => {
    event.preventDefault();

    if (!space) return;

    try {
      select(tokenIndex("token-search"));
    } catch (e) {
      error((e as Error).message);
    }
  },
);


element("analogy-form").addEventListener(
  "submit",
  runAnalogy,
);

element("reset-view").addEventListener(
  "click",
  () => scene?.reset(),
);

element("focus-token").addEventListener(
  "click",
  () => scene?.focus(),
);

toggle(
  "rotate",
  (value) => scene?.setRotate(value),
);

toggle(
  "labels",
  (value) => scene?.setLabels(value),
);

toggle("motion", (value) => {
  scene?.setMotion(!value);

  if (value)
    element("rotate").setAttribute(
      "aria-pressed",
      "false",
    );
});


// keep this synced if the system motion prefrence changes
const motionChanged = () => {
  element("motion").setAttribute(
    "aria-pressed",
    String(reduced.matches),
  );

  scene?.setMotion(!reduced.matches);

  if (reduced.matches)
    element("rotate").setAttribute(
      "aria-pressed",
      "false",
    );
};


reduced.addEventListener(
  "change",
  motionChanged,
);

motionChanged();

input("depth").addEventListener(
  "input",
  () => space && slice(),
);

input("width").addEventListener(
  "input",
  () => space && slice(),
);

window.addEventListener(
  "pagehide",
  () => scene?.dispose(),
  { once: true },
);


// load the embedings before creating the actual scene
loadSpace()
  .then((data) => {
    space = data;

    const { manifest } = space;

    manifest.tokens.forEach((token) => {
      const option =
        document.createElement("option");

      option.value = token.label;

      element("token-options").append(option);
    });

    element("loading").remove();

    try {
      scene = new BrainScene(
        element("space"),
        space,
        select,
        fallback,
      );
    } catch (e) {
      console.warn("3D unavailable:", e);
      fallback();
    }

    element<HTMLAnchorElement>(
      "data-source",
    ).href = manifest.source;

    element("projection-note").textContent =
      `These three axes retain ${(100 * manifest.projection.variance.reduce((sum, n) => sum + n, 0)).toFixed(1)}% of the variance in this subset. The visual origin is the subset mean, not the zero embedding.`;

    element("variance-note").textContent =
      `3D retains ${(100 * manifest.projection.variance.reduce((sum, n) => sum + n, 0)).toFixed(1)}% of variance. Rankings use all 768 dimensions.`;

    element("dataset-status").textContent =
      `GPT-2 wte.weight · ${manifest.count.toLocaleString()} of ${manifest.fullVocabularySize.toLocaleString()} tokens · PCA → 3D`;

    select(space.byLabel.get("king")!);
    slice();

    document.body.dataset.ready = "true";
  })

  .catch((e) => {
    element("loading").textContent =
      (e as Error).message;

    error((e as Error).message);
  });