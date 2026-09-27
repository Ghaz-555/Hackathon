import {
  Component,
  lazy,
  Suspense,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  ArrowRight,
  Box,
  ChevronLeft,
  ChevronRight,
  Focus,
  Layers3,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import { useLab } from "./useLab";
import { stages } from "./stages";
import { tokenLabel } from "./api";
const Scene = lazy(() => import("./ExplorerScene"));
const LossChart = lazy(() => import("./LossChart"));
class CanvasBoundary extends Component<
  { children: ReactNode; onFailure: () => void },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onFailure();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}
const number = (v: number) => v.toFixed(5);
function Values({ values, label }: { values: number[]; label: string }) {
  return (
    <div className="vector">
      <span>{label}</span>
      <div>
        {values.map((v, i) => (
          <code key={i} title={`Dimension ${i}: ${v}`}>
            {number(v)}
          </code>
        ))}
      </div>
    </div>
  );
}
export default function App() {
  const lab = useLab();
  const { data } = lab;
  const [stage, setStage] = useState(0),
    [token, setToken] = useState(0),
    [playing, setPlaying] = useState(false),
    [focus, setFocus] = useState(false),
    [cameraReset, setCameraReset] = useState(0);
  const [draft, setDraft] = useState(lab.prompt),
    [drawer, setDrawer] = useState<"sample" | "train" | null>(null),
    [seed, setSeed] = useState(7),
    [hover, setHover] = useState("Hover over a tile to inspect its value");
  const [reduced, setReduced] = useState(
    () => matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [diagram, setDiagram] = useState(
    () => new URLSearchParams(location.search).get("view") === "diagram",
  );
  useEffect(() => {
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const cb = () => setReduced(media.matches);
    media.addEventListener("change", cb);
    return () => media.removeEventListener("change", cb);
  }, []);
  useEffect(() => {
    if (reduced) setPlaying(false);
  }, [reduced]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setStage((i) => {
          if (i === 5) {
            setPlaying(false);
            return i;
          }
          return i + 1;
        }),
      4500,
    );
    return () => clearInterval(timer);
  }, [playing]);
  useEffect(() => {
    setToken(Math.max(0, (data?.trace.tokens.length ?? 1) - 1));
  }, [data?.prompt]);
  useEffect(() => {
    const escape = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawer(null);
    };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, []);
  const select = (i: number) => {
    setStage(i);
    setPlaying(false);
    setHover("Hover over a tile to inspect its value");
  };
  const selected = stages[stage],
    row = Math.min(token, (data?.trace.tokens.length ?? 1) - 1);
  function inspect(e: React.FormEvent) {
    e.preventDefault();
    const normalized = draft.toLowerCase();
    if (!normalized.length) return;
    const unknown = [
      ...new Set([...normalized].filter((c) => !data?.characters.includes(c))),
    ];
    if (unknown.length) {
      lab.setError(
        `Unknown characters: ${unknown.map((c) => JSON.stringify(c)).join(", ")}. Try “the cat”.`,
      );
      return;
    }
    lab.setError("");
    lab.setPrompt(normalized);
    setDraft(normalized);
    setPlaying(false);
    setStage(0);
  }
  const probabilities =
    data?.probabilities
      .map((p, i) => ({ p, c: data.characters[i], i }))
      .sort((a, b) => b.p - a.p) ?? [];
  return (
    <div className="explorer-app">
      <header className="explorer-header">
        <a className="explorer-brand" href="#explorer">
          <Box size={26} />
          glasswork<span>.</span>
        </a>
        <span className="header-divider" />
        <span className="header-caption">THE MODEL, OPENED UP</span>
        <div className="header-right">
          <a className="subtle-button" href="/brain.html">
            Embedding space ↗
          </a>
          <span className="live-dot" />
          {data
            ? `${data.parameter_count} parameters · ${data.config.n_heads} attention head`
            : "Connecting to your model"}
          <button
            className="subtle-button"
            onClick={() => setDiagram((v) => !v)}
          >
            {diagram ? "3D view" : "Diagram view"}
          </button>
        </div>
      </header>
      <div className="explorer-layout" id="explorer">
        <section
          className="world-panel"
          aria-label="Interactive model explorer"
        >
          <div className="world-heading">
            <p className="overline">SMALL MODEL. EVERY OPERATION VISIBLE.</p>
            <h1>Step inside a prediction.</h1>
            <p>
              Follow a character. Open a layer. Watch the numbers become a
              possibility.
            </p>
          </div>
          <form className="floating-prompt" onSubmit={inspect}>
            <label htmlFor="model-prompt">YOUR INPUT</label>
            <input
              id="model-prompt"
              aria-label="Model prompt"
              value={draft}
              maxLength={data?.config.context_length ?? 128}
              onChange={(e) => setDraft(e.target.value)}
              disabled={lab.busy || !data}
              spellCheck={false}
            />
            <button
              disabled={!lab.ready || !draft.length}
              title="Run forward pass"
              aria-label="Inspect prompt"
            >
              <ArrowRight size={18} />
            </button>
          </form>
          {lab.error && (
            <div className="error-banner" role="alert">
              <span>{lab.error}</span>
              <button onClick={lab.recover} disabled={lab.busy}>
                Reconnect
              </button>
            </div>
          )}
          <div
            className="scene-container"
            data-testid="scene"
            aria-busy={lab.pending}
          >
            {!data ? (
              <div className="scene-loading">
                <Box size={35} />
                <span>
                  {lab.error ? "Waiting for connection" : "Opening your model…"}
                </span>
              </div>
            ) : diagram ? (
              <div className="diagram-view">
                <p className="overline">ACCESSIBLE DIAGRAM · SAME LIVE MODEL</p>
                <div className="diagram-stations">
                  {stages.map((s, i) => (
                    <button
                      key={s.name}
                      className={stage === i ? "active" : ""}
                      onClick={() => select(i)}
                    >
                      <span>0{i + 1}</span>
                      <Box size={30} />
                      <strong>{s.name}</strong>
                      <small>{s.short}</small>
                      <ArrowRight size={15} />
                    </button>
                  ))}
                </div>
                <p>Select a stage to see its actual values in the inspector.</p>
              </div>
            ) : (
              <CanvasBoundary onFailure={() => setDiagram(true)}>
                <Suspense
                  fallback={
                    <div className="scene-loading">
                      Loading the 3D explorer…
                    </div>
                  }
                >
                  <Scene
                    data={data}
                    stage={stage}
                    token={row}
                    playing={playing}
                    reduced={reduced}
                    onStage={select}
                    onValue={setHover}
                    reset={cameraReset}
                    focus={focus}
                    changed={lab.changed}
                    onFailure={() => setDiagram(true)}
                  />
                </Suspense>
              </CanvasBoundary>
            )}
          </div>
          <div className="world-toolbar">
            <span>
              <i /> {lab.pending ? "Computing…" : "Live computation"} <b> / </b>{" "}
              Animated explanation, not execution timing
            </span>
            <div>
              <button
                title="Focus selected stage"
                aria-label="Focus selected stage"
                aria-pressed={focus}
                onClick={() => setFocus((v) => !v)}
                disabled={diagram}
              >
                <Focus size={17} />
              </button>
              <button
                title="Reset camera"
                aria-label="Reset camera"
                onClick={() => {
                  setFocus(false);
                  setCameraReset((v) => v + 1);
                }}
                disabled={diagram}
              >
                <Maximize2 size={17} />
              </button>
              <button
                className={reduced ? "on" : ""}
                aria-pressed={reduced}
                onClick={() => setReduced((v) => !v)}
              >
                Reduce motion
              </button>
            </div>
          </div>
          <div className="scene-hint">
            <span>
              {diagram
                ? "All stages are keyboard accessible"
                : "Live tensor flow · drag to orbit · scroll to zoom"}
            </span>
            <code>{hover}</code>
          </div>
          <nav className="stage-timeline" aria-label="Model stages">
            {stages.map((s, i) => (
              <button
                key={s.name}
                onClick={() => select(i)}
                className={stage === i ? "active" : ""}
                aria-current={stage === i ? "step" : undefined}
              >
                <span>0{i + 1}</span>
                <strong>{s.name}</strong>
                <small>{s.short}</small>
              </button>
            ))}
          </nav>
          <div className="transport">
            <div>
              <button
                className="play-button"
                disabled={!data}
                onClick={() => {
                  if (stage === 5) setStage(0);
                  setPlaying((v) => !v);
                }}
                aria-label={playing ? "Pause walkthrough" : "Play walkthrough"}
              >
                {playing ? <Pause size={16} /> : <Play size={16} />}
              </button>
              <span>
                {playing
                  ? "Following the forward pass…"
                  : "Follow the forward pass"}
                <small>
                  {reduced
                    ? "Reduced motion · stages advance without moving particles"
                    : "Guided camera tour · six real operations."}
                </small>
              </span>
            </div>
            <span className="step-counter">
              0{stage + 1}
              <b> / 06</b>
            </span>
            <button
              className="experiment-button"
              onClick={() => setDrawer("sample")}
              disabled={!data}
            >
              <SlidersHorizontal size={15} />
              Experiment
            </button>
          </div>
        </section>
        <aside
          className="inspector"
          aria-label="Selected stage inspector"
          style={{ "--stage-color": selected.color } as React.CSSProperties}
        >
          <div className="inspector-top">
            <span>INSIDE THE MODEL</span>
            <span>0{stage + 1} / 06</span>
          </div>
          <div className="stage-glyph">
            <Layers3 size={30} />
          </div>
          <p className="overline">{selected.short}</p>
          <h2>{selected.name}</h2>
          <p className="stage-description">{selected.description}</p>
          <div className="token-picker">
            <label htmlFor="token">FOLLOW A CHARACTER</label>
            <select
              id="token"
              aria-label="Selected character"
              value={row}
              disabled={!data}
              onChange={(e) => {
                setToken(+e.target.value);
                setPlaying(false);
              }}
            >
              {data?.trace.tokens.map((c, i) => (
                <option key={i} value={i}>
                  {i + 1} · {tokenLabel(c)}
                  {i === data.trace.tokens.length - 1
                    ? " · final position"
                    : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="equation">
            <span>THE OPERATION</span>
            <code>
              {stage === 5 && lab.temperature === 0
                ? "p = one_hot(argmax(z))"
                : selected.formula}
            </code>
          </div>
          {data && (
            <div className="stage-values" key={`${stage}-${row}`}>
              {stage === 0 && (
                <Values
                  label={`Character “${tokenLabel(data.trace.tokens[row])}” · 4 dimensions`}
                  values={data.trace.token_embeddings[row]}
                />
              )}
              {stage === 1 && (
                <>
                  <Values
                    label="Character vector E"
                    values={data.trace.token_embeddings[row]}
                  />
                  <Values
                    label={`Position vector P · index ${row}`}
                    values={data.trace.position_embeddings[row]}
                  />
                  <Values
                    label="Combined E + P"
                    values={data.trace.embedding_sum[0][row]}
                  />
                </>
              )}
              {stage === 2 && (
                <>
                  {(["queries", "keys", "values"] as const).map((field, i) => (
                    <Values
                      key={field}
                      label={["Query Q", "Key K", "Value V"][i]}
                      values={data.trace[field][row]}
                    />
                  ))}
                </>
              )}
              {stage === 3 && (
                <>
                  <p className="data-caption">
                    Attention from position {row + 1}. Future positions have
                    weight zero.
                  </p>
                  <div className="attention-list">
                    {data.trace.blocks[0].attention[0][0][row]
                      .slice(0, row + 1)
                      .map((weight, i) => (
                        <div key={i}>
                          <span>
                            {i + 1} · {tokenLabel(data.trace.tokens[i])}
                          </span>
                          <div>
                            <i style={{ width: `${weight * 100}%` }} />
                          </div>
                          <code>{(weight * 100).toFixed(2)}%</code>
                        </div>
                      ))}
                  </div>
                  <Values
                    label="Weighted values · A × V"
                    values={data.trace.blocks[0].head_outputs[0][0][row]}
                  />
                </>
              )}
              {stage === 4 && (
                <>
                  <p className="data-caption">
                    Prediction uses position {data.trace.tokens.length}, the
                    final character.
                  </p>
                  <Values
                    label="Final hidden representation"
                    values={data.trace.blocks[0].head_outputs[0][0].at(-1)!}
                  />
                  <div className="weight-matrix" aria-label="Output weights">
                    {data.trace.output_weights.flatMap((r, i) =>
                      r.map((v, j) => (
                        <span
                          key={`${i}-${j}`}
                          title={`Wout[${i},${j}] = ${v}`}
                          className={
                            lab.changed && Math.abs(lab.changed[i][j]) > 1e-12
                              ? "changed"
                              : ""
                          }
                          style={{
                            opacity: 0.3 + Math.min(Math.abs(v), 1) * 0.7,
                          }}
                        />
                      )),
                    )}
                  </div>
                  <p className="data-caption">
                    4 × 10 · {data.trainable_parameter_count} trainable weights{" "}
                    {lab.changed
                      ? "· amber = changed since training started"
                      : ""}
                  </p>
                  <button
                    className="inspector-action"
                    onClick={() => setDrawer("train")}
                  >
                    <Sparkles size={14} />
                    Watch these weights learn
                  </button>
                </>
              )}
              {stage === 5 && (
                <>
                  <div className="probability-list">
                    {probabilities.map(({ p, c }) => (
                      <div key={c}>
                        <span>{tokenLabel(c)}</span>
                        <div>
                          <i style={{ width: `${p * 100}%` }} />
                        </div>
                        <code>{(p * 100).toFixed(2)}%</code>
                      </div>
                    ))}
                  </div>
                  <p className="data-caption">
                    T = {data.temperature.toFixed(2)} · entropy{" "}
                    {data.entropy.toFixed(3)} nats
                  </p>
                  <button
                    className="inspector-action"
                    onClick={() => setDrawer("sample")}
                  >
                    <SlidersHorizontal size={14} />
                    Explore temperature
                  </button>
                </>
              )}
            </div>
          )}
          <details className="reading-key">
            <summary>How to read this view</summary>
            <p>{selected.detail}</p>
            <p>
              The 3D layout is an educational diagram. Values are captured from
              your model. For long prompts, the scene shows up to 16 positions
              around the selected character; the inspector retains the full
              attention row.
            </p>
          </details>
          <div className="inspector-nav">
            <button
              onClick={() => select(Math.max(0, stage - 1))}
              disabled={stage === 0}
            >
              <ChevronLeft size={16} />
              Back
            </button>
            <button
              onClick={() => select(Math.min(5, stage + 1))}
              disabled={stage === 5}
            >
              Next stage
              <ChevronRight size={16} />
            </button>
          </div>
        </aside>
      </div>
      <footer className="explorer-footer">
        <span>GLASSWORK / A SMALL MODEL WITH NOTHING TO HIDE</span>
        <span>
          {data?.step ?? "—"} updates · weights{" "}
          {data?.fingerprint.slice(0, 10) ?? "…"} ·{" "}
          {data?.trainable_parameter_count ?? 40} trainable parameters
        </span>
      </footer>
      {drawer && (
        <section className="lab-drawer" aria-label="Experiment controls">
          <div className="drawer-heading">
            <div>
              <button
                className={drawer === "sample" ? "active" : ""}
                onClick={() => setDrawer("sample")}
              >
                Sampling lab
              </button>
              <button
                className={drawer === "train" ? "active" : ""}
                onClick={() => {
                  setDrawer("train");
                  select(4);
                }}
              >
                Learning lab
              </button>
            </div>
            <button
              aria-label="Close experiments"
              onClick={() => setDrawer(null)}
            >
              <X size={19} />
            </button>
          </div>
          {lab.error && <p className="lab-message" role="status">{lab.error} <button disabled={lab.busy} onClick={lab.recover}>Retry connection</button></p>}
          {drawer === "sample" ? (
            <div className="sampling-controls">
              <div>
                <p className="overline">CHANGE THE CHOICE, KEEP THE WEIGHTS</p>
                <h3>Turn up the unexpected.</h3>
                <label htmlFor="temperature">
                  Temperature <strong>{lab.temperature.toFixed(2)}</strong>
                </label>
                <input
                  id="temperature"
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={lab.temperature}
                  disabled={lab.busy || !data}
                  onChange={(e) => {
                    lab.setTemperature(+e.target.value);
                    select(5);
                  }}
                />
                <div className="range-labels">
                  <span>0 · greedy</span>
                  <span>2 · more varied</span>
                </div>
                <div className="lab-probabilities" aria-label="Live sampling probabilities" aria-busy={lab.pending}>
                  {probabilities.map(({ p, c }) => <div key={c} title={`${tokenLabel(c)}: ${(p * 100).toFixed(2)}%`}>
                    <span style={{ height: `${p * 90 + 2}px` }} />
                    <b>{tokenLabel(c)}</b><small>{(p * 100).toFixed(1)}%</small>
                  </div>)}
                </div>
                <p className="lab-caption">{lab.pending ? "Updating distribution…" : `Entropy ${data?.entropy.toFixed(3)} nats · weights unchanged`}</p>
              </div>
              <div className="sample-box">
                <div>
                  <label>
                    Seed{" "}
                    <input
                      aria-label="Sampling seed"
                      type="number"
                      value={seed}
                      min={0}
                      max={4294967295}
                      disabled={lab.busy}
                      onChange={(e) =>
                        setSeed(
                          Math.min(4294967295, Math.max(0, Math.floor(+e.target.value))),
                        )
                      }
                    />
                  </label>
                  <button
                    className="primary-action"
                    disabled={!lab.ready}
                    onClick={() => lab.generate(seed)}
                  >
                    <Play size={13} />
                    Generate 64 characters
                  </button>
                </div>
                <pre>
                  {lab.sample ||
                    "Your continuation will appear here. Same model + settings + seed gives the same text."}
                </pre>
                <small>
                  A ten-character vocabulary and a tiny training corpus:
                  imperfect text is expected.
                </small>
              </div>
            </div>
          ) : (
            <div className="learning-controls">
              <div>
                <p className="overline">ONLY THE OUTPUT PROJECTION LEARNS</p>
                <h3>Watch the amber weights change.</h3>
                <p>
                  One step updates 40 output weights with SGD. The attention and
                  embedding stages stay fixed.
                </p>
                <p className="lab-lesson">{data?.mode === "trained" ? "You are viewing a trained checkpoint. Choose Random weights to see learning from the beginning, then train and compare the curves." : "You are learning from a fresh start. Each update uses the training sentence below; your input only changes the prediction being inspected."}</p>
                <p className="lab-corpus">Practice: <code>{data?.training_text}</code><br />Held out: <code>{data?.validation_text}</code></p>
                <div className="learning-actions">
                  <button
                    disabled={!lab.ready}
                    onClick={() => {
                      select(4);
                      lab.train(1);
                    }}
                  >
                    Train 1 step
                  </button>
                  <button
                    className="primary-action"
                    disabled={!lab.ready && !lab.training}
                    onClick={() => {
                      select(4);
                      lab.training ? lab.stop() : lab.train(25);
                    }}
                  >
                    {lab.training ? (
                      <>
                        <Square size={13} />
                        Stop
                      </>
                    ) : (
                      <>
                        <Sparkles size={13} />
                        Train 25 steps
                      </>
                    )}
                  </button>
                </div>
                <div className="reset-actions">
                  <button
                    disabled={!lab.ready}
                    onClick={() => lab.reset("random")}
                  >
                    <RotateCcw size={12} />
                    Random weights
                  </button>
                  <button
                    disabled={!lab.ready}
                    onClick={() => lab.reset("trained")}
                  >
                    Restore checkpoint
                  </button>
                </div>
                <p className="training-progress" aria-live="polite">
                  {lab.training
                    ? `Learning… ${lab.completed} steps complete`
                    : `${lab.completed} steps completed this run · ${data?.step} total updates`}
                </p>
              </div>
              <Suspense fallback={<p>Loading loss curves…</p>}>
                <LossChart data={data} />
              </Suspense>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
