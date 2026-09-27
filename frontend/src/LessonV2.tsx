import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  FlaskConical,
  Layers3,
  Play,
  RotateCcw,
  Sparkles,
  Square,
  Terminal,
  Thermometer,
  X,
} from "lucide-react";
import {
  api,
  ApiError,
  tokenLabel,
  type Inspection,
  type ModelState,
} from "./api";

const LossChart = lazy(() => import("./LossChart"));

type Lesson = "temperature" | "learning";
// Deduplicate initial session acquisition, including React StrictMode effect checks.
let bootPromise: Promise<{ session_id: string } & ModelState> | null = null;
function acquireSession() {
  if (!bootPromise)
    bootPromise = (async () => {
      const saved = sessionStorage.getItem("glasswork-session");
      if (saved) {
        try {
          return {
            session_id: saved,
            ...(await api<ModelState>(`/sessions/${saved}`)),
          };
        } catch (error) {
          if (!(error instanceof ApiError && error.status === 404)) throw error;
        }
      }
      const session = await api<{ session_id: string } & ModelState>(
        "/sessions",
        {},
      );
      sessionStorage.setItem("glasswork-session", session.session_id);
      return session;
    })().catch((error) => {
      bootPromise = null;
      throw error;
    });
  return bootPromise;
}

export default function App() {
  const [sid, setSid] = useState("");
  const [data, setData] = useState<Inspection | null>(null);
  const [lesson, setLesson] = useState<Lesson>("temperature");
  const [draft, setDraft] = useState("the "),
    [prompt, setPrompt] = useState("the ");
  const [temperature, setTemperature] = useState(1);
  const [busy, setBusy] = useState(false),
    [inspecting, setInspecting] = useState(true);
  const [error, setError] = useState(""),
    [sample, setSample] = useState("");
  const [seed, setSeed] = useState(7),
    [head, setHead] = useState(0),
    [block, setBlock] = useState(0);
  const [answer, setAnswer] = useState<string | null>(null),
    [notes, setNotes] = useState(false);
  const [before, setBefore] = useState<{
    probabilities: number[];
    fingerprint: string;
    step: number;
  } | null>(null);
  const [training, setTraining] = useState(false),
    [completed, setCompleted] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const stop = useRef(false),
    requestId = useRef(0);
  const notesTrigger = useRef<HTMLElement | null>(null);
  function openNotes() {
    notesTrigger.current = document.activeElement as HTMLElement;
    setNotes(true);
  }
  const ready = !!data && !busy && !inspecting;

  function report(error: unknown) {
    setError(
      error instanceof Error
        ? error.message
        : "Something went wrong. Please try again.",
    );
  }
  useEffect(() => {
    let active = true;
    acquireSession()
      .then((session) => {
        if (active) setSid(session.session_id);
      })
      .catch((e) => {
        if (active) {
          report(e);
          setInspecting(false);
        }
      });
    return () => {
      active = false;
    };
  }, [refresh]);

  // A sequence number prevents a slow old response overwriting a newer experiment.
  useEffect(() => {
    if (!sid) return;
    const id = ++requestId.current;
    setInspecting(true);
    setSample("");
    setBefore(null);
    const timer = setTimeout(() => {
      api<Inspection>(`/sessions/${sid}/inspect`, { prompt, temperature })
        .then((value) => {
          if (id === requestId.current) {
            setData(value);
            setError("");
          }
        })
        .catch((e) => {
          if (id === requestId.current) report(e);
        })
        .finally(() => {
          if (id === requestId.current) setInspecting(false);
        });
    }, 100);
    return () => {
      clearTimeout(timer);
      requestId.current++;
    };
  }, [sid, prompt, temperature, refresh]);

  useEffect(() => {
    if (!notes) return;
    const previous = notesTrigger.current;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNotes(false);
      if (event.key !== "Tab" || !dialog) return;
      const buttons = [
        ...dialog.querySelectorAll<HTMLElement>("button, a, input, select"),
      ];
      const first = buttons[0],
        last = buttons.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      }
      if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      previous?.focus();
    };
  }, [notes]);

  async function reset(mode: "random" | "trained") {
    if (!data) return;
    setBusy(true);
    setError("");
    setBefore(null);
    setSample("");
    setCompleted(0);
    requestId.current++;
    try {
      await api(`/sessions/${sid}/reset`, { revision: data.revision, mode });
      setData(
        await api<Inspection>(`/sessions/${sid}/inspect`, {
          prompt,
          temperature,
        }),
      );
    } catch (e) {
      report(e);
    } finally {
      setBusy(false);
    }
  }
  async function sampleText() {
    setBusy(true);
    setError("");
    try {
      const value = await api<{ text: string }>(`/sessions/${sid}/sample`, {
        prompt,
        temperature,
        seed,
        count: 64,
      });
      setSample(value.text);
    } catch (e) {
      report(e);
    } finally {
      setBusy(false);
    }
  }
  async function train(count: number) {
    if (!data) return;
    let revision = data.revision;
    setBusy(true);
    setTraining(true);
    setCompleted(0);
    setError("");
    setSample("");
    stop.current = false;
    requestId.current++;
    setBefore({
      probabilities: [...data.probabilities],
      fingerprint: data.fingerprint,
      step: data.step,
    });
    try {
      for (let i = 0; i < count && !stop.current; i++) {
        const response = await api<{ after: Inspection }>(
          `/sessions/${sid}/train`,
          { prompt, temperature, revision, steps: 1 },
        );
        revision = response.after.revision;
        setData(response.after);
        setCompleted(i + 1);
      }
    } catch (e) {
      report(e);
    } finally {
      setBusy(false);
      setTraining(false);
    }
  }
  async function recover() {
    stop.current = true;
    if (sid) await api(`/sessions/${sid}`, undefined, "DELETE").catch(() => {});
    sessionStorage.removeItem("glasswork-session");
    bootPromise = null;
    setSid("");
    setData(null);
    setError("");
    setInspecting(true);
    setRefresh((v) => v + 1);
  }
  function applyPrompt(e: React.FormEvent) {
    e.preventDefault();
    if (!draft.length) return;
    const normalized = draft.toLowerCase();
    const unknown = [...normalized].filter(
      (c) => !data?.characters.includes(c),
    );
    if (unknown.length) {
      setError(
        `Unknown characters: ${[...new Set(unknown)].map((c) => JSON.stringify(c)).join(", ")}. Try known lowercase text such as “the cat ”.`,
      );
      return;
    }
    setBefore(null);
    setDraft(normalized);
    setPrompt(normalized);
  }
  const top = data
    ? data.probabilities
        .map((p, i) => ({ p, i, label: tokenLabel(data.characters[i]) }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 8)
    : [];
  const maxProbability = Math.max(
    ...top.map((x) =>
      Math.max(
        x.p,
        lesson === "learning" ? (before?.probabilities[x.i] ?? 0) : 0,
      ),
    ),
    0.01,
  );
  const attention = data?.trace.blocks[block]?.attention[0][head];
  const visibleTokens = data?.trace.tokens ?? [...prompt];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#main" aria-label="Glasswork home">
          <span className="brand-icon">
            <Box size={25} />
          </span>
          glasswork<span className="brand-dot">.</span>
        </a>
        <div className="workspace">
          <span className="workspace-icon">
            <FlaskConical size={18} />
          </span>
          <div>
            Your personal lab<small>Small model. Real mathematics.</small>
          </div>
          <span className="tiny-dot" />
        </div>
        <p className="nav-label">THE LEARNING LAB</p>
        <nav aria-label="Lessons">
          <button
            aria-label="Temperature lab"
            disabled={busy}
            className={
              lesson === "temperature" ? "nav-item active" : "nav-item"
            }
            onClick={() => {
              setLesson("temperature");
              setAnswer(null);
            }}
          >
            <Thermometer size={18} />
            <span>Temperature lab</span>
            <span className="nav-number">01</span>
          </button>
          <button
            aria-label="Learning lab"
            disabled={busy}
            className={lesson === "learning" ? "nav-item active" : "nav-item"}
            onClick={() => {
              setLesson("learning");
              setAnswer(null);
            }}
          >
            <Sparkles size={18} />
            <span>Learning lab</span>
            <span className="nav-number">02</span>
          </button>
        </nav>
        <div className="sidebar-note">
          <Layers3 size={21} />
          <h3>Real model. Open book.</h3>
          <p>
            Every probability comes from a team-built attention model using
            NumPy. You can change it and see what happens.
          </p>
          <button onClick={openNotes}>
            Meet the model <ArrowUpRight size={14} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <span className="initials">G</span>
          <div>
            Glasswork V1<small>Interactive AI classroom</small>
          </div>
          <span className="version">02</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            Learning lab <ChevronRight size={14} />
            <strong>
              {lesson === "temperature" ? "Temperature" : "Learning"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="connection">
              <i className={sid && !error ? "connected" : ""} />
              {sid ? "Local model connected" : "Connecting to model"}
            </span>
            <button
              className="icon-button"
              aria-label="About this model"
              onClick={openNotes}
            >
              <CircleHelp size={18} />
            </button>
          </div>
        </header>
        <main id="main">
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                <span /> EXPERIMENT {lesson === "temperature" ? "01" : "02"} /
                HANDS-ON AI
              </p>
              <h1>
                {lesson === "temperature" ? (
                  <>
                    A little randomness.
                    <br />A lot to discover.
                  </>
                ) : (
                  <>
                    Watch a model
                    <br />
                    learn something new.
                  </>
                )}
              </h1>
              <p className="intro">
                {lesson === "temperature"
                  ? "Open the black box. Change the temperature, follow the probabilities, and find out what really changes."
                  : "Give the model practice. Watch its weights change and compare what it predicts before and after."}
              </p>
            </div>
            <div className="model-badge">
              <span className="model-badge-icon">
                <Box size={22} />
              </span>
              <div>
                {data?.model_name ?? "Loading model"}
                <small>
                  {data ? data.parameter_count.toLocaleString() : "…"}{" "}
                  parameters
                </small>
              </div>
              <span className="badge">LIVE</span>
            </div>
          </div>
          {error && (
            <div className="error" role="alert">
              <span>{error}</span>
              <button onClick={recover} disabled={busy}>
                Start a fresh session
              </button>
            </div>
          )}
          <section
            className="journey card"
            aria-label="Model architecture overview"
          >
            <div className="journey-copy">
              <span className="section-kicker">ONE CHARACTER AT A TIME</span>
              <h2>
                From your words
                <br /> to what comes next.
              </h2>
              <button className="text-link" onClick={openNotes}>
                See how it works <ArrowRight size={14} />
              </button>
            </div>
            <div className="journey-track">
              <div className="journey-stage">
                <div className="token-stack">
                  {[...prompt].slice(-4).map((c, i) => (
                    <span key={i}>{tokenLabel(c)}</span>
                  ))}
                </div>
                <span>01 / Your input</span>
              </div>
              <span className="flow-line" />
              <div className="journey-stage">
                <div
                  className="matrix-object"
                  aria-label="Single causal attention layer"
                >
                  <div className="matrix-layer front">
                    {Array.from({ length: 16 }, (_, i) => (
                      <i key={i} />
                    ))}
                  </div>
                </div>
                <span>02 / Single attention head</span>
              </div>
              <span className="flow-line" />
              <div className="journey-stage">
                <div className="prediction-token">
                  {top[0]?.label ?? "…"}
                  <span>
                    {top[0] ? (top[0].p * 100).toFixed(1) + "%" : "loading"}
                  </span>
                </div>
                <span>03 / Next character</span>
              </div>
            </div>
            <span className="journey-caption">
              Conceptual overview · predictions are live
            </span>
          </section>
          <div className="experiment-grid">
            <section className="card probabilities">
              <div className="card-header">
                <div>
                  <p className="section-kicker">
                    {lesson === "temperature"
                      ? "THE PREDICTION"
                      : "THE COMPARISON"}
                  </p>
                  <h2>What comes next?</h2>
                </div>
                <span className="soft-tag">
                  {inspecting
                    ? "Updating…"
                    : `${data?.characters.length ?? "…"} possible characters`}
                </span>
              </div>
              <form className="prompt-form" onSubmit={applyPrompt}>
                <label htmlFor="prompt">Give the model a starting point</label>
                <div className="input-row">
                  <span className="prompt-mark">›</span>
                  <input
                    id="prompt"
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    maxLength={data?.config.context_length ?? 128}
                    disabled={busy || !data}
                    spellCheck={false}
                    autoComplete="off"
                  />
                  <button
                    aria-label="Inspect prompt"
                    disabled={!ready || !draft.length}
                    type="submit"
                  >
                    <ArrowRight size={19} />
                  </button>
                </div>
                <div className="input-caption">
                  <span>
                    Character-level model · input normalized to lowercase
                  </span>
                  <span>
                    {[...draft].length}/{data?.config.context_length ?? 128}
                  </span>
                </div>
              </form>
              <div className="chart-title">
                <span>Next-character probability</span>
                <span>TOP 8 OF {data?.characters.length ?? "…"}</span>
              </div>
              <div
                className={
                  "probability-chart " + (inspecting ? "updating" : "")
                }
                aria-label="Next character probabilities"
                aria-busy={inspecting}
              >
                {top.length ? (
                  top.map((item, index) => (
                    <div className="prob-row" key={item.i}>
                      <span className="token-label">{item.label}</span>
                      <div className="bar-track">
                        {lesson === "learning" && before && (
                          <div
                            className="before-bar"
                            style={{
                              width: `${Math.min((before.probabilities[item.i] / maxProbability) * 100, 100)}%`,
                            }}
                          />
                        )}
                        <div
                          className={"bar " + (index === 0 ? "bar-leader" : "")}
                          style={{
                            width: `${(item.p / maxProbability) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="prob-value">
                        {(item.p * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="loading">Loading the trained model…</div>
                )}
              </div>
              <div className="chart-foot">
                <span>
                  <i className="legend-square" />
                  {lesson === "learning" && before
                    ? "After training"
                    : "Current probability"}
                </span>
                {lesson === "learning" && before ? (
                  <span>
                    <i className="legend-square previous" />
                    Before training
                  </span>
                ) : (
                  <span>
                    ␣ space &nbsp; ↵ newline · bars scaled to the largest value
                  </span>
                )}
              </div>
              <div className="sample-panel">
                <div className="sample-heading">
                  <span>
                    <Terminal size={15} /> TRY A CONTINUATION
                  </span>
                  <button
                    disabled={!ready}
                    className="text-link"
                    onClick={sampleText}
                  >
                    <Play size={12} />
                    {busy && !training ? "Working…" : "Generate 64 characters"}
                  </button>
                </div>
                <p className={sample ? "generated" : "sample-placeholder"}>
                  {sample ||
                    "A tiny model, a real prediction. See where it takes your sentence."}
                </p>
                <div className="sample-settings">
                  <label>
                    Sampling seed{" "}
                    <input
                      aria-label="Sampling seed"
                      type="number"
                      min={0}
                      max={4294967295}
                      value={seed}
                      disabled={busy}
                      onChange={(e) =>
                        setSeed(
                          Math.min(
                            4294967295,
                            Math.max(0, Number(e.target.value)),
                          ),
                        )
                      }
                    />
                  </label>
                  <span>Same model + settings + seed → same text</span>
                </div>
              </div>
            </section>
            <div className="controls-column">
              <section className="card control-card">
                <p className="section-kicker">YOUR EXPERIMENT</p>
                <div className="control-title">
                  <h2>
                    {lesson === "temperature"
                      ? "Turn up the unexpected."
                      : "Practice changes weights."}
                  </h2>
                  <span className="control-icon">
                    {lesson === "temperature" ? (
                      <Thermometer size={21} />
                    ) : (
                      <Sparkles size={21} />
                    )}
                  </span>
                </div>
                {lesson === "temperature" ? (
                  <>
                    <p className="muted">
                      Temperature reshapes the chances of each next character.
                      Try moving the slider.
                    </p>
                    <div className="temperature-value">
                      <span>{temperature.toFixed(2)}</span>
                      <span>
                        {temperature === 0
                          ? "Greedy"
                          : temperature < 0.7
                            ? "More predictable"
                            : temperature > 1.3
                              ? "More varied"
                              : "Balanced"}
                      </span>
                    </div>
                    <label className="sr-only" htmlFor="temperature">
                      Temperature
                    </label>
                    <input
                      id="temperature"
                      className="slider"
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={temperature}
                      disabled={busy || !data}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                    />
                    <div className="slider-labels">
                      <span>0 · focused</span>
                      <span>2 · exploratory</span>
                    </div>
                    <div className="presets">
                      {[0.3, 1, 1.8].map((t) => (
                        <button
                          key={t}
                          disabled={busy || !data}
                          className={temperature === t ? "selected" : ""}
                          onClick={() => setTemperature(t)}
                        >
                          {t === 0.3
                            ? "Focused"
                            : t === 1
                              ? "Balanced"
                              : "Surprising"}
                        </button>
                      ))}
                    </div>
                    <div className="formula">
                      <span>THE MATH, MADE VISIBLE</span>
                      <code>pᵢ = exp(zᵢ / T) / Σ exp(zⱼ / T)</code>
                      <p>
                        {temperature === 0
                          ? "At zero, we select the largest logit directly."
                          : `At T = ${temperature.toFixed(2)}, entropy is ${data?.entropy.toFixed(3) ?? "…"} nats.`}
                      </p>
                    </div>
                    <div className="insight">
                      <Check size={16} />
                      <p>
                        <strong>Weights stay the same.</strong> Temperature
                        changes sampling, not what the model has learned.
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="muted">
                      Your team’s model learns an output mapping from fixed
                      attention features. Only {data?.trainable_parameter_count}{" "}
                      of {data?.parameter_count} weights change.
                    </p>
                    <p className="comparison-note">
                      Training text: “{data?.training_text}”<br />
                      Comparison temperature: {temperature.toFixed(2)} · held
                      fixed during training
                    </p>
                    <div className="training-state">
                      <span>MODEL STATE</span>
                      <strong>
                        {data?.mode === "random"
                          ? "Your training experiment"
                          : "Trained checkpoint"}
                      </strong>
                      <small>
                        Optimizer step {data?.step.toLocaleString() ?? "—"}
                      </small>
                    </div>
                    <button
                      className="button secondary full"
                      disabled={!ready}
                      onClick={() => reset("random")}
                    >
                      <RotateCcw size={15} />
                      Reset to random weights
                    </button>
                    <div className="training-buttons">
                      <button
                        className="button secondary"
                        disabled={!ready}
                        onClick={() => train(1)}
                      >
                        Train 1 step
                      </button>
                      <button
                        className="button primary"
                        disabled={!ready && !training}
                        onClick={() =>
                          training ? (stop.current = true) : train(25)
                        }
                      >
                        {training ? (
                          <>
                            <Square size={13} />
                            Stop
                          </>
                        ) : (
                          <>
                            <Play size={13} />
                            Train 25 steps
                          </>
                        )}
                      </button>
                    </div>
                    <div className="train-progress" aria-live="polite">
                      {training
                        ? `Learning… ${completed} steps complete. Stop takes effect after the current step.`
                        : completed
                          ? `${completed} steps complete. Your model’s weights changed.`
                          : "Each step trains on one sampled next-character example."}
                    </div>
                    <div className="insight">
                      <Sparkles size={16} />
                      <p>
                        <strong>This is actual learning.</strong>{" "}
                        {data?.optimizer} updates the output projection.
                        Embeddings and attention weights stay fixed.
                      </p>
                    </div>
                    <button
                      className="text-link restore"
                      disabled={!ready}
                      onClick={() => reset("trained")}
                    >
                      Restore trained checkpoint <ArrowUpRight size={13} />
                    </button>
                  </>
                )}
              </section>
              <section className="question-card">
                <span className="question-icon">
                  <CircleHelp size={19} />
                </span>
                <p className="section-kicker">PAUSE & PREDICT</p>
                <h3>
                  {lesson === "temperature"
                    ? "Does a higher temperature teach the model new things?"
                    : "Does lower training loss always mean better predictions on new text?"}
                </h3>
                <div className="answer-buttons">
                  <button
                    className={answer === "yes" ? "chosen" : ""}
                    onClick={() => setAnswer("yes")}
                  >
                    Yes, it does
                  </button>
                  <button
                    className={answer === "no" ? "chosen" : ""}
                    onClick={() => setAnswer("no")}
                  >
                    No, it doesn’t
                  </button>
                </div>
                {answer && (
                  <p className="answer-feedback" role="status">
                    {answer === "no" ? "Exactly. " : "Try this distinction: "}
                    {lesson === "temperature"
                      ? "The weights are unchanged. Only training updates what the model learns."
                      : "A model can memorize training examples. Held-out loss helps reveal overfitting."}
                  </p>
                )}
              </section>
            </div>
          </div>
          {lesson === "learning" && (
            <Suspense
              fallback={<div className="card loading">Loading loss chart…</div>}
            >
              <LossChart data={data} />
            </Suspense>
          )}
          <section className="bottom-grid">
            <div className="card attention-card">
              <div className="card-header">
                <div>
                  <p className="section-kicker">A PEEK INSIDE</p>
                  <h2>Where information flows</h2>
                </div>
                <div className="selects">
                  <select
                    aria-label="Attention layer"
                    value={block}
                    disabled={!ready}
                    onChange={(e) => setBlock(+e.target.value)}
                  >
                    {Array.from(
                      { length: data?.config.n_layers ?? 1 },
                      (_, i) => (
                        <option key={i} value={i}>
                          Layer {i + 1}
                        </option>
                      ),
                    )}
                  </select>
                  <select
                    aria-label="Attention head"
                    value={head}
                    disabled={!ready}
                    onChange={(e) => setHead(+e.target.value)}
                  >
                    {Array.from(
                      { length: data?.config.n_heads ?? 1 },
                      (_, i) => (
                        <option key={i} value={i}>
                          Head {i + 1}
                        </option>
                      ),
                    )}
                  </select>
                </div>
              </div>
              <div className="attention-body">
                <div className="heatmap-scroll">
                  <div
                    className="heatmap"
                    style={{
                      gridTemplateColumns: `22px repeat(${visibleTokens.length}, minmax(8px, 28px))`,
                    }}
                    role="img"
                    aria-label="Actual causal attention matrix. Rows are query characters, columns are earlier or current characters."
                  >
                    <span />
                    {visibleTokens.map((c, i) => (
                      <span className="heat-label" key={"c" + i}>
                        {tokenLabel(c)}
                      </span>
                    ))}
                    {attention?.map((row, r) => (
                      <div className="heat-row" key={r}>
                        <span className="heat-label">
                          {tokenLabel(visibleTokens[r])}
                        </span>
                        {row.map((value, c) => (
                          <span
                            key={c}
                            className={"heat-cell " + (c > r ? "masked" : "")}
                            style={
                              c <= r
                                ? {
                                    backgroundColor: `rgba(64, 133, 104, ${0.08 + value * 0.92})`,
                                  }
                                : undefined
                            }
                            title={`${tokenLabel(visibleTokens[r])} → ${tokenLabel(visibleTokens[c])}: ${c > r ? "future token masked" : (value * 100).toFixed(2) + "%"}`}
                          />
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="heat-explanation">
                  <p>
                    Darker squares mean stronger attention. Future characters
                    are masked.
                  </p>
                  <span>
                    Actual attention weights, not a complete explanation of the
                    model’s reasoning.
                  </span>
                </div>
              </div>
            </div>
            <div className="card notebook">
              <div className="notebook-icon">
                <Layers3 size={20} />
              </div>
              <p className="section-kicker">YOUR LAB NOTE</p>
              <h3>
                {lesson === "temperature"
                  ? "Sampling ≠ learning."
                  : "A better fit isn’t the whole story."}
              </h3>
              <p>
                {lesson === "temperature"
                  ? "A temperature slider changes how a model chooses. Training changes the model itself. That small difference is a big idea."
                  : "Watch both loss curves. A model can improve on familiar sentences while getting worse on sentences it hasn’t practiced."}
              </p>
              <button
                className="text-link"
                onClick={() => {
                  setLesson(
                    lesson === "temperature" ? "learning" : "temperature",
                  );
                  setAnswer(null);
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
              >
                Go to the{" "}
                {lesson === "temperature" ? "learning" : "temperature"} lab{" "}
                <ArrowRight size={14} />
              </button>
            </div>
          </section>
          <footer>
            <span>
              GLASSWORK <span className="footer-dot">/</span> Small enough to
              understand. Real enough to learn from.
            </span>
            <span title={data?.fingerprint}>
              Weights: {data?.fingerprint.slice(0, 10) ?? "connecting"} ·{" "}
              {data?.step ?? "—"} updates
            </span>
          </footer>
        </main>
      </div>
      {notes && (
        <div className="modal-backdrop" onClick={() => setNotes(false)}>
          <section
            className="modal card"
            role="dialog"
            aria-modal="true"
            aria-label="Meet the model"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="icon-button modal-close"
              aria-label="Close model notes"
              autoFocus
              onClick={() => setNotes(false)}
            >
              <X size={20} />
            </button>
            <span className="model-badge-icon">
              <Box size={26} />
            </span>
            <p className="eyebrow">SMALL MODEL, REAL MATH</p>
            <h2>Nothing up its sleeve.</h2>
            <p>
              Glasswork runs your team’s {data?.model_name} engine, built with
              NumPy. It has {data?.parameter_count} parameters. Training uses{" "}
              {data?.optimizer} to update {data?.trainable_parameter_count}{" "}
              output-projection weights; its embeddings and attention
              projections remain fixed.
            </p>
            <div className="spec-grid">
              <div>
                <strong>{data?.config.n_layers}</strong>
                <span>Attention layers</span>
              </div>
              <div>
                <strong>{data?.config.n_heads}</strong>
                <span>Attention heads</span>
              </div>
              <div>
                <strong>{data?.config.context_length}</strong>
                <span>Characters of context</span>
              </div>
              <div>
                <strong>{data?.characters.length}</strong>
                <span>Known characters</span>
              </div>
            </div>
            <p>
              Training corpus: “{data?.training_text}”. Held-out comparison: “
              {data?.validation_text}”. This is a tiny output-layer learning
              experiment with fixed attention, not a full transformer training
              run. Generated text is often incoherent.
            </p>
            <p>
              Your session has its own copy. Training here never overwrites the
              saved checkpoint. Sessions expire after an hour of inactivity.
            </p>
            <button className="button primary" onClick={() => setNotes(false)}>
              Back to the experiment <ArrowRight size={15} />
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
