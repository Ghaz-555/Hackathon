import { useEffect, useRef, useState } from "react";
import { api, ApiError, type Inspection, type ModelState } from "./api";
// Share only session creation, not model state, across StrictMode effect checks.
let boot: Promise<string> | null = null;
function session() {
  if (!boot)
    boot = (async () => {
      const old = sessionStorage.getItem("glasswork-session");
      if (old) {
        try {
          await api(`/sessions/${old}`);
          return old;
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) throw e;
        }
      }
      const value = await api<{ session_id: string }>("/sessions", {});
      sessionStorage.setItem("glasswork-session", value.session_id);
      return value.session_id;
    })().catch((e) => {
      boot = null;
      throw e;
    });
  return boot;
}
export function useLab() {
  const [sid, setSid] = useState(""),
    [data, setData] = useState<Inspection | null>(null),
    [error, setError] = useState("");
  const [prompt, setPrompt] = useState("the cat"),
    [temperature, setTemperature] = useState(1),
    [busy, setBusy] = useState(false),
    [pending, setPending] = useState(true);
  const [sample, setSample] = useState(""),
    [training, setTraining] = useState(false),
    [completed, setCompleted] = useState(0),
    [changed, setChanged] = useState<number[][] | null>(null);
  const [retry, setRetry] = useState(0);
  const sequence = useRef(0),
    stop = useRef(false);
  const fail = (e: unknown) =>
    setError(
      e instanceof Error ? e.message : "Request failed. Please try again.",
    );
  useEffect(() => {
    let alive = true;
    session()
      .then((s) => {
        if (alive) setSid(s);
      })
      .catch((e) => {
        if (alive) {
          fail(e);
          setPending(false);
        }
      });
    return () => {
      alive = false;
    };
  }, [retry]);
  useEffect(() => {
    if (!sid) return;
    // A slow response from an older prompt must never replace a newer trace.
    const id = ++sequence.current;
    setPending(true);
    setChanged(null);
    setSample("");
    const timer = setTimeout(() => {
      api<Inspection>(`/sessions/${sid}/inspect`, { prompt, temperature })
        .then((d) => {
          if (id !== sequence.current) return;
          if (!(d.trace.schema_version >= 4))
            throw new Error(
              "Restart the Glasswork server to load the new 3D trace.",
            );
          setData(d);
          setError("");
        })
        .catch((e) => {
          if (id === sequence.current) fail(e);
        })
        .finally(() => {
          if (id === sequence.current) setPending(false);
        });
    }, 120);
    return () => {
      clearTimeout(timer);
      sequence.current++;
    };
  }, [sid, prompt, temperature, retry]);
  async function generate(seed: number) {
    setBusy(true);
    setError("");
    try {
      const r = await api<{ text: string }>(`/sessions/${sid}/sample`, {
        prompt,
        temperature,
        seed,
        count: 64,
      });
      setSample(r.text);
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  async function reset(mode: "random" | "trained") {
    if (!data) return;
    setBusy(true);
    setChanged(null);
    setSample("");
    setError("");
    try {
      await api<ModelState>(`/sessions/${sid}/reset`, {
        revision: data.revision,
        mode,
      });
      setData(
        await api<Inspection>(`/sessions/${sid}/inspect`, {
          prompt,
          temperature,
        }),
      );
      setCompleted(0);
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  async function train(count: number) {
    if (!data) return;
    setBusy(true);
    setTraining(true);
    setError("");
    setSample("");
    setCompleted(0);
    stop.current = false;
    // Highlight real deltas against the start of this run, not random effects.
    const before = data.trace.output_weights;
    let revision = data.revision;
    try {
      // One request per update allows stopping without rolling back a completed step.
      for (let i = 0; i < count && !stop.current; i++) {
        const r = await api<{ after: Inspection }>(`/sessions/${sid}/train`, {
          revision,
          prompt,
          temperature,
          steps: 1,
        });
        revision = r.after.revision;
        setChanged(
          r.after.trace.output_weights.map((row, rindex) =>
            row.map((v, c) => v - before[rindex][c]),
          ),
        );
        setData(r.after);
        setCompleted(i + 1);
      }
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
      setTraining(false);
    }
  }
  async function recover() {
    if (sid) await api(`/sessions/${sid}`, undefined, "DELETE").catch(() => {});
    boot = null;
    sessionStorage.removeItem("glasswork-session");
    setSid("");
    setData(null);
    setError("");
    setPending(true);
    setRetry((v) => v + 1);
  }
  return {
    data,
    error,
    setError,
    prompt,
    setPrompt,
    temperature,
    setTemperature,
    busy,
    pending,
    ready: !!data && !busy && !pending,
    sample,
    generate,
    reset,
    train,
    training,
    completed,
    changed,
    stop: () => {
      stop.current = true;
    },
    recover,
  };
}
