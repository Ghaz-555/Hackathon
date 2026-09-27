import { useEffect, useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts"
import "./App.css"

type Probabilities = Record<string, number>

function displayCharacter(character: string) {
  return character === " " ? "[space]" : character
}

function App() {
  const [sessionId, setSessionId] = useState("")
  const [status, setStatus] = useState("Connecting...")

  const [prompt, setPrompt] = useState("the ca")
  const [temperature, setTemperature] = useState(1)
  const [probabilities, setProbabilities] = useState<Probabilities>({})
  const [attention, setAttention] = useState<number[][]>([])

  const [trainingLosses, setTrainingLosses] = useState<number[]>([])
  const [heldOutLosses, setHeldOutLosses] = useState<number[]>([])
  const [isTraining, setIsTraining] = useState(false)

  const [generatedText, setGeneratedText] = useState("")
  const [seed, setSeed] = useState(42)

  useEffect(() => {
    async function createSession() {
      try {
        const response = await fetch("http://127.0.0.1:8000/session", {
          method: "POST",
        })

        const data = await response.json()

        setSessionId(data.session_id)
        setStatus("Model connected")
      } catch (error) {
        console.error(error)
        setStatus("Connection failed")
      }
    }

    createSession()
  }, [])

  async function predict() {
    if (!sessionId) return

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        text: prompt,
        temperature,
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      alert(data.detail)
      return
    }

    setProbabilities(data.probabilities)
    setAttention(data.attention_weights)
  }

  async function trainModel() {
    if (!sessionId) return

    setIsTraining(true)

    try {
      const response = await fetch("http://127.0.0.1:8000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          text: "the cat sat on the mat",
          epochs: 10,
          learning_rate: 0.1,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        alert(data.detail)
        return
      }

      setTrainingLosses(data.training_losses)
      setHeldOutLosses(data.held_out_losses)
    } finally {
      setIsTraining(false)
    }
  }

  async function generateText() {
    if (!sessionId) return

    const response = await fetch("http://127.0.0.1:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        prompt,
        length: 20,
        temperature,
        seed,
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      alert(data.detail)
      return
    }

    setGeneratedText(data.generated_text)
  }

  const probabilityData = Object.entries(probabilities).sort(
    ([, a], [, b]) => b - a,
  )

  const lossData = trainingLosses.map((loss, index) => ({
    epoch: index + 1,
    training: loss,
    heldOut: heldOutLosses[index],
  }))

  return (
    <main className="app">
      <header className="hero">
        <div>
          <h1>
            Glass<span>Box</span>
          </h1>
          <p>See inside a language model.</p>
        </div>

        <div className="status">
          <span className="status-dot" />
          {status}
          {sessionId && " ✓"}
        </div>
      </header>

      <div className="dashboard">
        <section className="card">
          <div className="card-heading">
            <div className="icon">◉</div>
            <div>
              <h2>Temperature Experiment</h2>
              <p>
                See how temperature changes predictions without changing
                the model's weights.
              </p>
            </div>
          </div>

          <label>Prompt</label>
          <input
            className="text-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          <div className="temperature-label">
            <span>Temperature</span>
            <strong>{temperature.toFixed(1)}</strong>
          </div>

          <input
            className="slider"
            type="range"
            min="0.1"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />

          <div className="range-labels">
            <span>0.1</span>
            <span>1.0</span>
            <span>2.0</span>
          </div>

          <button className="primary-button" onClick={predict}>
            Predict Next Character →
          </button>

          {probabilityData.length > 0 && (
            <div className="probabilities">
              <h3>Next-character probabilities</h3>

              {probabilityData.map(([character, probability], index) => (
                <div className="probability-row" key={character}>
                  <span className="rank">{index + 1}</span>
                  <span className="character">
                    {displayCharacter(character)}
                  </span>

                  <div className="bar-track">
                    <div
                      className="bar"
                      style={{ width: `${probability * 500}%` }}
                    />
                  </div>

                  <span className="percentage">
                    {(probability * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-heading">
            <div className="icon cyan">▥</div>
            <div>
              <h2>Training Lab</h2>
              <p>
                Train the model and compare training loss with held-out loss.
              </p>
            </div>
          </div>

          <button
            className="primary-button training-button"
            onClick={trainModel}
            disabled={isTraining}
          >
            {isTraining ? "Training..." : "▶ Train for 10 Epochs"}
          </button>

          {lossData.length > 0 ? (
            <div className="chart-container">
              <h3>Loss over training</h3>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#293653" />
                  <XAxis dataKey="epoch" stroke="#9ba9c6" />
                  <YAxis
                    domain={["auto", "auto"]}
                    stroke="#9ba9c6"
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#111a2d",
                      border: "1px solid #354567",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="training"
                    name="Training Loss"
                    stroke="#54a8ff"
                    strokeWidth={3}
                  />
                  <Line
                    type="monotone"
                    dataKey="heldOut"
                    name="Held-out Loss"
                    stroke="#e36cff"
                    strokeWidth={3}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty-state">
              Run training to visualize how loss changes.
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-heading">
            <div className="icon">⌕</div>
            <div>
              <h2>Attention</h2>
              <p>
                See how much each character attends to earlier characters.
              </p>
            </div>
          </div>

          {attention.length === 0 ? (
            <div className="empty-state">
              Run a prediction to reveal attention.
            </div>
          ) : (
            <div className="attention-wrapper">
              <table className="attention-table">
                <thead>
                  <tr>
                    <th />
                    {prompt.split("").map((character, index) => (
                      <th key={index}>{displayCharacter(character)}</th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {attention.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      <th>{displayCharacter(prompt[rowIndex])}</th>

                      {prompt.split("").map((_, columnIndex) => {
                        const value = row[columnIndex]

                        return (
                          <td
                            key={columnIndex}
                            className={value === undefined ? "empty-cell" : ""}
                            style={
                              value !== undefined
                                ? {
                                    background: `rgba(164, 82, 255, ${
                                      0.15 + value * 0.85
                                    })`,
                                  }
                                : undefined
                            }
                          >
                            {value === undefined
                              ? "—"
                              : `${(value * 100).toFixed(0)}%`}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-heading">
            <div className="icon pink">✦</div>
            <div>
              <h2>Generate</h2>
              <p>
                Generate text with the current model, temperature, and seed.
              </p>
            </div>
          </div>

          <div className="generate-controls">
            <div>
              <label>Seed</label>
              <input
                className="text-input"
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </div>

            <button className="generate-button" onClick={generateText}>
              ✦ Generate Text
            </button>
          </div>

          <h3>Generated output</h3>

          <div className="output-box">
            {generatedText || "Generated text will appear here..."}
          </div>

          <div className="lesson-note">
            <strong>Try it:</strong> Keep the same seed and change only the
            temperature. The model weights stay the same, but sampling changes.
          </div>
        </section>
      </div>

      <footer>
        GlassBox · Tiny transformer-style character model built with NumPy
      </footer>
    </main>
  )
}

export default App