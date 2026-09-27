import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Inspection } from "./api";
export default function LossChartView({ data }: { data: Inspection | null }) {
  const latest = data?.history.at(-1),
    first = data?.history[0];
  return (
    <section className="card loss-card">
      <div className="card-header">
        <div>
          <p className="section-kicker">LEARNING, MEASURED</p>
          <h2>Practice vs. a fresh challenge</h2>
        </div>
        <div className="loss-stats">
          <div>
            <span>TRAINING LOSS</span>
            <strong>{latest?.train.toFixed(3) ?? "—"}</strong>
          </div>
          <div>
            <span>HELD-OUT LOSS</span>
            <strong>{latest?.validation.toFixed(3) ?? "—"}</strong>
          </div>
        </div>
      </div>
      <p className="muted">
        Lower is better. The curves evaluate next-character prediction on the
        training and held-out corpora. Loss is measured in nats per character.
      </p>
      <div className="loss-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data?.history ?? []}
            margin={{ top: 15, right: 25, left: 0, bottom: 10 }}
          >
            <CartesianGrid
              strokeDasharray="3 5"
              vertical={false}
              stroke="#e6e9e1"
            />
            <XAxis
              dataKey="step"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #e0e4da",
                fontSize: 12,
              }}
            />
            <Line
              name="Training loss"
              dataKey="train"
              stroke="#368369"
              strokeWidth={2.5}
              dot={data?.history.length === 1}
              isAnimationActive={false}
            />
            <Line
              name="Held-out loss"
              dataKey="validation"
              stroke="#b89b65"
              strokeWidth={2.5}
              dot={data?.history.length === 1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-foot">
        <span>
          <i className="legend-square" />
          Training <i className="legend-square heldout" />
          Held-out
        </span>
        <span>
          Baseline: {first?.train.toFixed(3)} training /{" "}
          {first?.validation.toFixed(3)} held-out · x-axis: optimizer step
        </span>
      </div>
    </section>
  );
}
