import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const RADIAN = Math.PI / 180
const renderLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
  if (percent < 0.05) return null
  const r = innerRadius + (outerRadius - innerRadius) * 0.5
  const x = cx + r * Math.cos(-midAngle * RADIAN)
  const y = cy + r * Math.sin(-midAngle * RADIAN)
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={12} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 shadow-xl text-xs">
      <p className="font-semibold text-white">{payload[0].name}: {payload[0].value}</p>
    </div>
  )
}

export default function WinLossChart({ wins = 0, losses = 0 }) {
  if (wins + losses === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm">
        No closed trades
      </div>
    )
  }

  const data = [
    { name: 'Wins',   value: wins   },
    { name: 'Losses', value: losses },
  ]

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={45}
          outerRadius={75}
          paddingAngle={3}
          dataKey="value"
          labelLine={false}
          label={renderLabel}
        >
          <Cell fill="#34d399" />
          <Cell fill="#f87171" />
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          iconType="circle"
          iconSize={8}
          formatter={v => <span className="text-xs text-slate-400">{v}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}
