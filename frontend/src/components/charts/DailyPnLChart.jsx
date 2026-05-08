import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const val = payload[0].value
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 shadow-xl text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      <p className={`font-semibold ${val >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        {val >= 0 ? '+' : ''}${val.toFixed(2)}
      </p>
    </div>
  )
}

export default function DailyPnLChart({ data = [] }) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm">
        No daily data yet
      </div>
    )
  }
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={d => d.slice(5)}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={v => `$${v}`}
          width={50}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="#475569" />
        <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.pnl >= 0 ? '#34d399' : '#f87171'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
