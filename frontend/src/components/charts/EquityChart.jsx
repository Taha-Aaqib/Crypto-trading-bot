import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
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

export default function EquityChart({ data = [] }) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm">
        No closed trades yet
      </div>
    )
  }

  const lastVal = data[data.length - 1]?.pnl ?? 0
  const color   = lastVal >= 0 ? '#34d399' : '#f87171'
  const gradId  = lastVal >= 0 ? 'gradGreen' : 'gradRed'

  const formatted = data.map(d => ({
    ...d,
    time: d.time ? new Date(d.time).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '',
  }))

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={formatted} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="gradGreen" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#34d399" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#34d399" stopOpacity={0}   />
          </linearGradient>
          <linearGradient id="gradRed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#f87171" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#f87171" stopOpacity={0}   />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="time"
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={v => `$${v}`}
          width={55}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 2" />
        <Area
          type="monotone"
          dataKey="pnl"
          stroke={color}
          strokeWidth={2}
          fill={`url(#${gradId})`}
          dot={false}
          activeDot={{ r: 4, fill: color }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
