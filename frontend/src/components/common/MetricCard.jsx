import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export default function MetricCard({ title, value, sub, trend, icon: Icon, accent = 'blue' }) {
  const accentMap = {
    blue:    'text-blue-400 bg-blue-500/10',
    green:   'text-emerald-400 bg-emerald-500/10',
    red:     'text-red-400 bg-red-500/10',
    yellow:  'text-yellow-400 bg-yellow-500/10',
    purple:  'text-purple-400 bg-purple-500/10',
  }

  const trendIcon =
    trend === 'up'   ? <TrendingUp  size={13} className="text-emerald-400" /> :
    trend === 'down' ? <TrendingDown size={13} className="text-red-400" />    :
    trend === 'none' ? <Minus        size={13} className="text-slate-500" />  : null

  return (
    <div className="bg-card p-5 flex flex-col gap-3">
      <div className="flex items-start justify-between">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</p>
        {Icon && (
          <span className={`flex items-center justify-center w-8 h-8 rounded-lg text-sm ${accentMap[accent] ?? accentMap.blue}`}>
            <Icon size={15} />
          </span>
        )}
      </div>
      <div>
        <p className="text-2xl font-bold text-white leading-none">{value ?? '—'}</p>
        {(sub || trendIcon) && (
          <div className="flex items-center gap-1.5 mt-1.5">
            {trendIcon}
            {sub && <p className="text-xs text-slate-400">{sub}</p>}
          </div>
        )}
      </div>
    </div>
  )
}
