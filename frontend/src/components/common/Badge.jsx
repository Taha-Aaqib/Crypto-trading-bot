const STYLES = {
  long:    'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  short:   'bg-red-500/15 text-red-400 border-red-500/30',
  buy:     'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  sell:    'bg-red-500/15 text-red-400 border-red-500/30',
  open:    'bg-blue-500/15 text-blue-400 border-blue-500/30',
  closed:  'bg-slate-500/15 text-slate-400 border-slate-500/30',
  pending: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
  filled:  'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  expired: 'bg-slate-500/15 text-slate-400 border-slate-500/30',
  INFO:    'bg-blue-500/15 text-blue-400 border-blue-500/30',
  WARNING: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
  ERROR:   'bg-red-500/15 text-red-400 border-red-500/30',
  DEBUG:   'bg-purple-500/15 text-purple-400 border-purple-500/30',
  CRITICAL:'bg-red-700/20 text-red-300 border-red-600/40',
  smc:     'bg-blue-500/15 text-blue-400 border-blue-500/30',
  ml:      'bg-purple-500/15 text-purple-400 border-purple-500/30',
  sentiment:'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
}

export default function Badge({ label, variant }) {
  const key = variant ?? label?.toLowerCase()
  const cls = STYLES[key] ?? 'bg-slate-500/15 text-slate-400 border-slate-500/30'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md border text-xs font-medium ${cls}`}>
      {label}
    </span>
  )
}
