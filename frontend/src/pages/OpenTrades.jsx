import { useEffect } from 'react'
import { Clock, AlertTriangle } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import { fetchOpenTrades, fetchPendingOrders } from '../services/api'
import Badge from '../components/common/Badge'
import LoadingSpinner from '../components/common/LoadingSpinner'
import EmptyState from '../components/common/EmptyState'

function pnlColor(val) {
  if (val == null) return 'text-slate-500'
  return val >= 0 ? 'text-emerald-400' : 'text-red-400'
}

export default function OpenTrades() {
  const { data: openData, loading: oLoading, refetch: rO } = usePolling(fetchOpenTrades, 8000)
  const { data: ordersData, loading: ordLoading, refetch: rOrd } = usePolling(fetchPendingOrders, 8000)

  useEffect(() => {
    const h = () => { rO(); rOrd() }
    window.addEventListener('dashboard-refresh', h)
    return () => window.removeEventListener('dashboard-refresh', h)
  }, [rO, rOrd])

  const trades  = openData?.trades ?? []
  const pending = (ordersData?.orders ?? []).filter(o => o.status === 'pending')

  return (
    <div className="space-y-6">
      {/* Open positions */}
      <div className="bg-card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-700 flex items-center justify-between">
          <p className="text-sm font-semibold text-white">Open Positions</p>
          <Badge label={`${trades.length} active`} variant="open" />
        </div>

        {oLoading ? <LoadingSpinner label="Loading positions…" /> :
         !trades.length ? <EmptyState title="No open positions" message="The bot has no active trades." icon={AlertTriangle} /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-slate-800/60">
                <tr className="text-slate-400">
                  {['ID','Symbol','Dir','Entry','Stop Loss','Take Profit','Size','Partial TP','Split Entry','Since'].map(h => (
                    <th key={h} className="px-4 py-3 text-left font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {trades.map(t => (
                  <tr key={t.id} className="hover:bg-slate-700/30 transition-colors">
                    <td className="px-4 py-3 font-mono text-slate-500">#{t.id}</td>
                    <td className="px-4 py-3 font-semibold text-white">{t.symbol}</td>
                    <td className="px-4 py-3"><Badge label={t.direction} /></td>
                    <td className="px-4 py-3 font-mono">${t.entry_price?.toFixed(2)}</td>
                    <td className="px-4 py-3 font-mono text-red-400">${t.stop_loss?.toFixed(2)}</td>
                    <td className="px-4 py-3 font-mono text-emerald-400">${t.take_profit?.toFixed(2)}</td>
                    <td className="px-4 py-3 font-mono">${t.quantity?.toFixed(2)}</td>
                    <td className="px-4 py-3">
                      {t.partial_tp_taken
                        ? <Badge label="Done" variant="filled" />
                        : <span className="text-slate-600 text-xs">—</span>}
                    </td>
                    <td className="px-4 py-3">
                      {t.is_split_entry
                        ? <Badge label={t.split_entry_filled ? 'Filled' : 'Pending'} variant={t.split_entry_filled ? 'filled' : 'pending'} />
                        : <span className="text-slate-600 text-xs">—</span>}
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
                      {t.entry_time ? new Date(t.entry_time).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pending limit orders */}
      <div className="bg-card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock size={15} className="text-yellow-400" />
            <p className="text-sm font-semibold text-white">Pending Split-Entry Orders</p>
          </div>
          <Badge label={`${pending.length} pending`} variant="pending" />
        </div>

        {ordLoading ? <LoadingSpinner label="Loading orders…" /> :
         !pending.length ? <EmptyState title="No pending limit orders" message="Split-entry limit orders will appear here." /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-slate-800/60">
                <tr className="text-slate-400">
                  {['Order ID','Trade ID','Symbol','Side','Limit Price','Size','Status','Created','Expires'].map(h => (
                    <th key={h} className="px-4 py-3 text-left font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {pending.map(o => (
                  <tr key={o.id} className="hover:bg-slate-700/30 transition-colors">
                    <td className="px-4 py-3 font-mono text-slate-500">#{o.id}</td>
                    <td className="px-4 py-3 font-mono text-slate-500">#{o.trade_id}</td>
                    <td className="px-4 py-3 font-semibold text-white">{o.symbol}</td>
                    <td className="px-4 py-3"><Badge label={o.side} /></td>
                    <td className="px-4 py-3 font-mono">${o.limit_price?.toFixed(2)}</td>
                    <td className="px-4 py-3 font-mono">${o.quantity?.toFixed(2)}</td>
                    <td className="px-4 py-3"><Badge label={o.status} /></td>
                    <td className="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
                      {o.created_time ? new Date(o.created_time).toLocaleString() : '—'}
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
                      {o.expiry_time ? new Date(o.expiry_time).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
