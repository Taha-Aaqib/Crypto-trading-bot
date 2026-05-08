import { useEffect } from 'react'
import {
  TrendingUp, TrendingDown, Activity, BarChart3, Target, Percent,
} from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import { fetchMetrics, fetchOpenTrades } from '../services/api'
import MetricCard from '../components/common/MetricCard'
import LoadingSpinner from '../components/common/LoadingSpinner'
import EquityChart from '../components/charts/EquityChart'
import WinLossChart from '../components/charts/WinLossChart'
import Badge from '../components/common/Badge'

export default function Dashboard() {
  const {
    data: metrics, loading: mLoading, error: mError, refetch: refetchM,
  } = usePolling(fetchMetrics, 10000)

  const {
    data: openData, loading: oLoading, refetch: refetchO,
  } = usePolling(fetchOpenTrades, 10000)

  useEffect(() => {
    const handler = () => { refetchM(); refetchO() }
    window.addEventListener('dashboard-refresh', handler)
    return () => window.removeEventListener('dashboard-refresh', handler)
  }, [refetchM, refetchO])

  if (mLoading) return <LoadingSpinner label="Loading dashboard…" />

  if (mError) {
    return (
      <div className="bg-red-900/20 border border-red-700/40 rounded-xl p-6 text-center">
        <p className="text-red-400 font-medium">Could not connect to API</p>
        <p className="text-red-400/70 text-sm mt-1">{mError}</p>
        <p className="text-slate-500 text-xs mt-3">
          Make sure the API server is running: <code className="font-mono">uvicorn api.server:app --port 8000</code>
        </p>
      </div>
    )
  }

  const m = metrics ?? {}
  const pnlPositive = (m.total_pnl ?? 0) >= 0

  return (
    <div className="space-y-6">
      {/* KPI Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Trades"
          value={m.total_trades ?? 0}
          sub={`${m.open_trades ?? 0} open`}
          icon={Activity}
          accent="blue"
        />
        <MetricCard
          title="Win Rate"
          value={m.closed_trades ? `${m.win_rate}%` : '—'}
          sub={m.closed_trades ? `${m.winning_trades}W / ${m.losing_trades}L` : 'No closed trades'}
          trend={m.win_rate > 50 ? 'up' : m.win_rate < 50 ? 'down' : 'none'}
          icon={Percent}
          accent={m.win_rate >= 50 ? 'green' : 'red'}
        />
        <MetricCard
          title="Total P&L"
          value={m.total_pnl != null ? `$${m.total_pnl.toFixed(2)}` : '—'}
          sub={pnlPositive ? 'Profitable' : 'In drawdown'}
          trend={pnlPositive ? 'up' : 'down'}
          icon={pnlPositive ? TrendingUp : TrendingDown}
          accent={pnlPositive ? 'green' : 'red'}
        />
        <MetricCard
          title="Profit Factor"
          value={m.profit_factor ?? '—'}
          sub={m.profit_factor >= 1.5 ? 'Excellent' : m.profit_factor >= 1 ? 'Profitable' : 'Below 1:1'}
          icon={Target}
          accent={m.profit_factor >= 1.5 ? 'green' : m.profit_factor >= 1 ? 'yellow' : 'red'}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Equity curve — wider */}
        <div className="lg:col-span-2 bg-card p-5">
          <p className="text-sm font-semibold text-white mb-4">Equity Curve</p>
          <div className="h-56">
            <EquityChart data={m.equity_curve ?? []} />
          </div>
        </div>

        {/* Win/Loss donut */}
        <div className="bg-card p-5">
          <p className="text-sm font-semibold text-white mb-4">Win / Loss</p>
          <div className="h-56">
            <WinLossChart wins={m.winning_trades ?? 0} losses={m.losing_trades ?? 0} />
          </div>
        </div>
      </div>

      {/* Open trades summary */}
      <div className="bg-card p-5">
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm font-semibold text-white">Open Positions</p>
          <Badge label={`${openData?.trades?.length ?? 0} active`} variant="open" />
        </div>

        {oLoading ? (
          <LoadingSpinner size="sm" />
        ) : !openData?.trades?.length ? (
          <p className="text-sm text-slate-500 py-4 text-center">No open positions</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700">
                  {['Symbol', 'Direction', 'Entry', 'SL', 'TP', 'Size (USD)', 'Split Entry'].map(h => (
                    <th key={h} className="pb-2 pr-4 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {openData.trades.map(t => (
                  <tr key={t.id} className="hover:bg-slate-700/30 transition-colors">
                    <td className="py-2.5 pr-4 font-medium text-white">{t.symbol}</td>
                    <td className="py-2.5 pr-4"><Badge label={t.direction} /></td>
                    <td className="py-2.5 pr-4 font-mono">${t.entry_price?.toFixed(2)}</td>
                    <td className="py-2.5 pr-4 font-mono text-red-400">${t.stop_loss?.toFixed(2)}</td>
                    <td className="py-2.5 pr-4 font-mono text-emerald-400">${t.take_profit?.toFixed(2)}</td>
                    <td className="py-2.5 pr-4 font-mono">${t.quantity?.toFixed(2)}</td>
                    <td className="py-2.5 pr-4">
                      {t.is_split_entry
                        ? <Badge label={t.split_entry_filled ? 'Filled' : 'Pending'} variant={t.split_entry_filled ? 'filled' : 'pending'} />
                        : <span className="text-slate-600">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Extra stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Avg Win" value={m.avg_win != null ? `$${m.avg_win.toFixed(2)}` : '—'} accent="green" icon={TrendingUp} />
        <MetricCard title="Avg Loss" value={m.avg_loss != null ? `$${Math.abs(m.avg_loss).toFixed(2)}` : '—'} accent="red" icon={TrendingDown} />
        <MetricCard title="Max Drawdown" value={m.max_drawdown != null ? `${m.max_drawdown.toFixed(1)}%` : '—'} accent="yellow" icon={BarChart3} />
        <MetricCard title="Closed Trades" value={m.closed_trades ?? 0} accent="purple" icon={Activity} />
      </div>
    </div>
  )
}
