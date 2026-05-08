import { useEffect } from 'react'
import { Target, TrendingUp, TrendingDown, BarChart3, Percent, Activity } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import { fetchMetrics } from '../services/api'
import MetricCard from '../components/common/MetricCard'
import LoadingSpinner from '../components/common/LoadingSpinner'
import EquityChart from '../components/charts/EquityChart'
import WinLossChart from '../components/charts/WinLossChart'
import DailyPnLChart from '../components/charts/DailyPnLChart'

export default function Performance() {
  const { data: m, loading, refetch } = usePolling(fetchMetrics, 15000)

  useEffect(() => {
    window.addEventListener('dashboard-refresh', refetch)
    return () => window.removeEventListener('dashboard-refresh', refetch)
  }, [refetch])

  if (loading) return <LoadingSpinner label="Loading performance data…" />

  const metrics = m ?? {}

  return (
    <div className="space-y-6">
      {/* KPI strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <MetricCard title="Total Trades"    value={metrics.total_trades ?? '—'} icon={Activity}     accent="blue"   />
        <MetricCard title="Win Rate"        value={metrics.closed_trades ? `${metrics.win_rate}%` : '—'} icon={Percent}    accent={metrics.win_rate >= 50 ? 'green' : 'red'} />
        <MetricCard title="Total P&L"       value={metrics.total_pnl != null ? `$${metrics.total_pnl.toFixed(2)}` : '—'} icon={metrics.total_pnl >= 0 ? TrendingUp : TrendingDown} accent={metrics.total_pnl >= 0 ? 'green' : 'red'} />
        <MetricCard title="Profit Factor"   value={metrics.profit_factor ?? '—'} icon={Target}       accent={metrics.profit_factor >= 1.5 ? 'green' : metrics.profit_factor >= 1 ? 'yellow' : 'red'} />
        <MetricCard title="Max Drawdown"    value={metrics.max_drawdown != null ? `${metrics.max_drawdown.toFixed(1)}%` : '—'} icon={BarChart3} accent="yellow" />
        <MetricCard title="Avg Win / Loss"  value={metrics.avg_win != null ? `$${metrics.avg_win.toFixed(2)}` : '—'} sub={metrics.avg_loss != null ? `Avg loss $${Math.abs(metrics.avg_loss).toFixed(2)}` : undefined} icon={TrendingUp} accent="green" />
      </div>

      {/* Equity + Win/Loss */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 bg-card p-5">
          <p className="text-sm font-semibold text-white mb-4">Cumulative P&L (Equity Curve)</p>
          <div className="h-64">
            <EquityChart data={metrics.equity_curve ?? []} />
          </div>
        </div>
        <div className="bg-card p-5">
          <p className="text-sm font-semibold text-white mb-4">Trade Outcomes</p>
          <div className="h-64">
            <WinLossChart wins={metrics.winning_trades ?? 0} losses={metrics.losing_trades ?? 0} />
          </div>
        </div>
      </div>

      {/* Daily P&L bar chart */}
      <div className="bg-card p-5">
        <p className="text-sm font-semibold text-white mb-4">Daily P&L (Last 30 Days)</p>
        <div className="h-56">
          <DailyPnLChart data={metrics.daily_pnl ?? []} />
        </div>
      </div>

      {/* Detailed stats table */}
      <div className="bg-card p-5">
        <p className="text-sm font-semibold text-white mb-4">Detailed Statistics</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-3">
          {[
            ['Total Trades',    metrics.total_trades ?? '—'],
            ['Closed Trades',   metrics.closed_trades ?? '—'],
            ['Open Trades',     metrics.open_trades ?? '—'],
            ['Winning Trades',  metrics.winning_trades ?? '—'],
            ['Losing Trades',   metrics.losing_trades ?? '—'],
            ['Win Rate',        metrics.closed_trades ? `${metrics.win_rate}%` : '—'],
            ['Total P&L',       metrics.total_pnl != null ? `$${metrics.total_pnl.toFixed(2)}` : '—'],
            ['Avg Win',         metrics.avg_win != null ? `$${metrics.avg_win.toFixed(2)}` : '—'],
            ['Avg Loss',        metrics.avg_loss != null ? `$${metrics.avg_loss.toFixed(2)}` : '—'],
            ['Profit Factor',   metrics.profit_factor ?? '—'],
            ['Max Drawdown',    metrics.max_drawdown != null ? `${metrics.max_drawdown.toFixed(2)}%` : '—'],
          ].map(([label, value]) => (
            <div key={label} className="flex items-center justify-between py-2.5 border-b border-slate-800">
              <span className="text-xs text-slate-400">{label}</span>
              <span className="text-xs font-semibold text-white font-mono">{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
