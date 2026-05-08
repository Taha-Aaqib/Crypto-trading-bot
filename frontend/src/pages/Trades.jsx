import { useState, useEffect, useCallback } from 'react'
import { Search, Filter, Download } from 'lucide-react'
import { fetchTrades } from '../services/api'
import Badge from '../components/common/Badge'
import LoadingSpinner from '../components/common/LoadingSpinner'
import EmptyState from '../components/common/EmptyState'
import Pagination from '../components/common/Pagination'

const PAGE_SIZE = 20

function fmt(v, prefix = '') {
  if (v == null) return '—'
  return `${prefix}${Number(v).toFixed(2)}`
}

export default function Trades() {
  const [data,   setData]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [page,   setPage]   = useState(1)
  const [symbol, setSymbol] = useState('All')
  const [status, setStatus] = useState('All')
  const [search, setSearch] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = { limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE }
      if (symbol !== 'All') params.symbol = symbol
      if (status !== 'All') params.status = status.toLowerCase()
      const res = await fetchTrades(params)
      setData(res)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [page, symbol, status])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    const handler = () => load()
    window.addEventListener('dashboard-refresh', handler)
    return () => window.removeEventListener('dashboard-refresh', handler)
  }, [load])

  const trades = data?.trades ?? []
  const filtered = search
    ? trades.filter(t => t.symbol.toLowerCase().includes(search.toLowerCase()))
    : trades
  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1

  const exportCSV = () => {
    if (!filtered.length) return
    const cols = ['id','symbol','side','entry_price','exit_price','quantity','pnl','pnl_percentage','status','entry_time','exit_time']
    const rows = [cols.join(','), ...filtered.map(t => cols.map(c => t[c] ?? '').join(','))]
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `trades_${new Date().toISOString().slice(0,10)}.csv`; a.click()
  }

  return (
    <div className="space-y-4">
      {/* Filters bar */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-40 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Filter symbol…"
            className="w-full pl-8 pr-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
          />
        </div>

        <select
          value={symbol} onChange={e => { setSymbol(e.target.value); setPage(1) }}
          className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-blue-500"
        >
          {['All','BTC/USDT','ETH/USDT'].map(s => <option key={s}>{s}</option>)}
        </select>

        <select
          value={status} onChange={e => { setStatus(e.target.value); setPage(1) }}
          className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-blue-500"
        >
          {['All','Open','Closed'].map(s => <option key={s}>{s}</option>)}
        </select>

        <button
          onClick={exportCSV}
          className="ml-auto flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <Download size={13} /> Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="bg-card overflow-hidden">
        {loading ? (
          <LoadingSpinner label="Loading trades…" />
        ) : !filtered.length ? (
          <EmptyState title="No trades found" message="Try adjusting your filters." />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-slate-800/60">
                  <tr className="text-slate-400">
                    {['ID','Symbol','Side','Entry','Exit','Size','P&L','P&L %','Status','Entry Time','Exit Time'].map(h => (
                      <th key={h} className="px-4 py-3 text-left font-medium whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {filtered.map(t => {
                    const pnlPos = t.pnl != null ? t.pnl >= 0 : null
                    return (
                      <tr key={t.id} className="hover:bg-slate-700/30 transition-colors">
                        <td className="px-4 py-3 font-mono text-slate-500">#{t.id}</td>
                        <td className="px-4 py-3 font-medium text-white">{t.symbol}</td>
                        <td className="px-4 py-3"><Badge label={t.direction} /></td>
                        <td className="px-4 py-3 font-mono">${fmt(t.entry_price)}</td>
                        <td className="px-4 py-3 font-mono">{t.exit_price ? `$${fmt(t.exit_price)}` : '—'}</td>
                        <td className="px-4 py-3 font-mono">${fmt(t.quantity)}</td>
                        <td className={`px-4 py-3 font-mono font-semibold ${pnlPos === true ? 'text-emerald-400' : pnlPos === false ? 'text-red-400' : 'text-slate-500'}`}>
                          {t.pnl != null ? `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}` : '—'}
                        </td>
                        <td className={`px-4 py-3 font-mono ${pnlPos === true ? 'text-emerald-400' : pnlPos === false ? 'text-red-400' : 'text-slate-500'}`}>
                          {t.pnl_percentage != null ? `${t.pnl_percentage >= 0 ? '+' : ''}${t.pnl_percentage.toFixed(2)}%` : '—'}
                        </td>
                        <td className="px-4 py-3"><Badge label={t.status} /></td>
                        <td className="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
                          {t.entry_time ? new Date(t.entry_time).toLocaleString() : '—'}
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
                          {t.exit_time ? new Date(t.exit_time).toLocaleString() : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="px-4 py-3 border-t border-slate-800 flex items-center justify-between">
              <p className="text-xs text-slate-500">{data?.total ?? 0} total trades</p>
              <Pagination page={page} pages={totalPages} onPage={setPage} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
