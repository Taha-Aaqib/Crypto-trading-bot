import { useState, useEffect, useCallback } from 'react'
import { Search, ChevronDown, RefreshCw } from 'lucide-react'
import { fetchLogs } from '../services/api'
import LoadingSpinner from '../components/common/LoadingSpinner'
import EmptyState from '../components/common/EmptyState'
import Pagination from '../components/common/Pagination'
import Badge from '../components/common/Badge'

const LEVELS = ['All', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
const PAGE_SIZE = 100

function LogRow({ entry }) {
  const [expanded, setExpanded] = useState(false)
  const long = entry.message.length > 120

  return (
    <tr
      className="hover:bg-slate-700/30 transition-colors cursor-pointer"
      onClick={() => long && setExpanded(e => !e)}
    >
      <td className="px-4 py-2.5 font-mono text-slate-400 whitespace-nowrap text-xs">
        {entry.timestamp}
      </td>
      <td className="px-4 py-2.5">
        <Badge label={entry.level} variant={entry.level} />
      </td>
      <td className="px-4 py-2.5 text-slate-500 text-xs font-mono">{entry.logger}</td>
      <td className="px-4 py-2.5 text-xs text-slate-200 max-w-xl">
        <span className={`block ${!expanded && long ? 'truncate' : ''}`}>
          {entry.message}
        </span>
        {long && (
          <ChevronDown
            size={12}
            className={`mt-0.5 text-slate-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          />
        )}
      </td>
    </tr>
  )
}

export default function Logs() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [level,   setLevel]   = useState('All')
  const [search,  setSearch]  = useState('')
  const [dSearch, setDSearch] = useState('')
  const [page,    setPage]    = useState(1)
  const [auto,    setAuto]    = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = { page, limit: PAGE_SIZE }
      if (level !== 'All') params.level = level
      if (dSearch) params.search = dSearch
      const res = await fetchLogs(params)
      setData(res)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [page, level, dSearch])

  useEffect(() => { load() }, [load])

  // Auto-refresh
  useEffect(() => {
    if (!auto) return
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [auto, load])

  useEffect(() => {
    const h = () => load()
    window.addEventListener('dashboard-refresh', h)
    return () => window.removeEventListener('dashboard-refresh', h)
  }, [load])

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => { setDSearch(search); setPage(1) }, 400)
    return () => clearTimeout(t)
  }, [search])

  const logs  = data?.logs ?? []

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-48 max-w-sm">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search log messages…"
            className="w-full pl-8 pr-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
          />
        </div>

        <div className="flex gap-1.5 flex-wrap">
          {LEVELS.map(l => (
            <button
              key={l}
              onClick={() => { setLevel(l); setPage(1) }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border
                ${level === l
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800'}`}
            >
              {l}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setAuto(a => !a)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors
              ${auto ? 'bg-emerald-600/20 border-emerald-600/40 text-emerald-400' : 'border-slate-700 text-slate-400 hover:bg-slate-800'}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${auto ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
            Auto-refresh
          </button>
          <button
            onClick={load}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* Stats strip */}
      {data && (
        <div className="text-xs text-slate-500">
          Showing <span className="text-slate-300">{logs.length}</span> of{' '}
          <span className="text-slate-300">{data.total}</span> entries
          {level !== 'All' && <> · filtered to <Badge label={level} variant={level} /></>}
        </div>
      )}

      {/* Log table */}
      <div className="bg-card overflow-hidden">
        {loading ? <LoadingSpinner label="Reading log files…" /> :
         !logs.length ? <EmptyState title="No log entries" message="No logs match the current filters." /> : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-800/60">
                  <tr className="text-slate-400 text-xs">
                    <th className="px-4 py-3 text-left font-medium whitespace-nowrap">Timestamp</th>
                    <th className="px-4 py-3 text-left font-medium">Level</th>
                    <th className="px-4 py-3 text-left font-medium">Logger</th>
                    <th className="px-4 py-3 text-left font-medium">Message</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {logs.map((entry, i) => <LogRow key={i} entry={entry} />)}
                </tbody>
              </table>
            </div>
            <div className="px-4 py-3 border-t border-slate-800">
              <Pagination page={page} pages={data.pages} onPage={setPage} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
