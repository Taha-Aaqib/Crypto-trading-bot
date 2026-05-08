import { Menu, RefreshCw } from 'lucide-react'
import { useLocation } from 'react-router-dom'

const PAGE_TITLES = {
  '/':            'Dashboard',
  '/open-trades': 'Open Trades',
  '/trades':      'Trade History',
  '/signals':     'Signal Feed',
  '/performance': 'Performance',
  '/logs':        'System Logs',
}

export default function Navbar({ onMenuClick, lastUpdated, onRefresh }) {
  const { pathname } = useLocation()
  const title = PAGE_TITLES[pathname] ?? 'Dashboard'

  return (
    <header className="flex items-center justify-between px-4 h-14 border-b border-slate-800 bg-slate-900 flex-shrink-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <Menu size={18} />
        </button>
        <h1 className="text-base font-semibold text-white">{title}</h1>
      </div>

      <div className="flex items-center gap-4">
        {lastUpdated && (
          <span className="hidden sm:block text-xs text-slate-500">
            Updated {lastUpdated}
          </span>
        )}
        <button
          onClick={onRefresh}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400
            hover:text-white hover:bg-slate-800 border border-slate-700 transition-colors"
        >
          <RefreshCw size={13} />
          Refresh
        </button>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-slate-400">Live</span>
        </div>
      </div>
    </header>
  )
}
