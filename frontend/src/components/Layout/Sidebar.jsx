import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  TrendingUp,
  Activity,
  Zap,
  ScrollText,
  BarChart3,
  Bot,
} from 'lucide-react'

const NAV = [
  { to: '/',            label: 'Dashboard',    icon: LayoutDashboard },
  { to: '/open-trades', label: 'Open Trades',  icon: Activity },
  { to: '/trades',      label: 'Trade History',icon: TrendingUp },
  { to: '/signals',     label: 'Signals',      icon: Zap },
  { to: '/performance', label: 'Performance',  icon: BarChart3 },
  { to: '/logs',        label: 'Logs',         icon: ScrollText },
]

export default function Sidebar({ open, onClose }) {
  return (
    <>
      {/* Backdrop (mobile) */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/50 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-30 flex flex-col w-64 bg-slate-900 border-r border-slate-800
          transform transition-transform duration-200
          ${open ? 'translate-x-0' : '-translate-x-full'}
          lg:relative lg:translate-x-0 lg:flex`}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-blue-600">
            <Bot size={20} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white leading-tight">SMC Trading Bot</p>
            <p className="text-xs text-slate-400">AI-Enhanced</p>
          </div>
        </div>

        {/* Nav links */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                ${isActive
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>


      </aside>
    </>
  )
}
