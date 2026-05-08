import { InboxIcon } from 'lucide-react'

export default function EmptyState({ title = 'No data', message, icon: Icon = InboxIcon }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
      <div className="flex items-center justify-center w-12 h-12 rounded-full bg-slate-800 text-slate-500">
        <Icon size={22} />
      </div>
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {message && <p className="text-xs text-slate-500 max-w-xs">{message}</p>}
    </div>
  )
}
