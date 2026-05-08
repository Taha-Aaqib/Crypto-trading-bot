import { ChevronLeft, ChevronRight } from 'lucide-react'

export default function Pagination({ page, pages, onPage }) {
  if (pages <= 1) return null
  const start = Math.max(1, page - 2)
  const end   = Math.min(pages, page + 2)
  const nums  = Array.from({ length: end - start + 1 }, (_, i) => start + i)

  const btn = (label, target, disabled) => (
    <button
      key={label}
      onClick={() => !disabled && onPage(target)}
      disabled={disabled}
      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
        ${disabled
          ? 'text-slate-600 cursor-not-allowed'
          : 'text-slate-400 hover:text-white hover:bg-slate-700'
        }`}
    >
      {label}
    </button>
  )

  return (
    <div className="flex items-center justify-center gap-1 mt-4">
      {btn(<ChevronLeft size={14} />, page - 1, page <= 1)}
      {start > 1  && <>{btn(1, 1, false)}<span className="text-slate-600 text-xs">…</span></>}
      {nums.map(n => (
        <button
          key={n}
          onClick={() => onPage(n)}
          className={`w-8 h-8 rounded-lg text-xs font-medium transition-colors
            ${n === page
              ? 'bg-blue-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-slate-700'
            }`}
        >
          {n}
        </button>
      ))}
      {end < pages && <><span className="text-slate-600 text-xs">…</span>{btn(pages, pages, false)}</>}
      {btn(<ChevronRight size={14} />, page + 1, page >= pages)}
    </div>
  )
}
