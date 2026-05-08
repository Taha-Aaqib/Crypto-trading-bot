export default function LoadingSpinner({ size = 'md', label }) {
  const s = size === 'sm' ? 'w-4 h-4 border-2' : size === 'lg' ? 'w-10 h-10 border-3' : 'w-7 h-7 border-2'
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <div className={`${s} border-slate-700 border-t-blue-500 rounded-full animate-spin`} />
      {label && <p className="text-sm text-slate-500">{label}</p>}
    </div>
  )
}
