import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Polls `fetchFn` every `intervalMs` milliseconds.
 * Returns { data, loading, error, refetch }.
 */
export function usePolling(fetchFn, intervalMs = 10000, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)

  const run = useCallback(async () => {
    try {
      const result = await fetchFn()
      setData(result)
      setError(null)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setLoading(true)
    run()
    timerRef.current = setInterval(run, intervalMs)
    return () => clearInterval(timerRef.current)
  }, [run, intervalMs])

  return { data, loading, error, refetch: run }
}
