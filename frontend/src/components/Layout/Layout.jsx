import { Outlet } from 'react-router-dom'
import { useState, useCallback } from 'react'
import Sidebar from './Sidebar'
import Navbar from './Navbar'

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)

  const handleRefresh = useCallback(() => {
    // Broadcast a custom event that pages can listen to
    window.dispatchEvent(new CustomEvent('dashboard-refresh'))
    setLastUpdated(new Date().toLocaleTimeString())
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-slate-900">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Navbar
          onMenuClick={() => setSidebarOpen(true)}
          lastUpdated={lastUpdated}
          onRefresh={handleRefresh}
        />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
