import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout/Layout'
import Dashboard from './pages/Dashboard'
import Trades from './pages/Trades'
import OpenTrades from './pages/OpenTrades'
import Signals from './pages/Signals'
import Logs from './pages/Logs'
import Performance from './pages/Performance'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="trades" element={<Trades />} />
          <Route path="open-trades" element={<OpenTrades />} />
          <Route path="signals" element={<Signals />} />
          <Route path="logs" element={<Logs />} />
          <Route path="performance" element={<Performance />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
