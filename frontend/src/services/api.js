import axios from 'axios'

const BASE = '/api'

const api = axios.create({ baseURL: BASE, timeout: 10000 })

export const fetchHealth = () => api.get('/health').then(r => r.data)

export const fetchMetrics = () => api.get('/metrics').then(r => r.data)

export const fetchTrades = (params = {}) =>
  api.get('/trades', { params }).then(r => r.data)

export const fetchOpenTrades = () =>
  api.get('/trades/open').then(r => r.data)

export const fetchSignals = (params = {}) =>
  api.get('/signals', { params }).then(r => r.data)

export const fetchPendingOrders = () =>
  api.get('/pending-orders').then(r => r.data)

export const fetchLogs = (params = {}) =>
  api.get('/logs', { params }).then(r => r.data)

export const fetchConfig = () => api.get('/config').then(r => r.data)
