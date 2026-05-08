# AI-Enhanced SMC Crypto Trading Bot

**Final Year Project — University of Management and Technology**

> An automated crypto trading bot combining Smart Money Concepts (SMC), multi-timeframe analysis, PyTorch LSTM models, and sentiment analysis — with a React dashboard and FastAPI backend.

**Team:** Muhammad Taha Aaqib · Omer Farooq · Asadullah Bin Hassan  
**Advisor:** Muhammad Asim Butt (asim.butt@umt.edu.pk)

---

## Features

- **Multi-Timeframe SMC Analysis** — 1D bias, 4H structure, 15M entry (CHOCH, BOS, FVG, Liquidity Zones)
- **ML Ensemble** — PyTorch LSTM + MTF Pattern Recognition model per trading pair
- **Sentiment Filtering** — FinBERT on RSS feeds + CoinGecko market metrics + Fear & Greed Index
- **Risk Management** — ATR-based SL/TP, partial TP, trailing stop, split entry, circuit breaker
- **Paper & Live Trading** — via Binance Futures (CCXT)
- **React Dashboard** — real-time trades, signals, P&L, equity curve, logs viewer
- **Backtesting** — fast vectorized backtester with walk-forward testing

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Binance account (Futures enabled) — for paper or live trading

---

## Setup (Step by Step)

### 1. Clone the repo

```bash
git clone https://github.com/Taha-Aaqib/Crypto-trading-bot.git
cd Crypto-trading-bot
```

### 2. Create Python virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **TA-Lib** requires a C library. If install fails:
> - **Windows:** Download wheel from https://github.com/cgohlke/talib-binary/releases
> - **Linux:** `sudo apt-get install libta-lib-dev`
> - **macOS:** `brew install ta-lib`

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Configure environment variables

Copy the example file and fill in your API keys:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` and add your keys:

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
COINGECKO_API_KEY=your_coingecko_key      # Free at coingecko.com
REDDIT_CLIENT_ID=your_reddit_id           # Optional — for sentiment
REDDIT_CLIENT_SECRET=your_reddit_secret   # Optional — for sentiment
REDDIT_USER_AGENT=crypto_bot/1.0
```

> **Get API keys:**
> - Binance: https://www.binance.com/en/my/settings/api-management
> - CoinGecko (free): https://www.coingecko.com/en/developers/dashboard
> - Reddit (optional): https://www.reddit.com/prefs/apps

### 6. Train the ML models

```bash
python train_lstm_model.py
python train_mtf_models.py
```

> Models are saved to `models/saved_models/`. This takes a few minutes — it downloads historical data automatically.

---

## Running the Bot

### Option A — One click (Windows)

```bash
# Start bot
python main.py

# Start dashboard (separate terminal or double-click)
.\start_dashboard.bat
```

### Option B — Manual (all platforms)

**Terminal 1 — Trading bot:**
```bash
python main.py
```

**Terminal 2 — FastAPI backend:**
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — React dashboard:**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

> **Alternative:** Run the old Streamlit dashboard with `streamlit run dashboard.py` (no API server needed)

---

## Backtesting

```bash
# Default: BTC/USDT last 3 months
python run_fast_backtest.py

# Custom symbol and period
python run_fast_backtest.py --symbol ETH/USDT --months 6

# Full logic backtest
python run_full_logic_backtest.py

# Walk-forward test
python run_walk_forward_test.py
```

---

## Project Structure

```
Crypto-trading-bot/
├── main.py                   # Bot entry point
├── config/
│   └── config.yaml           # All settings (no secrets here)
├── src/
│   ├── data/                 # Data fetching & preprocessing
│   ├── indicators/           # SMC detector, TA indicators
│   ├── models/               # LSTM, MTF pattern, ensemble
│   ├── trading/              # Strategy, risk manager, order executor
│   ├── sentiment/            # FinBERT + RSS + CoinGecko + Fear&Greed
│   ├── backtesting/          # Backtesting engine
│   └── utils/                # Logger, DB manager, helpers
├── api/
│   └── server.py             # FastAPI backend for dashboard
├── frontend/                 # React + Vite + Tailwind dashboard
├── tests/                    # Unit tests
├── models/saved_models/      # Trained model weights (generated locally)
├── data/                     # SQLite database + cache
├── logs/                     # Bot logs
├── requirements.txt
└── start_dashboard.bat       # Windows: starts API + React in one click
```

---

## Configuration

All settings are in `config/config.yaml`. Key options:

```yaml
trading:
  mode: "paper"          # "paper" (safe) or "live" (real money)
  leverage: 10           # Futures leverage
  symbols: ["BTC/USDT", "ETH/USDT"]

strategy:
  confluence_threshold: 0.60   # Higher = fewer but better trades

risk:
  max_risk_per_trade: 0.015    # 1.5% of portfolio per trade
  take_profit_rr_ratio: 2.0    # 2:1 risk-reward
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Technologies

| Layer | Technology |
|---|---|
| Trading bot | Python, CCXT, TA-Lib |
| ML models | PyTorch (LSTM), scikit-learn |
| Sentiment | HuggingFace FinBERT, feedparser |
| Backend API | FastAPI, SQLAlchemy, SQLite |
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| Data | Binance API, CoinGecko, RSS feeds |

---

## ⚠️ Risk Disclaimer

This is an academic project. Cryptocurrency trading carries significant financial risk. Always run in **paper mode** first. Never trade with money you cannot afford to lose.

---

## License

Educational use only — Final Year Project, University of Management and Technology.
