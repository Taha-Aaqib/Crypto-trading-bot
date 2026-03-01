# AI-Enhanced Smart Money Concept-Based Crypto Trading Bot

**FYP - Fall 2025**  
**University of Management and Technology**

## Team Members

- Muhammad Taha Aaqib (F2022266369)
- Omer Farooq (F2022266366)
- Asadullah Bin Hassan (F2022266361)

**Advisor:** Muhammad Asim Butt (asim.butt@umt.edu.pk)

## Project Description

An advanced, modular trading system designed to operate in real-time cryptocurrency markets. The bot integrates Smart Money Concepts (SMC) such as CHOCH (Change of Character), BOS (Break of Structure), Liquidity Zones, and Fair Value Gaps (FVGs) to generate trade signals.

### Key Features

- **Multi-Timeframe SMC Analysis**

  - 1D: Directional bias (CHOCH, BOS, FVG)
  - 4H: Trend structure evaluation
  - 15M: Entry signals with liquidity analysis

- **EMA Trend Filter**

  - Uses EMA-50 and EMA-200 on 15-minute timeframe
  - Only executes trades when price structure and EMA align

- **Sentiment Analysis**

  - Real-time Twitter sentiment using NLP (VADER, TextBlob)
  - News sentiment integration
  - Trade filtering based on market sentiment

- **Risk Management**

  - Position sizing based on ATR
  - Stop loss and take profit automation
  - Daily loss limits and circuit breakers
  - Maximum open position limits

- **Paper and Live Trading Modes**

  - Test strategies without risk using paper mode
  - Seamless transition to live trading

- **Real-time Dashboard**
  - Built with Streamlit
  - Visualizes signals, performance, and system metrics

## Installation

### 1. Prerequisites

- Python 3.12+ (Already set up in virtual environment)
- Binance account (for live trading) or use testnet for testing

### 2. Virtual Environment (Already Created)

The virtual environment `venv` is already set up and packages are installed.

To activate:

```powershell

.\venv\Scripts\Activate.ps1
```

 cd "d:\FYP\Crypto-trading-bot";
 

### 3. Configuration

Edit `config/config.yaml` and add your API keys:

```yaml
exchange:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_API_SECRET"
  testnet: true # Set to false for live trading

twitter:
  bearer_token: "YOUR_TWITTER_BEARER_TOKEN"
  # ... other Twitter API credentials
```

## Usage

### Running the Bot

1. Activate virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

cd path; Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\venv\Scripts\Activate.ps1


2. Run the main bot:

```powershell
python main.py
```

The bot will:

- Fetch multi-timeframe data every 15 minutes
- Analyze SMC patterns, technical indicators, and sentiment
- Generate and execute trades based on strategy
- Manage open positions with stop loss and take profit
- Generate daily performance summaries

### Running the Dashboard

```powershell
streamlit run dashboard/app.py
```

Access the dashboard at `http://localhost:8501`

### Backtesting

```python
python -m src.backtesting.backtest_engine
```

## Project Structure

```
smart_trading_bot/
├── config/
│   └── config.yaml           # Configuration file
├── src/
│   ├── data/                 # Data fetching and preprocessing
│   ├── indicators/           # SMC detector and technical indicators
│   ├── sentiment/            # Sentiment analysis
│   ├── trading/              # Strategy, risk management, execution
│   ├── models/               # ML models (future)
│   ├── backtesting/          # Backtesting engine
│   └── utils/                # Utilities (logging, database, helpers)
├── dashboard/                # Streamlit dashboard
├── data/                     # Data storage
├── logs/                     # Log files
├── main.py                   # Main entry point
└── requirements.txt          # Python dependencies
```

## Key Modules

### 1. Smart Money Concept Detector (`src/indicators/smc_detector.py`)

- Detects CHOCH (Change of Character)
- Identifies BOS (Break of Structure)
- Finds Fair Value Gaps (FVG)
- Marks liquidity zones

### 2. Trading Strategy (`src/trading/strategy.py`)

- Multi-timeframe analysis integration
- Signal generation with all filters
- Entry and exit logic

### 3. Risk Manager (`src/trading/risk_manager.py`)

- Position sizing calculations
- Daily loss limits
- Circuit breaker implementation

### 4. Order Executor (`src/trading/order_executor.py`)

- Paper trading simulation
- Live order execution via CCXT
- Trade management

## Development Timeline

### FYP 1 (Fall Semester - Oct 2024 - Jan 2025)

- ✅ Literature review on SMC and sentiment analysis
- ✅ Binance API setup and testing
- ✅ SMC detection logic (CHOCH, BOS, FVG)
- ✅ Multi-timeframe analysis
- ✅ Twitter sentiment integration
- 🔄 Backtester development (in progress)
- 🔄 Initial dashboard UI

### FYP 2 (Spring Semester - Feb - Jun 2025)

- Real-time trading module integration
- Risk management engine refinement
- Live trading testing
- Dashboard completion
- Final report and presentation

## Testing

Run unit tests:

```powershell
pytest tests/
```

## Technologies Used

- **Python 3.12+** - Core language
- **CCXT** - Exchange connectivity
- **Pandas & NumPy** - Data processing
- **TextBlob, VADER, HuggingFace** - Sentiment analysis
- **Tweepy** - Twitter API integration
- **Streamlit** - Dashboard
- **SQLAlchemy** - Database ORM
- **TA-Lib** - Technical indicators
- **PyTorch & scikit-learn** - ML models (future)

## Trading Strategy Logic

1. **Analyze 1D timeframe** for overall market bias
2. **Check 4H timeframe** for trend structure confirmation
3. **Monitor 15M timeframe** for entry opportunities
4. **Validate with EMA filter** (price > EMA50 > EMA200 for long)
5. **Check sentiment** from news and market data
6. **Calculate position size** based on risk management
7. **Execute trade** with stop loss and take profit
8. **Monitor and manage** open positions

## Risk Disclaimer

This is an educational project for academic purposes. Cryptocurrency trading carries significant risk. Never trade with money you cannot afford to lose. Always test thoroughly in paper mode before considering live trading.

## License

This project is for educational purposes as part of Final Year Project at University of Management and Technology.

## Contact

For questions or collaboration:

- Muhammad Taha Aaqib - F2022266369
- Omer Farooq - F2022266366
- Asadullah Bin Hassan - F2022266361

**Advisor:** Muhammad Asim Butt (asim.butt@umt.edu.pk)

---

**Note:** Remember to never commit your API keys to version control. Always use `.env` files or keep them in `config.yaml` which should be in `.gitignore`.
