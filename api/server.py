"""
FastAPI REST API server for the React dashboard.

Reads from the same SQLite database as the trading bot WITHOUT importing
or modifying any existing backend code.

Run from the Crypto-trading-bot/ directory:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import glob
import math
import os
import re
import yaml

from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ── Paths ─────────────────────────────────────────────────────────────────────
_API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_API_DIR)  # Crypto-trading-bot/


def _load_config() -> dict:
    cfg_path = os.path.join(PROJECT_DIR, "config", "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


cfg = _load_config()

_db_rel = cfg.get("database", {}).get("path", "data/trading_bot.db")
DB_PATH = os.path.join(PROJECT_DIR, _db_rel) if not os.path.isabs(_db_rel) else _db_rel

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── ORM models (mirror of existing schema — read-only) ────────────────────────
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    side = Column(String)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    status = Column(String)
    strategy = Column(String)
    timeframe = Column(String)
    notes = Column(Text)
    partial_tp_taken = Column(Boolean)
    is_split_entry = Column(Boolean)
    split_entry_filled = Column(Boolean)
    avg_entry_price = Column(Float)


class PendingLimitOrder(Base):
    __tablename__ = "pending_limit_orders"
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer)
    symbol = Column(String)
    side = Column(String)
    limit_price = Column(Float)
    quantity = Column(Float)
    created_time = Column(DateTime)
    expiry_time = Column(DateTime)
    status = Column(String)
    fill_price = Column(Float)
    fill_time = Column(DateTime)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    signal_type = Column(String)
    direction = Column(String)
    strength = Column(Float)
    timestamp = Column(DateTime)
    details = Column(Text)
    executed = Column(Boolean)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    portfolio_value = Column(Float)


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="SMC Trading Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Serialisers ────────────────────────────────────────────────────────────────
def _trade_dict(t: Trade) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "side": t.side,
        "direction": "long" if t.side == "buy" else "short",
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "stop_loss": t.stop_loss,
        "take_profit": t.take_profit,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "pnl": t.pnl,
        "pnl_percentage": t.pnl_percentage,
        "status": t.status,
        "strategy": t.strategy,
        "timeframe": t.timeframe,
        "notes": t.notes,
        "partial_tp_taken": bool(t.partial_tp_taken),
        "is_split_entry": bool(t.is_split_entry),
        "split_entry_filled": bool(t.split_entry_filled),
        "avg_entry_price": t.avg_entry_price,
    }


def _order_dict(o: PendingLimitOrder) -> dict:
    return {
        "id": o.id,
        "trade_id": o.trade_id,
        "symbol": o.symbol,
        "side": o.side,
        "limit_price": o.limit_price,
        "quantity": o.quantity,
        "status": o.status,
        "fill_price": o.fill_price,
        "created_time": o.created_time.isoformat() if o.created_time else None,
        "expiry_time": o.expiry_time.isoformat() if o.expiry_time else None,
        "fill_time": o.fill_time.isoformat() if o.fill_time else None,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db_exists": os.path.exists(DB_PATH),
        "db_path": DB_PATH,
    }


@app.get("/api/trades")
def list_trades(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Trade)
    if status:
        q = q.filter(Trade.status == status)
    if symbol and symbol.lower() != "all":
        q = q.filter(Trade.symbol == symbol)
    total = q.count()
    trades = q.order_by(Trade.entry_time.desc()).offset(offset).limit(limit).all()
    return {"total": total, "trades": [_trade_dict(t) for t in trades]}


@app.get("/api/trades/open")
def open_trades(db: Session = Depends(get_db)):
    trades = db.query(Trade).filter(Trade.status == "open").all()
    return {"trades": [_trade_dict(t) for t in trades]}


@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db)):
    all_trades = db.query(Trade).all()
    closed = [t for t in all_trades if t.status == "closed"]
    open_t = [t for t in all_trades if t.status == "open"]

    n_closed = len(closed)
    wins = [t for t in closed if t.pnl and t.pnl > 0]
    losses = [t for t in closed if t.pnl and t.pnl <= 0]
    win_rate = round(len(wins) / n_closed * 100, 2) if n_closed else 0
    total_pnl = round(sum(t.pnl for t in closed if t.pnl), 2)
    avg_win = round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0
    profit_factor = round(abs(avg_win / avg_loss), 2) if avg_loss else 0

    # Max drawdown from equity curve
    sorted_c = sorted(closed, key=lambda t: t.exit_time or datetime.min)
    running, peak, max_dd = 0.0, 0.0, 0.0
    equity: list[dict] = []
    for t in sorted_c:
        if t.pnl and t.exit_time:
            running += t.pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            equity.append({"time": t.exit_time.isoformat(), "pnl": round(running, 2)})

    # Daily PnL breakdown (last 30 days)
    daily: dict[str, float] = {}
    for t in sorted_c:
        if t.pnl and t.exit_time:
            day = t.exit_time.strftime("%Y-%m-%d")
            daily[day] = round(daily.get(day, 0) + t.pnl, 2)
    daily_pnl = [{"date": k, "pnl": v} for k, v in sorted(daily.items())[-30:]]

    return {
        "total_trades": len(all_trades),
        "closed_trades": n_closed,
        "open_trades": len(open_t),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": round(max_dd * 100, 2),
        "equity_curve": equity,
        "daily_pnl": daily_pnl,
    }


@app.get("/api/signals")
def list_signals(
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    total = db.query(Signal).count()
    sigs = (
        db.query(Signal)
        .order_by(Signal.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "signals": [
            {
                "id": s.id,
                "symbol": s.symbol,
                "signal_type": s.signal_type,
                "direction": s.direction,
                "strength": s.strength,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "details": s.details,
                "executed": bool(s.executed),
            }
            for s in sigs
        ],
    }


@app.get("/api/pending-orders")
def pending_orders(db: Session = Depends(get_db)):
    orders = (
        db.query(PendingLimitOrder)
        .order_by(PendingLimitOrder.created_time.desc())
        .all()
    )
    return {"orders": [_order_dict(o) for o in orders]}


@app.get("/api/logs")
def get_logs(
    level: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    limit: int = Query(100, le=500),
):
    log_dir = os.path.join(
        PROJECT_DIR, cfg.get("logging", {}).get("log_dir", "logs")
    )
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\S+) - (\w+) - (.+)$"
    )

    entries: list[dict] = []
    for lf in sorted(glob.glob(os.path.join(log_dir, "*.log")), reverse=True)[:5]:
        try:
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = pattern.match(line.strip())
                    if not m:
                        continue
                    entry_level = m.group(3)
                    msg = m.group(4)
                    if level and entry_level != level.upper():
                        continue
                    if search and search.lower() not in msg.lower():
                        continue
                    entries.append(
                        {
                            "timestamp": m.group(1),
                            "logger": m.group(2),
                            "level": entry_level,
                            "message": msg,
                        }
                    )
        except Exception:
            pass

    entries.sort(key=lambda x: x["timestamp"], reverse=True)
    total = len(entries)
    off = (page - 1) * limit
    return {
        "total": total,
        "page": page,
        "pages": max(1, math.ceil(total / limit)),
        "logs": entries[off : off + limit],
    }


@app.get("/api/config")
def get_config():
    safe_keys = ("trading", "strategy", "risk", "ensemble", "ml_model", "sentiment", "timeframes")
    return {k: cfg[k] for k in safe_keys if k in cfg}
