"""
Database manager for storing trades, logs, and analytics
Supports SQLite and PostgreSQL
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.utils.logger import get_logger

Base = declarative_base()
logger = get_logger()


class Trade(Base):
    """Trade model for database"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    side = Column(String(10))  # buy or sell
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_percentage = Column(Float, nullable=True)
    status = Column(String(20))  # open, closed, cancelled
    strategy = Column(String(50))
    timeframe = Column(String(10))
    notes = Column(Text, nullable=True)

    @property
    def direction(self):
        """Convert side (buy/sell) to direction (long/short) for strategy compatibility"""
        return 'long' if self.side == 'buy' else 'short'


class Signal(Base):
    """Signal model for database"""
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    signal_type = Column(String(20))  # smc, sentiment, event
    direction = Column(String(10))  # long, short, neutral
    strength = Column(Float)
    timestamp = Column(DateTime)
    details = Column(Text)
    executed = Column(Boolean, default=False)


class PerformanceMetric(Base):
    """Performance metrics model"""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    portfolio_value = Column(Float)


class DatabaseManager:
    """Manage database operations for the trading bot"""

    def __init__(self, config: Dict):
        self.config = config
        self.db_config = config['database']
        self.engine = self._create_engine()

        # Create tables
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        logger.info("Database initialized successfully")

    def _create_engine(self):
        """Create database engine based on configuration"""
        if self.db_config['type'] == 'sqlite':
            db_path = self.db_config['path']
            return create_engine(f'sqlite:///{db_path}')

        elif self.db_config['type'] == 'postgresql':
            user = self.db_config['user']
            password = self.db_config['password']
            host = self.db_config['host']
            port = self.db_config['port']
            database = self.db_config['database']
            return create_engine(
                f'postgresql://{user}:{password}@{host}:{port}/{database}'
            )

        else:
            raise ValueError(
                f"Unsupported database type: {self.db_config['type']}")

    def save_trade(self, trade_data: Dict) -> int:
        """Save a new trade to database"""
        trade = Trade(
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            entry_price=trade_data['entry_price'],
            quantity=trade_data['quantity'],
            stop_loss=trade_data['stop_loss'],
            take_profit=trade_data['take_profit'],
            entry_time=trade_data.get('entry_time', datetime.now()),
            status='open',
            strategy=trade_data.get('strategy', 'SMC'),
            timeframe=trade_data.get('timeframe', '15m'),
            notes=trade_data.get('notes', '')
        )

        self.session.add(trade)
        self.session.commit()

        logger.info(
            f"Saved trade: {trade_data['symbol']} {trade_data['side']}")
        return trade.id

    def update_trade(self, trade_id: int, update_data: Dict):
        """Update existing trade"""
        trade = self.session.query(Trade).filter_by(id=trade_id).first()

        if trade:
            for key, value in update_data.items():
                setattr(trade, key, value)

            self.session.commit()
            logger.info(f"Updated trade {trade_id}")
        else:
            logger.warning(f"Trade {trade_id} not found")

    def close_trade(self, trade_id: int, exit_price: float, exit_time: datetime = None):
        """Close a trade and calculate PnL"""
        trade = self.session.query(Trade).filter_by(id=trade_id).first()

        if trade:
            trade.exit_price = exit_price
            trade.exit_time = exit_time or datetime.now()
            trade.status = 'closed'

            # Calculate PnL
            if trade.side == 'buy':
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            else:  # sell/short
                trade.pnl = (trade.entry_price - exit_price) * trade.quantity

            trade.pnl_percentage = (
                trade.pnl / (trade.entry_price * trade.quantity)) * 100

            self.session.commit()
            logger.info(f"Closed trade {trade_id} with PnL: {trade.pnl:.2f}")
        else:
            logger.warning(f"Trade {trade_id} not found")

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        return self.session.query(Trade).filter_by(status='open').all()

    def get_trade(self, trade_id: int) -> Optional[Trade]:
        """Get a specific trade by ID"""
        return self.session.query(Trade).filter_by(id=trade_id).first()

    def get_trade_history(self, symbol: str = None, limit: int = 100) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        query = self.session.query(Trade)

        if symbol:
            query = query.filter_by(symbol=symbol)

        trades = query.order_by(Trade.entry_time.desc()).limit(limit).all()

        data = [{
            'id': t.id,
            'symbol': t.symbol,
            'side': t.side,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'pnl': t.pnl,
            'pnl_percentage': t.pnl_percentage,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'status': t.status
        } for t in trades]

        return pd.DataFrame(data)

    def save_signal(self, signal_data: Dict):
        """Save a trading signal"""
        signal = Signal(
            symbol=signal_data['symbol'],
            signal_type=signal_data['signal_type'],
            direction=signal_data['direction'],
            strength=signal_data['strength'],
            timestamp=signal_data.get('timestamp', datetime.now()),
            details=signal_data.get('details', ''),
            executed=signal_data.get('executed', False)
        )

        self.session.add(signal)
        self.session.commit()

    def save_performance_metrics(self, metrics: Dict):
        """Save daily performance metrics"""
        perf = PerformanceMetric(
            date=metrics.get('date', datetime.now()),
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            total_pnl=metrics['total_pnl'],
            sharpe_ratio=metrics.get('sharpe_ratio'),
            max_drawdown=metrics.get('max_drawdown'),
            portfolio_value=metrics['portfolio_value']
        )

        self.session.add(perf)
        self.session.commit()

    def close(self):
        """Close database connection"""
        self.session.close()
        logger.info("Database connection closed")
