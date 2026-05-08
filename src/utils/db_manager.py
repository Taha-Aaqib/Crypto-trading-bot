"""
Database manager for storing trades, logs, and analytics
Supports SQLite and PostgreSQL
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, inspect as sa_inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
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
    original_sl = Column(Float, nullable=True)  # Original SL before breakeven move
    partial_tp_taken = Column(Boolean, default=False)  # True after 30% closed at 1:1 R:R
    partial_tp_price = Column(Float, nullable=True)  # Price at which partial TP was executed
    partial_pnl = Column(Float, default=0.0)  # Accumulated PnL from partial closes
    # Split entry tracking
    is_split_entry = Column(Boolean, default=False)  # True if this trade uses split entry
    split_entry_filled = Column(Boolean, default=False)  # True when limit order portion filled
    avg_entry_price = Column(Float, nullable=True)  # Weighted average entry after both fills

    @property
    def direction(self):
        """Convert side (buy/sell) to direction (long/short) for strategy compatibility"""
        return 'long' if self.side == 'buy' else 'short'


class PendingLimitOrder(Base):
    """Pending limit order for split entry DCA strategy"""
    __tablename__ = 'pending_limit_orders'

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer)  # Link to parent trade
    symbol = Column(String(20))
    side = Column(String(10))  # buy or sell
    limit_price = Column(Float)  # Price at which to fill
    quantity = Column(Float)  # USD amount for this portion
    created_time = Column(DateTime)
    expiry_time = Column(DateTime)  # Auto-cancel after this time
    status = Column(String(20))  # pending, filled, cancelled, expired
    fill_price = Column(Float, nullable=True)  # Actual fill price
    fill_time = Column(DateTime, nullable=True)


class Signal(Base):
    """Signal model for database"""
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    signal_type = Column(String(20))  # smc, sentiment, ml
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

        # Run lightweight migrations (add new columns to existing tables)
        self._run_migrations()

        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

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

    def _run_migrations(self):
        """
        Lightweight schema migration: adds any new columns defined on models
        but missing from the actual database tables.
        This avoids needing Alembic for simple column additions.
        """
        from sqlalchemy import text
        inspector = sa_inspect(self.engine)

        for table_name, model_class in [('trades', Trade), ('signals', Signal), ('performance_metrics', PerformanceMetric), ('pending_limit_orders', PendingLimitOrder)]:
            if not inspector.has_table(table_name):
                continue

            existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
            model_columns = {col.name: col for col in model_class.__table__.columns}

            for col_name, col_obj in model_columns.items():
                if col_name not in existing_columns:
                    # Build ALTER TABLE statement
                    col_type = col_obj.type.compile(dialect=self.engine.dialect)
                    default_clause = ""
                    if col_obj.default is not None:
                        default_val = col_obj.default.arg
                        if isinstance(default_val, bool):
                            default_clause = f" DEFAULT {1 if default_val else 0}"
                        elif isinstance(default_val, (int, float)):
                            default_clause = f" DEFAULT {default_val}"
                        else:
                            default_clause = f" DEFAULT '{default_val}'"
                    elif col_obj.nullable:
                        default_clause = " DEFAULT NULL"

                    sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}{default_clause}"
                    try:
                        with self.engine.begin() as conn:
                            conn.execute(text(sql))
                        logger.info(f"Migration: added column '{col_name}' to '{table_name}'")
                    except Exception as e:
                        # Column might already exist or other benign issue
                        if 'duplicate' not in str(e).lower():
                            logger.debug(f"Migration note for {table_name}.{col_name}: {e}")

    def save_trade(self, trade_data: Dict) -> int:
        """Save a new trade to database"""
        session = self.Session()
        trade = Trade(
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            entry_price=trade_data['entry_price'],
            quantity=trade_data['quantity'],
            stop_loss=trade_data['stop_loss'],
            take_profit=trade_data['take_profit'],
            original_sl=trade_data['stop_loss'],  # Preserve original SL before any breakeven moves
            entry_time=trade_data.get('entry_time', datetime.now()),
            status='open',
            strategy=trade_data.get('strategy', 'SMC'),
            timeframe=trade_data.get('timeframe', '15m'),
            notes=trade_data.get('notes', '')
        )

        session.add(trade)
        session.commit()

        logger.info(
            f"Saved trade: {trade_data['symbol']} {trade_data['side']}")
        return trade.id

    def update_trade(self, trade_id: int, update_data: Dict):
        """Update existing trade"""
        session = self.Session()
        trade = session.query(Trade).filter_by(id=trade_id).first()

        if trade:
            for key, value in update_data.items():
                setattr(trade, key, value)

            session.commit()
            logger.info(f"Updated trade {trade_id}")
        else:
            logger.warning(f"Trade {trade_id} not found")

    def close_trade(self, trade_id: int, exit_price: float, exit_time: datetime = None):
        """Close a trade and calculate PnL (leverage-aware for futures, includes commission)"""
        session = self.Session()
        trade = session.query(Trade).filter_by(id=trade_id).first()

        if trade:
            trade.exit_price = exit_price
            trade.exit_time = exit_time or datetime.now()
            trade.status = 'closed'

            # Check market type, leverage, and commission from config
            config = self.config
            market_type = config.get('trading', {}).get('market_type', 'spot')
            leverage = config.get('trading', {}).get('leverage', 1)
            commission_rate = config.get('backtest', {}).get('commission', 0.001)  # 0.1% per side

            # trade.quantity = USD margin amount (e.g. $100)
            # For futures: actual position size in base = quantity * leverage / entry_price
            # PnL = price_diff * position_size_base (actual USD gained/lost)
            # For spot: position size in base = quantity / entry_price, no leverage
            effective_leverage = leverage if market_type == 'futures' else 1
            position_size_base = (trade.quantity * effective_leverage) / trade.entry_price

            if trade.side == 'buy':
                raw_pnl = (exit_price - trade.entry_price) * position_size_base
            else:  # sell/short
                raw_pnl = (trade.entry_price - exit_price) * position_size_base

            # Deduct trading commission (entry + exit)
            entry_commission = trade.entry_price * position_size_base * commission_rate
            exit_commission = exit_price * position_size_base * commission_rate
            trade.pnl = raw_pnl - entry_commission - exit_commission + (trade.partial_pnl or 0.0)

            # PnL % relative to the original margin (including any partial-closed portion)
            original_qty = trade.quantity  # current remaining margin
            if trade.partial_tp_taken:
                # Reconstruct original margin: remaining is (1 - close_fraction) of original
                close_fraction = self.config.get('risk', {}).get('partial_tp', {}).get('close_fraction', 0.30)
                original_qty = trade.quantity / (1 - close_fraction)
            trade.pnl_percentage = (trade.pnl / original_qty) * 100

            session.commit()
            logger.info(f"Closed trade {trade_id} with PnL: ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)")
        else:
            logger.warning(f"Trade {trade_id} not found")

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        session = self.Session()
        return session.query(Trade).filter_by(status='open').all()

    def get_trade(self, trade_id: int) -> Optional[Trade]:
        """Get a specific trade by ID"""
        session = self.Session()
        return session.query(Trade).filter_by(id=trade_id).first()

    def get_trade_history(self, symbol: str = None, limit: int = 100) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        session = self.Session()
        query = session.query(Trade)

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
        session = self.Session()
        signal = Signal(
            symbol=signal_data['symbol'],
            signal_type=signal_data['signal_type'],
            direction=signal_data['direction'],
            strength=signal_data['strength'],
            timestamp=signal_data.get('timestamp', datetime.now()),
            details=signal_data.get('details', ''),
            executed=signal_data.get('executed', False)
        )

        session.add(signal)
        session.commit()

    def save_performance_metrics(self, metrics: Dict):
        """Save daily performance metrics"""
        session = self.Session()
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

        session.add(perf)
        session.commit()

    def close(self):
        """Close database connection"""
        self.Session.remove()
        logger.info("Database connection closed")

    # ============ PENDING LIMIT ORDER METHODS ============

    def save_pending_limit_order(self, order_data: Dict) -> int:
        """Save a pending limit order for split entry"""
        session = self.Session()
        order = PendingLimitOrder(
            trade_id=order_data['trade_id'],
            symbol=order_data['symbol'],
            side=order_data['side'],
            limit_price=order_data['limit_price'],
            quantity=order_data['quantity'],
            created_time=order_data.get('created_time', datetime.now()),
            expiry_time=order_data['expiry_time'],
            status='pending'
        )
        session.add(order)
        session.commit()
        logger.info(f"Saved pending limit order for trade {order_data['trade_id']} at ${order_data['limit_price']:.2f}")
        return order.id

    def get_pending_limit_orders(self, symbol: str = None) -> List[PendingLimitOrder]:
        """Get all pending limit orders, optionally filtered by symbol"""
        session = self.Session()
        query = session.query(PendingLimitOrder).filter_by(status='pending')
        if symbol:
            query = query.filter_by(symbol=symbol)
        return query.all()

    def get_pending_limit_order_for_trade(self, trade_id: int) -> Optional[PendingLimitOrder]:
        """Get pending limit order for a specific trade"""
        session = self.Session()
        return session.query(PendingLimitOrder).filter_by(trade_id=trade_id, status='pending').first()

    def fill_pending_limit_order(self, order_id: int, fill_price: float):
        """Mark a pending limit order as filled"""
        session = self.Session()
        order = session.query(PendingLimitOrder).filter_by(id=order_id).first()
        if order:
            order.status = 'filled'
            order.fill_price = fill_price
            order.fill_time = datetime.now()
            session.commit()
            logger.info(f"Filled limit order {order_id} at ${fill_price:.2f}")

    def cancel_pending_limit_order(self, order_id: int, reason: str = 'cancelled'):
        """Cancel a pending limit order"""
        session = self.Session()
        order = session.query(PendingLimitOrder).filter_by(id=order_id).first()
        if order:
            order.status = reason  # 'cancelled' or 'expired'
            session.commit()
            logger.info(f"Cancelled limit order {order_id} - reason: {reason}")

    def expire_old_limit_orders(self):
        """Mark expired limit orders as expired (called periodically)"""
        session = self.Session()
        now = datetime.now()
        expired = session.query(PendingLimitOrder).filter(
            PendingLimitOrder.status == 'pending',
            PendingLimitOrder.expiry_time <= now
        ).all()
        for order in expired:
            order.status = 'expired'
            logger.info(f"Limit order {order.id} for trade {order.trade_id} expired")
        session.commit()
        return len(expired)
