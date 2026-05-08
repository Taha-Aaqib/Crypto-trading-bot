"""
Trading Bot Dashboard
Real-time monitoring of trades, split entries, and performance

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

from src.utils.db_manager import DatabaseManager, Trade, PendingLimitOrder
from src.utils.helpers import load_config
from src.data.data_fetcher import DataFetcher

# Page config
st.set_page_config(
    page_title="Crypto Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config and initialize DB
@st.cache_resource
def init_components():
    config = load_config()
    db_manager = DatabaseManager(config)
    data_fetcher = DataFetcher(config)
    return config, db_manager, data_fetcher

config, db_manager, data_fetcher = init_components()

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 2, 30, 5)
    st.sidebar.info(f"Refreshing every {refresh_interval}s")

# Title
st.title("📈 AI-Enhanced SMC Trading Bot")
st.markdown(f"**Mode:** {config['trading']['mode'].upper()} | **Market:** {config['trading'].get('market_type', 'spot').upper()} | **Leverage:** {config['trading'].get('leverage', 1)}x")

# ============ LIVE PRICES ============
st.header("💰 Live Prices")
price_cols = st.columns(len(config['trading']['symbols']))

for i, symbol in enumerate(config['trading']['symbols']):
    with price_cols[i]:
        try:
            ticker = data_fetcher.fetch_ticker(symbol)
            if ticker:
                price = ticker.get('last', 0)
                change_24h = ticker.get('percentage', 0)
                color = "green" if change_24h >= 0 else "red"
                st.metric(
                    label=symbol,
                    value=f"${price:,.2f}",
                    delta=f"{change_24h:.2f}%"
                )
        except Exception as e:
            st.error(f"{symbol}: Error")

# ============ OPEN TRADES ============
st.header("📊 Open Trades")

open_trades = db_manager.get_open_trades()

if open_trades:
    trades_data = []
    for trade in open_trades:
        # Get current price
        try:
            ticker = data_fetcher.fetch_ticker(trade.symbol)
            current_price = ticker.get('last', trade.entry_price) if ticker else trade.entry_price
        except:
            current_price = trade.entry_price

        # Calculate unrealized PnL
        leverage = config['trading'].get('leverage', 1) if config['trading'].get('market_type') == 'futures' else 1
        position_base = (trade.quantity * leverage) / trade.entry_price

        if trade.direction == 'long':
            unrealized_pnl = (current_price - trade.entry_price) * position_base
        else:
            unrealized_pnl = (trade.entry_price - current_price) * position_base

        pnl_pct = (unrealized_pnl / trade.quantity) * 100

        # Check for pending limit order (split entry)
        pending_order = db_manager.get_pending_limit_order_for_trade(trade.id)

        trades_data.append({
            'ID': trade.id,
            'Symbol': trade.symbol,
            'Direction': trade.direction.upper(),
            'Entry': f"${trade.entry_price:,.2f}",
            'Current': f"${current_price:,.2f}",
            'Size (USD)': f"${trade.quantity:,.2f}",
            'SL': f"${trade.stop_loss:,.2f}" if trade.stop_loss else "N/A",
            'TP': f"${trade.take_profit:,.2f}" if trade.take_profit else "N/A",
            'Unrealized PnL': f"${unrealized_pnl:,.2f} ({pnl_pct:+.1f}%)",
            'Split Entry': '✅ Pending' if pending_order else ('✅ Filled' if getattr(trade, 'split_entry_filled', False) else '❌ No'),
            'Partial TP': '✅ Yes' if getattr(trade, 'partial_tp_taken', False) else '❌ No',
        })

    df_trades = pd.DataFrame(trades_data)

    # Style the dataframe
    def highlight_pnl(val):
        if '+' in str(val):
            return 'background-color: #90EE90'
        elif '-' in str(val) and '$' in str(val):
            return 'background-color: #FFB6C1'
        return ''

    st.dataframe(
        df_trades.style.applymap(highlight_pnl, subset=['Unrealized PnL']),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No open trades")

# ============ PENDING LIMIT ORDERS (Split Entry) ============
st.header("⏳ Pending Limit Orders (Split Entry)")

pending_orders = db_manager.get_pending_limit_orders()

if pending_orders:
    orders_data = []
    for order in pending_orders:
        # Get current price
        try:
            ticker = data_fetcher.fetch_ticker(order.symbol)
            current_price = ticker.get('last', 0) if ticker else 0
        except:
            current_price = 0

        # Calculate distance to fill
        if current_price > 0:
            if order.side == 'buy':
                distance_pct = ((current_price - order.limit_price) / current_price) * 100
                fill_status = "🔴 Above" if current_price > order.limit_price else "🟢 Ready!"
            else:
                distance_pct = ((order.limit_price - current_price) / current_price) * 100
                fill_status = "🔴 Below" if current_price < order.limit_price else "🟢 Ready!"
        else:
            distance_pct = 0
            fill_status = "❓ Unknown"

        time_remaining = order.expiry_time - datetime.now()
        hours_remaining = max(0, time_remaining.total_seconds() / 3600)

        orders_data.append({
            'Order ID': order.id,
            'Trade ID': order.trade_id,
            'Symbol': order.symbol,
            'Side': order.side.upper(),
            'Limit Price': f"${order.limit_price:,.2f}",
            'Current': f"${current_price:,.2f}",
            'Distance': f"{distance_pct:.2f}%",
            'Fill Status': fill_status,
            'Size (USD)': f"${order.quantity:,.2f}",
            'Expires In': f"{hours_remaining:.1f}h",
            'Created': order.created_time.strftime('%H:%M:%S')
        })

    df_orders = pd.DataFrame(orders_data)
    st.dataframe(df_orders, use_container_width=True, hide_index=True)

    # Visual explanation
    st.markdown("""
    **Split Entry Strategy:**
    - 🔵 **40%** entered immediately at market price
    - 🟠 **60%** waiting for limit order to fill (better price near SL)
    - If limit fills → Better average entry → Smaller loss if SL hit
    - If trade closes before fill → Limit order cancelled
    """)
else:
    st.info("No pending limit orders")

# ============ RECENT TRADE HISTORY ============
st.header("📜 Recent Trade History")

trades_df = db_manager.get_trade_history(limit=20)

if not trades_df.empty:
    # Format the dataframe
    display_df = trades_df.copy()
    display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%m/%d %H:%M')
    display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%m/%d %H:%M')
    display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "Open")
    display_df['pnl_percentage'] = display_df['pnl_percentage'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")

    # Select columns to display
    display_df = display_df[['id', 'symbol', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_percentage', 'status', 'entry_time', 'exit_time']]
    display_df.columns = ['ID', 'Symbol', 'Side', 'Entry', 'Exit', 'PnL', 'PnL %', 'Status', 'Entry Time', 'Exit Time']

    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No trade history")

# ============ PERFORMANCE METRICS ============
st.header("📈 Performance Summary")

# Calculate metrics from closed trades
closed_trades = trades_df[trades_df['status'] == 'closed']

if not closed_trades.empty:
    col1, col2, col3, col4, col5 = st.columns(5)

    total_trades = len(closed_trades)
    winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
    losing_trades = len(closed_trades[closed_trades['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = closed_trades['pnl'].sum()

    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Wins / Losses", f"{winning_trades} / {losing_trades}")
    with col4:
        st.metric("Total PnL", f"${total_pnl:,.2f}", delta=f"{total_pnl:+,.2f}")
    with col5:
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        st.metric("Avg PnL/Trade", f"${avg_pnl:,.2f}")

# ============ SPLIT ENTRY CONFIGURATION ============
st.sidebar.header("⚙️ Split Entry Config")
split_cfg = config.get('risk', {}).get('split_entry', {})
st.sidebar.markdown(f"""
- **Enabled:** {'✅ Yes' if split_cfg.get('enabled', False) else '❌ No'}
- **Immediate:** {split_cfg.get('immediate_fraction', 0.4)*100:.0f}%
- **Limit Order:** {split_cfg.get('limit_fraction', 0.6)*100:.0f}%
- **Distance to SL:** {split_cfg.get('limit_distance_from_sl', 0.3)*100:.0f}%
- **Expiry:** {split_cfg.get('limit_order_expiry_hours', 4)}h
""")

# ============ PARTIAL TP CONFIGURATION ============
st.sidebar.header("⚙️ Partial TP Config")
partial_cfg = config.get('risk', {}).get('partial_tp', {})
st.sidebar.markdown(f"""
- **Enabled:** {'✅ Yes' if partial_cfg.get('enabled', False) else '❌ No'}
- **Close Fraction:** {partial_cfg.get('close_fraction', 0.3)*100:.0f}%
- **Trigger R:R:** {partial_cfg.get('rr_trigger', 1.0)}:1
""")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
