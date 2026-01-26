"""
Streamlit Dashboard for Trading Bot
Real-time visualization of signals, performance, and system metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path FIRST (before importing src modules)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from src
from src.utils.helpers import load_config
from src.utils.db_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="SMC Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration


@st.cache_resource
def get_db_manager():
    config = load_config()
    return DatabaseManager(config)


db_manager = get_db_manager()

# Title
st.title("🤖 AI-Enhanced SMC Trading Bot Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    auto_refresh = st.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)

    st.markdown("---")
    st.header("📊 Filters")

    symbol_filter = st.selectbox(
        "Symbol",
        ["All", "BTC/USDT", "ETH/USDT"]
    )

    time_filter = st.selectbox(
        "Time Period",
        ["Today", "Last 7 Days", "Last 30 Days", "All Time"]
    )

# Main content
col1, col2, col3, col4 = st.columns(4)

# Fetch trade data
trades_df = db_manager.get_trade_history(limit=1000)

# Initialize variables
closed_trades = pd.DataFrame()
winning_trades = 0
losing_trades = 0
win_rate = 0
total_pnl = 0

if not trades_df.empty:
    # Calculate metrics
    total_trades = len(trades_df)
    closed_trades = trades_df[trades_df['status'] == 'closed']

    if not closed_trades.empty:
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] < 0])
        win_rate = (winning_trades / len(closed_trades) *
                    100) if len(closed_trades) > 0 else 0
        total_pnl = closed_trades['pnl'].sum()

        # Display metrics
        with col1:
            st.metric("Total Trades", total_trades)

        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col3:
            st.metric("Total P&L", f"${total_pnl:.2f}",
                      delta=f"{total_pnl:.2f}")

        with col4:
            open_trades = len(trades_df[trades_df['status'] == 'open'])
            st.metric("Open Trades", open_trades)
    else:
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", "N/A")
        with col3:
            st.metric("Total P&L", "$0.00")
        with col4:
            open_trades = len(trades_df[trades_df['status'] == 'open'])
            st.metric("Open Trades", open_trades)
        st.info("No closed trades yet. Open trades will show below.")
else:
    with col1:
        st.metric("Total Trades", 0)
    with col2:
        st.metric("Win Rate", "N/A")
    with col3:
        st.metric("Total P&L", "$0.00")
    with col4:
        st.metric("Open Trades", 0)
    st.info("No trading data available. Start the bot to see metrics!")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Performance", "📈 Open Trades", "📜 Trade History", "⚡ Live Signals"])

with tab1:
    st.header("Performance Overview")

    if not trades_df.empty and not closed_trades.empty:
        # Equity curve
        closed_trades_sorted = closed_trades.sort_values('exit_time')
        closed_trades_sorted['cumulative_pnl'] = closed_trades_sorted['pnl'].cumsum(
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closed_trades_sorted['exit_time'],
            y=closed_trades_sorted['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            hovermode='x unified'
        )

        st.plotly_chart(fig, width='stretch')

        # Statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Statistics")
            st.write(f"**Total Trades:** {len(closed_trades)}")
            st.write(f"**Winning Trades:** {winning_trades}")
            st.write(f"**Losing Trades:** {losing_trades}")
            st.write(f"**Win Rate:** {win_rate:.2f}%")

        with col2:
            st.subheader("💰 P&L")
            avg_win = closed_trades[closed_trades['pnl'] >
                                    0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = closed_trades[closed_trades['pnl']
                                     < 0]['pnl'].mean() if losing_trades > 0 else 0

            st.write(f"**Total P&L:** ${total_pnl:.2f}")
            st.write(f"**Average Win:** ${avg_win:.2f}")
            st.write(f"**Average Loss:** ${avg_loss:.2f}")
            st.write(
                f"**Profit Factor:** {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")
    else:
        st.info("No performance data available yet")

with tab2:
    st.header("Open Trades")

    open_trades_list = db_manager.get_open_trades()

    if open_trades_list:
        open_trades_data = []
        for trade in open_trades_list:
            open_trades_data.append({
                'ID': trade.id,
                'Symbol': trade.symbol,
                'Side': trade.side.upper(),
                'Entry Price': f"${trade.entry_price:.2f}",
                'Quantity': f"{trade.quantity:.4f}",
                'Stop Loss': f"${trade.stop_loss:.2f}",
                'Take Profit': f"${trade.take_profit:.2f}",
                'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')
            })

        st.dataframe(pd.DataFrame(open_trades_data), width='stretch')
    else:
        st.info("No open trades")

with tab3:
    st.header("Trade History")

    if not trades_df.empty:
        # Display trade history table
        display_df = trades_df[['symbol', 'side', 'entry_price', 'exit_price',
                                'quantity', 'pnl', 'pnl_percentage', 'entry_time',
                                'exit_time', 'status']].copy()

        # Format columns
        display_df['entry_price'] = display_df['entry_price'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "")
        display_df['exit_price'] = display_df['exit_price'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "")
        display_df['pnl'] = display_df['pnl'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "")
        display_df['pnl_percentage'] = display_df['pnl_percentage'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "")

        st.dataframe(display_df, width='stretch')

        # Download button
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History",
            data=csv,
            file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trade history available")

with tab4:
    st.header("Live Signals")
    # Real-time market signals and analysis
    st.info("Live signals feature - Coming soon")
    st.write("This will display:")
    st.write("- Real-time price alerts")
    st.write("- ML model predictions")
    st.write("- SMC pattern notifications")
    st.write("- Entry/exit signals")

# Auto refresh
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
