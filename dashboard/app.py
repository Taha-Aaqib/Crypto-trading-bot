"""
Streamlit Dashboard for Trading Bot
Paper & Live mode with full Binance-level position details
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import ccxt
from dotenv import load_dotenv

# Add parent directory to path FIRST (before importing src modules)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env variables (API keys etc.) before config reads them
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from src.utils.helpers import load_config, update_config_fields
from src.utils.db_manager import DatabaseManager

# ─── Page Configuration ───
st.set_page_config(
    page_title="SMC Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS: hide running indicator + smooth fragment updates ───
st.markdown("""
<style>
    /* Hide the 'Running...' spinner so fragments don't flash the whole page */
    [data-testid="stStatusWidget"] { display: none !important; }
    /* Smooth metric transitions */
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Resources ───
@st.cache_resource
def get_db_manager():
    config = load_config()
    return DatabaseManager(config)


@st.cache_resource
def get_paper_exchange():
    """Exchange for paper mode (testnet / public data only)"""
    config = load_config()
    exchange_name = config['exchange']['name']
    exchange_class = getattr(ccxt, exchange_name)
    market_type = config.get('trading', {}).get('market_type', 'spot')
    options = {}
    if market_type == 'futures':
        options['defaultType'] = 'future'
    exchange = exchange_class({'enableRateLimit': True, 'options': options})
    if config.get('trading', {}).get('mode') == 'paper' or config['exchange'].get('testnet'):
        exchange.set_sandbox_mode(True)
    return exchange


@st.cache_resource
def get_live_exchange():
    """Exchange for live mode — authenticated, real Binance"""
    config = load_config()
    exchange_name = config['exchange']['name']
    exchange_class = getattr(ccxt, exchange_name)
    market_type = config.get('trading', {}).get('market_type', 'spot')
    default_type = 'future' if market_type == 'futures' else 'spot'
    exchange = exchange_class({
        'apiKey': config['exchange']['api_key'],
        'secret': config['exchange']['api_secret'],
        'enableRateLimit': True,
        'options': {'defaultType': default_type}
    })
    return exchange


def get_exchange_for_mode(mode):
    if mode == 'live':
        return get_live_exchange()
    return get_paper_exchange()


def fetch_live_prices(symbols, exchange):
    """Fetch live prices + mark prices for given symbols"""
    prices = {}
    for symbol in symbols:
        try:
            ticker = exchange.fetch_ticker(symbol)
            if ticker and 'last' in ticker:
                prices[symbol] = {
                    'last': ticker['last'],
                    'mark': ticker.get('mark', ticker['last']),
                    'high': ticker.get('high'),
                    'low': ticker.get('low'),
                    'change_pct': ticker.get('percentage', 0),
                    'volume': ticker.get('quoteVolume', 0),
                }
        except Exception:
            pass
    return prices


def fetch_live_balance(exchange):
    """Fetch USDT balance from exchange"""
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get('USDT', {})
        return {
            'total': float(usdt.get('total', 0) or 0),
            'free': float(usdt.get('free', 0) or 0),
            'used': float(usdt.get('used', 0) or 0),
        }
    except Exception:
        return None


def fetch_live_positions(exchange):
    """Fetch all open positions from Binance futures"""
    try:
        positions = exchange.fetch_positions()
        open_pos = []
        for pos in positions:
            amt = abs(float(pos.get('contracts', 0) or 0))
            notional = abs(float(pos.get('notional', 0) or 0))
            if amt > 0 or notional > 0:
                open_pos.append(pos)
        return open_pos
    except Exception:
        return []


# ─── Initialize ───
db_manager = get_db_manager()
_config = load_config()
_trading_mode = _config.get('trading', {}).get('mode', 'paper')
_market_type = _config.get('trading', {}).get('market_type', 'spot')
_leverage = _config.get('trading', {}).get('leverage', 1)
_initial_capital = _config.get('backtest', {}).get('initial_capital', 1000)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Trading Mode (Paper / Live) ──
    st.header("🔑 Trading Mode")
    tm_options = ["paper", "live"]
    tm_idx = tm_options.index(_trading_mode) if _trading_mode in tm_options else 0
    selected_tm = st.radio(
        "Account",
        tm_options,
        index=tm_idx,
        format_func=lambda x: "📝 PAPER (Simulated)" if x == "paper" else "💰 LIVE (Real Money)",
        horizontal=True,
        key="tm_radio"
    )
    if selected_tm == 'live':
        st.warning("⚠️ LIVE mode uses real funds!")

    st.markdown("---")

    # ── Market Settings ──
    st.header("⚙️ Market Settings")
    mk_options = ["spot", "futures"]
    mk_idx = mk_options.index(_market_type) if _market_type in mk_options else 0
    selected_mk = st.radio(
        "Market Type",
        mk_options,
        index=mk_idx,
        format_func=lambda x: "🟢 SPOT (Buy Only)" if x == "spot" else "🟠 FUTURES (Long & Short)",
        horizontal=True,
        key="mk_radio"
    )

    if selected_mk == "futures":
        selected_lev = st.select_slider(
            "Leverage",
            options=[1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100, 125],
            value=_leverage if _leverage in [1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100, 125] else 10,
            key="lev_slider"
        )
        st.caption("Isolated margin only")
    else:
        selected_lev = 1

    # Detect changes
    settings_changed = (
        selected_tm != _trading_mode or
        selected_mk != _market_type or
        (selected_mk == 'futures' and selected_lev != _leverage)
    )

    if settings_changed:
        open_check = db_manager.get_open_trades()
        switching = (selected_mk != _market_type or selected_tm != _trading_mode)
        if switching and open_check:
            st.warning(f"⚠️ Close all {len(open_check)} open trades before switching modes.")
        elif st.button("💾 Save & Apply", type="primary", use_container_width=True):
            updates = {
                'trading.mode': selected_tm,
                'trading.market_type': selected_mk,
                'trading.leverage': selected_lev,
                'trading.margin_mode': 'isolated',
            }
            if selected_tm == 'live':
                updates['exchange.testnet'] = False
            try:
                update_config_fields(updates)
                st.success("Settings saved! Restart the bot for changes to take effect.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")
    else:
        badge = "🔴 LIVE" if _trading_mode == 'live' else "🟢 PAPER"
        if _market_type == 'futures':
            badge += f" | FUTURES {_leverage}x ISOLATED"
        else:
            badge += " | SPOT (Buy Only)"
        st.info(f"**{badge}**")


# ═══════════════════════════════════════════════════════════════
#  TITLE & WALLET OVERVIEW (inside fragment for partial refresh)
# ═══════════════════════════════════════════════════════════════
if _trading_mode == 'live':
    st.title("💰 LIVE Trading Dashboard")
else:
    st.title("📝 Paper Trading Dashboard")
st.markdown("---")


@st.fragment(run_every=10)
def wallet_overview():
    """Wallet metrics — refreshes every 10s without full page reload"""
    _cfg = load_config()
    _mode = _cfg.get('trading', {}).get('mode', 'paper')
    _cap = _cfg.get('backtest', {}).get('initial_capital', 1000)

    trades_data = db_manager.get_trade_history(limit=1000)
    _closed = pd.DataFrame()
    _winning = _losing = _total = _open = 0
    _wr = _tpnl = 0.0

    if not trades_data.empty:
        _total = len(trades_data)
        _closed = trades_data[trades_data['status'] == 'closed']
        _open = len(trades_data[trades_data['status'] == 'open'])
        if not _closed.empty:
            _winning = len(_closed[_closed['pnl'] > 0])
            _losing = len(_closed[_closed['pnl'] < 0])
            _wr = (_winning / len(_closed) * 100) if len(_closed) > 0 else 0
            _tpnl = _closed['pnl'].sum()

    # Live: fetch real exchange balance
    _live_bal = None
    if _mode == 'live':
        try:
            _exch = get_live_exchange()
            _live_bal = fetch_live_balance(_exch)
        except Exception:
            pass

    if _live_bal:
        wt = _live_bal['total']
        wf = _live_bal['free']
        wu = _live_bal['used']
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Total Balance", f"${wt:.2f}")
        with c2:
            st.metric("Available", f"${wf:.2f}")
        with c3:
            st.metric("In Positions", f"${wu:.2f}")
        with c4:
            st.metric("Realized P&L", f"${_tpnl:.2f}", delta=f"${_tpnl:+.2f}")
        with c5:
            st.metric("Win Rate", f"{_wr:.1f}%" if (_winning + _losing) > 0 else "N/A")
        with c6:
            st.metric("Open / Total", f"{_open} / {_total}")
    else:
        # Include partial PnL from open trades in wallet balance
        _partial_pnl_open = 0.0
        if not trades_data.empty:
            _open_df = trades_data[trades_data['status'] == 'open']
            if 'partial_pnl' in _open_df.columns:
                _partial_pnl_open = _open_df['partial_pnl'].fillna(0).sum()
        wallet_balance = _cap + _tpnl + _partial_pnl_open
        open_margin = trades_data[trades_data['status'] == 'open']['quantity'].sum() if not trades_data.empty else 0
        wallet_pct = ((wallet_balance - _cap) / _cap * 100) if _cap else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Wallet Balance", f"${wallet_balance:.2f}", delta=f"{wallet_pct:+.2f}%")
        with c2:
            st.metric("Available", f"${max(wallet_balance - open_margin, 0):.2f}")
        with c3:
            _total_realized = _tpnl + _partial_pnl_open
            st.metric("Realized P&L", f"${_total_realized:.2f}", delta=f"${_total_realized:+.2f}")
        with c4:
            st.metric("Win Rate", f"{_wr:.1f}%" if (_winning + _losing) > 0 else "N/A")
        with c5:
            st.metric("Open / Total", f"{_open} / {_total}")


wallet_overview()
st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Open Trades", "📊 Performance", "📜 Trade History", "⚡ Market Info"])


# ─── TAB 1: Open Trades (Binance-level detail) ───
with tab1:
    if _trading_mode == 'live':
        st.header("Open Positions — Live Exchange")
    else:
        st.header("Open Trades — Paper Mode")

    @st.fragment(run_every=5)
    def live_open_trades():
        _cfg = load_config()
        _mt = _cfg.get('trading', {}).get('market_type', 'spot')
        _lev = _cfg.get('trading', {}).get('leverage', 1)
        _mode = _cfg.get('trading', {}).get('mode', 'paper')
        _is_futures = (_mt == 'futures')
        _is_live = (_mode == 'live')

        open_trades_list = db_manager.get_open_trades()
        if not open_trades_list:
            st.info("No open trades")
            st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
            return

        open_symbols = list(set(t.symbol for t in open_trades_list))
        _exch = get_exchange_for_mode(_mode)
        live_data = fetch_live_prices(open_symbols, _exch)

        # Real exchange positions (live+futures)
        exchange_positions = {}
        if _is_live and _is_futures:
            for pos in fetch_live_positions(_exch):
                exchange_positions[pos.get('symbol', '')] = pos

        open_rows = []
        total_unrealized = 0.0
        total_margin = 0.0

        for trade in open_trades_list:
            ticker = live_data.get(trade.symbol, {})
            current_price = ticker.get('last')
            mark_price = ticker.get('mark', current_price)

            effective_lev = _lev if _is_futures else 1
            margin = trade.quantity or 0
            entry_px = trade.entry_price or 0
            if entry_px <= 0 or margin <= 0:
                continue  # skip corrupt trade
            position_base = (margin * effective_lev) / entry_px
            notional = margin * effective_lev
            total_margin += margin

            # Unrealized PnL (uses mark price like Binance)
            calc_price = mark_price or current_price
            unrealized_pnl = None
            roe_pct = None
            if calc_price:
                if trade.side == 'buy':
                    unrealized_pnl = (calc_price - entry_px) * position_base
                else:
                    unrealized_pnl = (entry_px - calc_price) * position_base
                roe_pct = (unrealized_pnl / margin * 100) if margin else 0
                total_unrealized += unrealized_pnl

            # Liquidation price
            liq_price = None
            if _is_futures and _lev > 1:
                maint = 0.004
                if trade.side == 'buy':
                    liq_price = entry_px * (1 - (1 / _lev) + maint)
                else:
                    liq_price = entry_px * (1 + (1 / _lev) - maint)

            # Margin ratio (safety margin before liquidation)
            margin_ratio = None
            if liq_price and calc_price and _is_futures:
                if trade.side == 'buy':
                    total_range = entry_px - liq_price
                    remaining = calc_price - liq_price
                else:
                    total_range = liq_price - entry_px
                    remaining = liq_price - calc_price
                margin_ratio = max(0, (remaining / total_range * 100)) if total_range > 0 else 0

            # Duration
            if trade.entry_time:
                duration = datetime.now() - trade.entry_time
                hours = duration.total_seconds() / 3600
                if hours < 1:
                    dur_str = f"{int(duration.total_seconds() / 60)}m"
                elif hours < 24:
                    dur_str = f"{hours:.1f}h"
                else:
                    dur_str = f"{hours / 24:.1f}d"
            else:
                dur_str = "—"

            # Override with real exchange data if live
            ex_pos = exchange_positions.get(trade.symbol)
            if ex_pos:
                ex_liq = ex_pos.get('liquidationPrice')
                if ex_liq and float(ex_liq) > 0:
                    liq_price = float(ex_liq)
                ex_mark = ex_pos.get('markPrice')
                if ex_mark:
                    mark_price = float(ex_mark)
                ex_pnl = ex_pos.get('unrealizedPnl')
                if ex_pnl is not None:
                    unrealized_pnl = float(ex_pnl)
                    roe_pct = (unrealized_pnl / margin * 100) if margin else 0
                ex_notional = ex_pos.get('notional')
                if ex_notional:
                    notional = abs(float(ex_notional))
                ex_mr = ex_pos.get('marginRatio')
                if ex_mr:
                    margin_ratio = float(ex_mr) * 100

            # Build row
            row = {
                'Symbol': trade.symbol,
                'Side': f"{'🟢' if trade.side == 'buy' else '🔴'} {'LONG' if trade.side == 'buy' else 'SHORT'}",
            }

            if _is_futures:
                row['Leverage'] = f'{effective_lev}x'

            row['Margin'] = f"${margin:.2f}"
            row['Size'] = f"{position_base:.6f}"
            row['Entry Price'] = f"${entry_px:.2f}"
            row['Entry Time'] = trade.entry_time.strftime('%Y-%m-%d %H:%M') if trade.entry_time else '—'

            if _is_futures:
                row['Mark Price'] = f"${mark_price:.2f}" if mark_price else "—"
            else:
                row['Current'] = f"${current_price:.2f}" if current_price else "—"

            if unrealized_pnl is not None:
                row['PnL (USDT)'] = round(unrealized_pnl, 2)
                row['ROE %'] = round(roe_pct, 2)
            else:
                row['PnL (USDT)'] = None
                row['ROE %'] = None

            if _is_futures:
                row['Liq. Price'] = f"${liq_price:.2f}" if liq_price else "—"
                if margin_ratio is not None:
                    row['Margin Ratio'] = f"{margin_ratio:.1f}%"
                else:
                    row['Margin Ratio'] = "—"

            # Show original SL (pre-breakeven) if available, else current SL
            _orig_sl = getattr(trade, 'original_sl', None)
            if _orig_sl and trade.stop_loss and abs(_orig_sl - trade.stop_loss) > 0.01:
                row['SL'] = f"${trade.stop_loss:.2f} (orig: ${_orig_sl:.2f})"
            else:
                row['SL'] = f"${trade.stop_loss:.2f}" if trade.stop_loss else "—"
            row['TP'] = f"${trade.take_profit:.2f}" if trade.take_profit else "—"
            # Show partial TP info if taken
            _ptp_price = getattr(trade, 'partial_tp_price', None)
            _ppnl = getattr(trade, 'partial_pnl', 0) or 0
            if getattr(trade, 'partial_tp_taken', False) and _ptp_price:
                row['Partial TP'] = f"${_ptp_price:.2f} (+${_ppnl:.2f})"
            else:
                row['Partial TP'] = "—"
            row['Duration'] = dur_str
            open_rows.append(row)

        # Summary cards
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            pnl_c = '#ff4b4b' if total_unrealized < 0 else '#21ba45'
            st.markdown(f'<p style="font-size:0.875rem;color:rgba(250,250,250,0.6);margin:0 0 4px 0">Unrealized PnL</p>'
                        f'<p style="font-size:1.75rem;font-weight:700;color:{pnl_c};margin:0">${total_unrealized:+.2f}</p>',
                        unsafe_allow_html=True)
        with sc2:
            st.metric("Open Positions", len(open_trades_list))
        with sc3:
            st.metric("Margin Used", f"${total_margin:.2f}")
        with sc4:
            avg_roe = (total_unrealized / total_margin * 100) if total_margin else 0
            roe_c = '#ff4b4b' if avg_roe < 0 else '#21ba45'
            st.markdown(f'<p style="font-size:0.875rem;color:rgba(250,250,250,0.6);margin:0 0 4px 0">Avg ROE</p>'
                        f'<p style="font-size:1.75rem;font-weight:700;color:{roe_c};margin:0">{avg_roe:+.2f}%</p>',
                        unsafe_allow_html=True)

        # Styled dataframe with red/green PnL
        df_open = pd.DataFrame(open_rows)
        pnl_cols = [c for c in ['PnL (USDT)', 'ROE %'] if c in df_open.columns]

        def _color_pnl(val):
            if pd.isna(val) or val is None:
                return ''
            return f'color: {"#ff4b4b" if val < 0 else "#21ba45"}; font-weight: bold'

        fmt = {}
        if 'PnL (USDT)' in df_open.columns:
            fmt['PnL (USDT)'] = lambda x: f'${x:+.2f}' if pd.notna(x) else '—'
        if 'ROE %' in df_open.columns:
            fmt['ROE %'] = lambda x: f'{x:+.2f}%' if pd.notna(x) else '—'

        styled_df = df_open.style.map(_color_pnl, subset=pnl_cols).format(fmt, na_rep='—')
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
                   f"{'🔴 LIVE' if _is_live else '📝 Paper'} | "
                   f"{_mt.upper()}{f' {_lev}x' if _is_futures else ''}")

    live_open_trades()


# ─── TAB 2: Performance ───
with tab2:
    st.header("Performance Overview")

    @st.fragment(run_every=15)
    def performance_overview():
        _trades_df = db_manager.get_trade_history(limit=1000)
        _closed = pd.DataFrame()
        _winning = _losing = 0
        _wr = _tpnl = 0.0

        if not _trades_df.empty:
            _closed = _trades_df[_trades_df['status'] == 'closed']
            if not _closed.empty:
                _winning = len(_closed[_closed['pnl'] > 0])
                _losing = len(_closed[_closed['pnl'] < 0])
                _wr = (_winning / len(_closed) * 100) if len(_closed) > 0 else 0
                _tpnl = _closed['pnl'].sum()

        if not _closed.empty:
            closed_sorted = _closed.dropna(subset=['exit_time']).sort_values('exit_time')
            closed_sorted['cumulative_pnl'] = closed_sorted['pnl'].cumsum()
            closed_sorted['wallet'] = _initial_capital + closed_sorted['cumulative_pnl']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=closed_sorted['exit_time'],
                y=closed_sorted['wallet'],
                mode='lines+markers',
                name='Wallet Balance',
                line=dict(color='#00C853', width=2),
                hovertemplate='$%{y:.2f}<extra></extra>'
            ))
            fig.add_hline(y=_initial_capital, line_dash="dash",
                          line_color="gray", annotation_text=f"Start ${_initial_capital}")
            fig.update_layout(
                title="Wallet Equity Curve",
                xaxis_title="Date", yaxis_title="Wallet ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.subheader("📊 Statistics")
                st.write(f"**Initial Capital:** ${_initial_capital:.2f}")
                st.write(f"**Current Balance:** ${_initial_capital + _tpnl:.2f}")
                st.write(f"**Total Trades:** {len(_closed)}")
                st.write(f"**Winning:** {_winning} | **Losing:** {_losing}")
                st.write(f"**Win Rate:** {_wr:.2f}%")
            with pc2:
                st.subheader("💰 P&L")
                avg_win = _closed[_closed['pnl'] > 0]['pnl'].mean() if _winning > 0 else 0
                avg_loss = _closed[_closed['pnl'] < 0]['pnl'].mean() if _losing > 0 else 0
                st.write(f"**Total P&L:** ${_tpnl:.2f}")
                st.write(f"**Return:** {(_tpnl / _initial_capital * 100):.2f}%")
                st.write(f"**Average Win:** ${avg_win:.2f}")
                st.write(f"**Average Loss:** ${avg_loss:.2f}")
                pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                st.write(f"**Profit Factor:** {pf:.2f}")
        else:
            st.info("No closed trades yet — performance data will appear after your first trade closes.")

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    performance_overview()


# ─── TAB 3: Trade History ───
with tab3:
    st.header("Trade History")

    @st.fragment(run_every=15)
    def trade_history_view():
        _trades = db_manager.get_trade_history(limit=1000)

        if not _trades.empty:
            _mt_hist = _config.get('trading', {}).get('market_type', 'spot')
            _lev_hist = _config.get('trading', {}).get('leverage', 1)

            display_df = _trades[['symbol', 'side', 'entry_price', 'exit_price',
                                    'quantity', 'pnl', 'pnl_percentage', 'entry_time',
                                    'exit_time', 'status']].copy()
            display_df.columns = ['Symbol', 'Side', 'Entry', 'Exit', 'Margin ($)',
                                  'P&L ($)', 'P&L %', 'Entry Time', 'Exit Time', 'Status']

            # Add Leverage column
            if _mt_hist == 'futures':
                display_df.insert(2, 'Leverage', f'{_lev_hist}x')
            else:
                display_df.insert(2, 'Leverage', '1x')

            # Format Side to show LONG/SHORT
            display_df['Side'] = display_df['Side'].apply(
                lambda x: f"{'🟢' if x == 'buy' else '🔴'} {'LONG' if x == 'buy' else 'SHORT'}")

            display_df['Entry'] = display_df['Entry'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
            display_df['Exit'] = display_df['Exit'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
            display_df['Margin ($)'] = display_df['Margin ($)'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
            display_df['P&L ($)'] = display_df['P&L ($)'].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "—")
            display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = _trades.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade History",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trade history available")

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    trade_history_view()


# ─── TAB 4: Market Info ───
with tab4:
    st.header("Market Overview")

    @st.fragment(run_every=10)
    def market_info():
        _cfg = load_config()
        _symbols = _cfg.get('trading', {}).get('symbols', ['BTC/USDT', 'ETH/USDT'])
        _mt = _cfg.get('trading', {}).get('market_type', 'spot')
        _mode = _cfg.get('trading', {}).get('mode', 'paper')
        _exch = get_exchange_for_mode(_mode)

        live_data = fetch_live_prices(_symbols, _exch)

        for symbol in _symbols:
            data = live_data.get(symbol, {})
            if not data:
                continue

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                change = data.get('change_pct', 0) or 0
                st.metric(symbol, f"${data['last']:,.2f}", delta=f"{change:+.2f}%")
            with mc2:
                st.metric("24h High", f"${data.get('high', 0):,.2f}" if data.get('high') else "—")
            with mc3:
                st.metric("24h Low", f"${data.get('low', 0):,.2f}" if data.get('low') else "—")
            with mc4:
                vol = data.get('volume', 0) or 0
                if vol > 1_000_000_000:
                    st.metric("24h Volume", f"${vol / 1e9:.2f}B")
                elif vol > 1_000_000:
                    st.metric("24h Volume", f"${vol / 1e6:.1f}M")
                else:
                    st.metric("24h Volume", f"${vol:,.0f}")

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    market_info()


# ─── Footer ───
st.sidebar.markdown("---")
st.sidebar.caption("Positions refresh every 5s | Market data every 10s | History & Performance every 15s")
