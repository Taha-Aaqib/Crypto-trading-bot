"""
Backtest Result Visualizations
Generate charts and plots for FYP presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
from src.utils.logger import get_logger

logger = get_logger()

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class BacktestVisualizer:
    """
    Generate visualizations for backtest results
    
    Features:
    - Equity curve with drawdown overlay
    - Trade distribution and win/loss analysis
    - Monthly/weekly returns heatmap
    - Performance metrics dashboard
    - Trade duration analysis
    """
    
    def __init__(self, output_dir: str = "data/backtest"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Backtest visualizer initialized - Output: {output_dir}")
    
    def generate_all_plots(
        self,
        backtest_results: Dict,
        metrics: Dict,
        save_prefix: str = "backtest"
    ) -> Dict[str, str]:
        """
        Generate all visualization plots
        
        Args:
            backtest_results: Results from BacktestEngine
            metrics: Calculated metrics from BacktestMetrics
            save_prefix: Prefix for saved file names
            
        Returns:
            Dictionary mapping plot type to file path
        """
        logger.info("Generating backtest visualizations...")
        
        saved_files = {}
        
        # 1. Equity curve with drawdown
        equity_file = self.plot_equity_curve(
            backtest_results.get('equity_curve', []),
            save_path=f"{self.output_dir}/{save_prefix}_equity.png"
        )
        if equity_file:
            saved_files['equity_curve'] = equity_file
        
        # 2. Trade distribution
        trade_dist_file = self.plot_trade_distribution(
            backtest_results.get('closed_positions', []),
            save_path=f"{self.output_dir}/{save_prefix}_trades.png"
        )
        if trade_dist_file:
            saved_files['trade_distribution'] = trade_dist_file
        
        # 3. Performance dashboard
        dashboard_file = self.plot_performance_dashboard(
            metrics,
            save_path=f"{self.output_dir}/{save_prefix}_dashboard.png"
        )
        if dashboard_file:
            saved_files['dashboard'] = dashboard_file
        
        # 4. Monthly returns heatmap
        returns_file = self.plot_monthly_returns(
            backtest_results.get('equity_curve', []),
            save_path=f"{self.output_dir}/{save_prefix}_monthly_returns.png"
        )
        if returns_file:
            saved_files['monthly_returns'] = returns_file
        
        logger.info(f"Generated {len(saved_files)} visualization files")
        
        return saved_files
    
    def plot_equity_curve(
        self,
        equity_curve: List[Tuple[datetime, float]],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot equity curve with drawdown overlay"""
        if not equity_curve or len(equity_curve) < 2:
            logger.warning("Insufficient data for equity curve plot")
            return None
        
        # Extract data
        dates = [dt for dt, _ in equity_curve]
        equity_values = np.array([eq for _, eq in equity_curve])
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max * 100
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Equity curve
        ax1.plot(dates, equity_values, linewidth=2, color='#2E86AB', label='Equity')
        ax1.fill_between(dates, equity_values, alpha=0.3, color='#2E86AB')
        ax1.axhline(y=equity_values[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        ax2.fill_between(dates, drawdown, 0, color='#A23B72', alpha=0.7)
        ax2.plot(dates, drawdown, linewidth=1.5, color='#A23B72')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_trade_distribution(
        self,
        closed_positions: List,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot trade distribution and win/loss analysis"""
        if not closed_positions:
            logger.warning("No trades to plot")
            return None
        
        # Extract trade data
        pnls = [pos.realized_pnl for pos in closed_positions]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p <= 0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. P&L Distribution
        ax1 = axes[0, 0]
        ax1.hist(winning_pnls, bins=20, alpha=0.7, color='#06D6A0', label=f'Wins ({len(winning_pnls)})', edgecolor='black')
        ax1.hist(losing_pnls, bins=20, alpha=0.7, color='#EF476F', label=f'Losses ({len(losing_pnls)})', edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.set_xlabel('P&L ($)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Trade P&L Distribution', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative P&L
        ax2 = axes[0, 1]
        cumulative_pnl = np.cumsum(pnls)
        trade_numbers = list(range(1, len(pnls) + 1))
        ax2.plot(trade_numbers, cumulative_pnl, linewidth=2, color='#2E86AB', marker='o', markersize=4)
        ax2.fill_between(trade_numbers, cumulative_pnl, alpha=0.3, color='#2E86AB')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Trade Number', fontweight='bold')
        ax2.set_ylabel('Cumulative P&L ($)', fontweight='bold')
        ax2.set_title('Cumulative P&L Over Trades', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 3. Win/Loss Pie Chart
        ax3 = axes[1, 0]
        labels = ['Wins', 'Losses']
        sizes = [len(winning_pnls), len(losing_pnls)]
        colors = ['#06D6A0', '#EF476F']
        explode = (0.05, 0.05)
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11})
        ax3.set_title('Win/Loss Ratio', fontweight='bold', fontsize=12)
        
        # 4. Box plot of wins vs losses
        ax4 = axes[1, 1]
        box_data = [winning_pnls, [abs(x) for x in losing_pnls]]
        bp = ax4.boxplot(box_data, labels=['Wins', 'Losses'], patch_artist=True,
                         boxprops=dict(facecolor='#A8DADC', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
        bp['boxes'][0].set_facecolor('#06D6A0')
        bp['boxes'][1].set_facecolor('#EF476F')
        ax4.set_ylabel('Absolute P&L ($)', fontweight='bold')
        ax4.set_title('Win/Loss Magnitude Distribution', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade distribution: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_performance_dashboard(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Create performance metrics dashboard"""
        if not metrics:
            logger.warning("No metrics to plot")
            return None
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Helper function for metric boxes
        def add_metric_box(ax, title, value, subtitle="", color='#2E86AB'):
            ax.axis('off')
            ax.text(0.5, 0.7, title, ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='#333')
            ax.text(0.5, 0.4, value, ha='center', va='center', 
                   fontsize=24, fontweight='bold', color=color)
            if subtitle:
                ax.text(0.5, 0.15, subtitle, ha='center', va='center',
                       fontsize=9, color='#666')
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                      fill=False, edgecolor=color, linewidth=2))
        
        # Row 1: Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        total_return = metrics.get('total_return_pct', 0)
        color = '#06D6A0' if total_return >= 0 else '#EF476F'
        add_metric_box(ax1, 'Total Return', f"{total_return:.2f}%", 
                      f"${metrics.get('total_return_amount', 0):,.2f}", color=color)
        
        ax2 = fig.add_subplot(gs[0, 1])
        win_rate = metrics.get('win_rate', 0)
        color = '#06D6A0' if win_rate >= 50 else '#F18F01'
        add_metric_box(ax2, 'Win Rate', f"{win_rate:.1f}%",
                      f"{metrics.get('winning_trades', 0)}/{metrics.get('total_trades', 0)} trades", color=color)
        
        ax3 = fig.add_subplot(gs[0, 2])
        profit_factor = metrics.get('profit_factor', 0)
        color = '#06D6A0' if profit_factor >= 1.5 else ('#F18F01' if profit_factor >= 1.0 else '#EF476F')
        pf_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        add_metric_box(ax3, 'Profit Factor', pf_display, 
                      'Gross Profit / Gross Loss', color=color)
        
        # Row 2: Risk metrics
        ax4 = fig.add_subplot(gs[1, 0])
        sharpe = metrics.get('sharpe_ratio', 0)
        color = '#06D6A0' if sharpe >= 1.0 else ('#F18F01' if sharpe >= 0 else '#EF476F')
        add_metric_box(ax4, 'Sharpe Ratio', f"{sharpe:.2f}",
                      'Risk-adjusted return', color=color)
        
        ax5 = fig.add_subplot(gs[1, 1])
        max_dd = metrics.get('max_drawdown_pct', 0)
        color = '#06D6A0' if max_dd <= 10 else ('#F18F01' if max_dd <= 20 else '#EF476F')
        add_metric_box(ax5, 'Max Drawdown', f"{max_dd:.2f}%",
                      f"${metrics.get('max_drawdown_amount', 0):,.2f}", color=color)
        
        ax6 = fig.add_subplot(gs[1, 2])
        rr_ratio = metrics.get('risk_reward_ratio', 0)
        color = '#06D6A0' if rr_ratio >= 2.0 else ('#F18F01' if rr_ratio >= 1.0 else '#EF476F')
        rr_display = f"{rr_ratio:.2f}" if rr_ratio != float('inf') else "∞"
        add_metric_box(ax6, 'Risk-Reward', rr_display,
                      'Avg Win / Avg Loss', color=color)
        
        # Row 3: Additional metrics
        ax7 = fig.add_subplot(gs[2, 0])
        expectancy = metrics.get('expectancy', 0)
        color = '#06D6A0' if expectancy > 0 else '#EF476F'
        add_metric_box(ax7, 'Expectancy', f"${expectancy:.2f}",
                      'Expected profit per trade', color=color)
        
        ax8 = fig.add_subplot(gs[2, 1])
        cagr = metrics.get('cagr', 0)
        color = '#06D6A0' if cagr > 0 else '#EF476F'
        add_metric_box(ax8, 'CAGR', f"{cagr:.2f}%",
                      f"{metrics.get('trading_period_days', 0)} days", color=color)
        
        ax9 = fig.add_subplot(gs[2, 2])
        max_cons_loss = metrics.get('max_consecutive_losses', 0)
        color = '#06D6A0' if max_cons_loss <= 3 else ('#F18F01' if max_cons_loss <= 5 else '#EF476F')
        add_metric_box(ax9, 'Max Consecutive Losses', str(max_cons_loss),
                      f"Max wins: {metrics.get('max_consecutive_wins', 0)}", color=color)
        
        fig.suptitle('Backtest Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance dashboard: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_monthly_returns(
        self,
        equity_curve: List[Tuple[datetime, float]],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot monthly returns heatmap"""
        if not equity_curve or len(equity_curve) < 30:
            logger.warning("Insufficient data for monthly returns plot")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Resample to daily and calculate returns
        df_daily = df.resample('D').last().ffill()
        df_daily['returns'] = df_daily['equity'].pct_change()
        
        # Group by year and month
        df_daily['year'] = df_daily.index.year
        df_daily['month'] = df_daily.index.month
        
        # Calculate monthly returns
        monthly_returns = df_daily.groupby(['year', 'month'])['returns'].apply(
            lambda x: (1 + x).prod() - 1
        ).unstack(fill_value=0) * 100
        
        if monthly_returns.empty:
            logger.warning("No monthly returns data available")
            return None
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, max(6, len(monthly_returns) * 0.6)))
        
        sns.heatmap(monthly_returns, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   linewidths=1, cbar_kws={'label': 'Return (%)'}, ax=ax,
                   vmin=-10, vmax=10)
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Year', fontsize=12, fontweight='bold')
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved monthly returns: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
