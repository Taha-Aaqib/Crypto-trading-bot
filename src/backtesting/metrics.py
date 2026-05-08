"""
Backtesting Performance Metrics
Calculates comprehensive trading performance statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger()


class BacktestMetrics:
    """
    Calculate comprehensive backtest performance metrics

    Metrics include (as per FYP proposal requirements):
    - Win rate and accuracy
    - Profit factor
    - Sharpe ratio
    - Maximum drawdown
    - Total return and CAGR
    - Average win/loss
    - Risk-reward ratio
    """

    def __init__(self):
        pass

    def calculate_all_metrics(self, backtest_results: Dict) -> Dict:
        """
        Calculate all performance metrics from backtest results

        Args:
            backtest_results: Results dictionary from BacktestEngine

        Returns:
            Dictionary with all calculated metrics
        """
        logger.info("Calculating comprehensive performance metrics...")

        closed_positions = backtest_results.get('closed_positions', [])
        equity_curve = backtest_results.get('equity_curve', [])
        initial_capital = backtest_results.get('initial_capital', 10000)
        final_equity = backtest_results.get('final_equity', initial_capital)

        if not closed_positions:
            logger.warning("No closed positions to analyze")
            return self._empty_metrics(initial_capital, final_equity)

        # Extract trade data
        pnls = [pos.realized_pnl for pos in closed_positions]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl <= 0]

        # Calculate metrics
        total_return_value = self._calculate_total_return(
            initial_capital, final_equity)
        metrics = {
            # Basic metrics
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return_value,  # For compatibility
            'total_return_pct': total_return_value,  # Keep original
            'total_return_amount': final_equity - initial_capital,

            # Trade statistics
            'total_trades': len(closed_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': self._calculate_win_rate(winning_trades, losing_trades),

            # Profit metrics
            'gross_profit': sum(winning_trades) if winning_trades else 0,
            'gross_loss': abs(sum(losing_trades)) if losing_trades else 0,
            'net_profit': sum(pnls),
            'profit_factor': self._calculate_profit_factor(winning_trades, losing_trades),

            # Average trade metrics
            'average_win': np.mean(winning_trades) if winning_trades else 0,
            'average_loss': abs(np.mean(losing_trades)) if losing_trades else 0,
            'average_trade': np.mean(pnls),
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': abs(min(losing_trades)) if losing_trades else 0,

            # Risk metrics
            'max_drawdown_pct': self._calculate_max_drawdown(equity_curve),
            'max_drawdown_amount': self._calculate_max_drawdown_amount(equity_curve),
            'sharpe_ratio': self._calculate_sharpe_ratio(equity_curve),
            'sortino_ratio': self._calculate_sortino_ratio(equity_curve),

            # Risk-reward
            'risk_reward_ratio': self._calculate_risk_reward_ratio(winning_trades, losing_trades),
            'expectancy': self._calculate_expectancy(winning_trades, losing_trades),

            # Time-based metrics
            'trading_period_days': self._calculate_trading_period(equity_curve),
            'cagr': self._calculate_cagr(initial_capital, final_equity, equity_curve),
            'avg_trade_duration': self._calculate_avg_trade_duration(closed_positions),

            # Consecutive trades
            'max_consecutive_wins': self._calculate_max_consecutive_wins(pnls),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(pnls),

            # Additional statistics
            'total_commission_paid': sum(pos.commission for pos in closed_positions),
            'win_loss_ratio': len(winning_trades) / len(losing_trades) if losing_trades else float('inf'),
        }

        # Log key metrics
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
        logger.info("=" * 60)

        return metrics

    def _calculate_total_return(self, initial: float, final: float) -> float:
        """Calculate total return percentage"""
        if initial == 0:
            return 0
        return ((final - initial) / initial) * 100

    def _calculate_win_rate(self, winning_trades: List, losing_trades: List) -> float:
        """Calculate win rate percentage"""
        total = len(winning_trades) + len(losing_trades)
        if total == 0:
            return 0
        return (len(winning_trades) / total) * 100

    def _calculate_profit_factor(self, winning_trades: List, losing_trades: List) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        Values > 1.0 indicate profitable strategy
        """
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0

        return gross_profit / gross_loss

    def _calculate_max_drawdown(self, equity_curve: List[Tuple[datetime, float]]) -> float:
        """
        Calculate maximum drawdown percentage
        Important metric for FYP evaluation
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0

        # Extract equity values
        equity_values = [eq for _, eq in equity_curve]

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)

        # Calculate drawdown
        drawdown = (equity_values - running_max) / running_max * 100

        return abs(min(drawdown))

    def _calculate_max_drawdown_amount(self, equity_curve: List[Tuple[datetime, float]]) -> float:
        """Calculate maximum drawdown in dollar amount"""
        if not equity_curve or len(equity_curve) < 2:
            return 0

        equity_values = [eq for _, eq in equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdown = equity_values - running_max

        return abs(min(drawdown))

    def _calculate_sharpe_ratio(
        self,
        equity_curve: List[Tuple[datetime, float]],
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio (annualized)
        Measures risk-adjusted returns
        Required by FYP proposal
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0

        # Calculate daily returns
        equity_values = np.array([eq for _, eq in equity_curve])
        returns = np.diff(equity_values) / equity_values[:-1]

        if len(returns) == 0:
            return 0

        # Calculate excess returns
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf_rate

        # Calculate Sharpe ratio
        if np.std(excess_returns) == 0:
            return 0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)

        # Annualize (assuming 252 trading days)
        return sharpe * np.sqrt(252)

    def _calculate_sortino_ratio(
        self,
        equity_curve: List[Tuple[datetime, float]],
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only considers downside volatility)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0

        equity_values = np.array([eq for _, eq in equity_curve])
        returns = np.diff(equity_values) / equity_values[:-1]

        if len(returns) == 0:
            return 0

        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf_rate

        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0

        sortino = np.mean(excess_returns) / np.std(downside_returns)

        return sortino * np.sqrt(252)

    def _calculate_risk_reward_ratio(self, winning_trades: List, losing_trades: List) -> float:
        """
        Calculate average risk-reward ratio
        Important for FYP proposal validation
        """
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1

        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0

        return avg_win / avg_loss

    def _calculate_expectancy(self, winning_trades: List, losing_trades: List) -> float:
        """
        Calculate expectancy (average amount you expect to win per trade)
        """
        total_trades = len(winning_trades) + len(losing_trades)
        if total_trades == 0:
            return 0

        win_rate = len(winning_trades) / total_trades
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _calculate_trading_period(self, equity_curve: List[Tuple[datetime, float]]) -> int:
        """Calculate total trading period in days"""
        if not equity_curve or len(equity_curve) < 2:
            return 0

        start_date = equity_curve[0][0]
        end_date = equity_curve[-1][0]

        return (end_date - start_date).days

    def _calculate_cagr(
        self,
        initial_capital: float,
        final_equity: float,
        equity_curve: List[Tuple[datetime, float]]
    ) -> float:
        """
        Calculate Compound Annual Growth Rate
        Important metric for long-term strategy evaluation
        """
        days = self._calculate_trading_period(equity_curve)

        if days == 0 or initial_capital == 0:
            return 0

        years = days / 365.25

        if years == 0:
            return 0

        cagr = (pow(final_equity / initial_capital, 1 / years) - 1) * 100

        return cagr

    def _calculate_avg_trade_duration(self, closed_positions: List) -> float:
        """Calculate average trade duration in hours"""
        if not closed_positions:
            return 0

        durations = []
        for pos in closed_positions:
            if pos.exit_timestamp and pos.entry_timestamp:
                duration = (pos.exit_timestamp -
                            pos.entry_timestamp).total_seconds() / 3600
                durations.append(duration)

        return np.mean(durations) if durations else 0

    def _calculate_max_consecutive_wins(self, pnls: List[float]) -> int:
        """Calculate maximum consecutive winning trades"""
        if not pnls:
            return 0

        max_wins = 0
        current_wins = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0

        return max_wins

    def _calculate_max_consecutive_losses(self, pnls: List[float]) -> int:
        """Calculate maximum consecutive losing trades"""
        if not pnls:
            return 0

        max_losses = 0
        current_losses = 0

        for pnl in pnls:
            if pnl <= 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def _empty_metrics(self, initial: float, final: float) -> Dict:
        """Return empty metrics dictionary when no trades"""
        return {
            'initial_capital': initial,
            'final_equity': final,
            'total_return_pct': 0,
            'total_return_amount': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'profit_factor': 0,
            'average_win': 0,
            'average_loss': 0,
            'average_trade': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_amount': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'risk_reward_ratio': 0,
            'expectancy': 0,
            'trading_period_days': 0,
            'cagr': 0,
            'avg_trade_duration': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'total_commission_paid': 0,
            'win_loss_ratio': 0,
        }

    def generate_summary_report(self, metrics: Dict) -> str:
        """
        Generate a formatted text summary report
        Useful for FYP presentation and documentation
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("BACKTEST PERFORMANCE SUMMARY REPORT")
        report.append("=" * 70)
        report.append("")

        # Capital and Returns
        report.append("CAPITAL & RETURNS:")
        report.append(
            f"  Initial Capital:        ${metrics['initial_capital']:>12,.2f}")
        report.append(
            f"  Final Equity:           ${metrics['final_equity']:>12,.2f}")
        report.append(
            f"  Net Profit:             ${metrics['net_profit']:>12,.2f}")
        report.append(
            f"  Total Return:           {metrics['total_return_pct']:>12.2f}%")
        report.append(f"  CAGR:                   {metrics['cagr']:>12.2f}%")
        report.append("")

        # Trade Statistics
        report.append("TRADE STATISTICS:")
        report.append(
            f"  Total Trades:           {metrics['total_trades']:>12}")
        report.append(
            f"  Winning Trades:         {metrics['winning_trades']:>12}")
        report.append(
            f"  Losing Trades:          {metrics['losing_trades']:>12}")
        report.append(
            f"  Win Rate:               {metrics['win_rate']:>12.2f}%")
        report.append(
            f"  Win/Loss Ratio:         {metrics['win_loss_ratio']:>12.2f}")
        report.append("")

        # Profitability
        report.append("PROFITABILITY:")
        report.append(
            f"  Gross Profit:           ${metrics['gross_profit']:>12,.2f}")
        report.append(
            f"  Gross Loss:             ${metrics['gross_loss']:>12,.2f}")
        report.append(
            f"  Profit Factor:          {metrics['profit_factor']:>12.2f}")
        report.append(
            f"  Average Win:            ${metrics['average_win']:>12,.2f}")
        report.append(
            f"  Average Loss:           ${metrics['average_loss']:>12,.2f}")
        report.append(
            f"  Largest Win:            ${metrics['largest_win']:>12,.2f}")
        report.append(
            f"  Largest Loss:           ${metrics['largest_loss']:>12,.2f}")
        report.append(
            f"  Expectancy:             ${metrics['expectancy']:>12,.2f}")
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS:")
        report.append(
            f"  Maximum Drawdown:       {metrics['max_drawdown_pct']:>12.2f}%")
        report.append(
            f"  Max Drawdown ($):       ${metrics['max_drawdown_amount']:>12,.2f}")
        report.append(
            f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>12.2f}")
        report.append(
            f"  Sortino Ratio:          {metrics['sortino_ratio']:>12.2f}")
        report.append(
            f"  Risk-Reward Ratio:      {metrics['risk_reward_ratio']:>12.2f}")
        report.append("")

        # Additional Statistics
        report.append("ADDITIONAL STATISTICS:")
        report.append(
            f"  Trading Period (days):  {metrics['trading_period_days']:>12}")
        report.append(
            f"  Avg Trade Duration:     {metrics['avg_trade_duration']:>12.1f} hours")
        report.append(
            f"  Max Consecutive Wins:   {metrics['max_consecutive_wins']:>12}")
        report.append(
            f"  Max Consecutive Losses: {metrics['max_consecutive_losses']:>12}")
        report.append(
            f"  Total Commissions:      ${metrics['total_commission_paid']:>12,.2f}")
        report.append("")
        report.append("=" * 70)

        return "\n".join(report)
