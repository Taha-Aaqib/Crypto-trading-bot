"""
Reset Trading Bot - Clear old trades and start fresh
Use this to reset paper trading account and clear stale positions
"""

import sqlite3
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger()


def reset_trading_bot():
    """Reset the trading bot by clearing all trades and resetting balances"""

    print("\n" + "="*60)
    print("RESET TRADING BOT")
    print("="*60)
    print("\nThis will:")
    print("  1. Close all open trades")
    print("  2. Clear trade history")
    print("  3. Reset paper trading balance")
    print("\n" + "="*60)

    response = input(
        "\nAre you sure you want to reset? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Reset cancelled.")
        return

    try:
        # Connect to database
        db_path = 'data/trading_bot.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get stats before reset
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
        open_trades = cursor.fetchone()[0]

        print(f"\nFound:")
        print(f"  - Total trades: {total_trades}")
        print(f"  - Open trades: {open_trades}")

        # Delete all trades
        cursor.execute("DELETE FROM trades")
        deleted_trades = cursor.rowcount

        # Delete all logs (if table exists)
        deleted_logs = 0
        try:
            cursor.execute("DELETE FROM logs")
            deleted_logs = cursor.rowcount
        except sqlite3.OperationalError:
            # Logs table doesn't exist, skip
            pass

        # Commit changes
        conn.commit()
        conn.close()

        print(f"\n✅ Reset completed successfully!")
        print(f"  - Deleted {deleted_trades} trades")
        if deleted_logs > 0:
            print(f"  - Deleted {deleted_logs} log entries")
        print(f"\nYou can now start the bot fresh with: python main.py")

        logger.info(f"Bot reset completed - {deleted_trades} trades deleted")

    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        logger.error(f"Error resetting bot: {e}")


if __name__ == "__main__":
    reset_trading_bot()
