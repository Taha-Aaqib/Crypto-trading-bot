"""
Logging utility for the trading bot
Handles console and file logging with rotation
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import yaml


class TradingLogger:
    """Custom logger for trading bot with file rotation"""

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.log_config = self.config['logging']
        self.log_dir = self.log_config['log_dir']

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logger with console and file handlers"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(getattr(logging, self.log_config['level']))

        # Remove existing handlers
        logger.handlers.clear()

        # Format for logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        if self.log_config['console_output']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler with rotation
        if self.log_config['file_output']:
            log_file = os.path.join(
                self.log_dir,
                f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
            )

            max_bytes = self.log_config['max_file_size_mb'] * 1024 * 1024
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=self.log_config['backup_count'],
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self):
        """Return the configured logger instance"""
        return self.logger


# Global logger instance
def get_logger(name='TradingBot'):
    """Get or create a logger instance"""
    if not logging.getLogger(name).handlers:
        trading_logger = TradingLogger()
        return trading_logger.get_logger()
    return logging.getLogger(name)
