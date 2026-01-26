"""Check ML model label distribution"""
from src.models.trading_model import TradingModel
from src.data.data_fetcher import DataFetcher
import yaml
from datetime import datetime

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Get November data
fetcher = DataFetcher(config)
start_date = datetime(2025, 11, 1)
end_date = datetime(2025, 11, 30)
df = fetcher.fetch_historical_data('BTC/USDT', '15m', start_date, end_date)

# Create labels
model = TradingModel(config, symbol='BTC/USDT')
labels = model.create_labels(df, lookahead=5)

print('Label distribution in Nov 2025:')
print(labels.value_counts())
total = len(labels)
buy = (labels == 1).sum()
sell = (labels == -1).sum()
hold = (labels == 0).sum()
print(f'\nTotal: {total}')
print(f'BUY signals:  {buy} ({buy/total*100:.1f}%)')
print(f'SELL signals: {sell} ({sell/total*100:.1f}%)')
print(f'HOLD signals: {hold} ({hold/total*100:.1f}%)')
