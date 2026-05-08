"""
LSTM Trading Model — Setup Quality Predictor
Predicts P(trade setup succeeds) = probability that a trade hits TP before SL.

Instead of raw price direction (coin flip ~50%), this model learns what makes
a GOOD trade setup by training on simulated trade outcomes from historical data.

Architecture:
    Input (sequence_length, features) -> LSTM layers -> Attention -> Dense -> Output (1 logit)

Features include:
    - Market microstructure (returns, volatility, volume)
    - SMC context (OTE zone, order blocks, CHoCH/BOS recency, sweeps)
    - Technical state (RSI, divergence, EMA alignment, BB position)

Usage:
    model = LSTMTradingModel(config, symbol='BTC/USDT')
    results = model.train(df_with_indicators)
    p_success, confidence = model.predict(df_with_indicators)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils.logger import get_logger

logger = get_logger()

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"LSTM using device: {DEVICE}")


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences (regression)"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)  # float for regression
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMNetwork(nn.Module):
    """
    LSTM Neural Network for setup quality prediction (binary)
    
    Architecture:
        - 2 LSTM layers with dropout
        - Attention over sequence
        - Dense layers with ReLU
        - Single output logit: P(setup hits TP before SL)
    """
    
    def __init__(
        self, 
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism (simple)
        self.attention = nn.Linear(hidden_size, 1)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)  # raw logit for BCEWithLogitsLoss
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention over sequence
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dense layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc_out(out)  # raw logit — sigmoid applied by loss or at inference
        
        return out.squeeze(-1)  # shape: (batch,)


class LSTMTradingModel:
    """
    LSTM-based setup quality predictor
    
    Predicts P(setup succeeds) — the probability a trade entry will hit
    its take-profit before its stop-loss.
    
    Handles:
        - Feature engineering (temporal + SMC setup context)
        - Trade outcome simulation for labeling
        - Sequence creation and training
        - Prediction with confidence scoring
    """
    
    def __init__(self, config: Dict, symbol: str = None):
        self.config = config
        self.symbol = symbol
        self.model = None
        self.scaler = StandardScaler()
        
        # Model hyperparameters
        ml_config = config.get('ml_model', {})
        self.sequence_length = ml_config.get('sequence_length', 60)
        self.hidden_size = ml_config.get('lstm_hidden_size', 128)
        self.num_layers = ml_config.get('lstm_num_layers', 2)
        self.dropout = ml_config.get('lstm_dropout', 0.2)
        self.learning_rate = ml_config.get('lstm_learning_rate', 0.001)
        self.batch_size = ml_config.get('lstm_batch_size', 64)
        self.epochs = ml_config.get('lstm_epochs', 100)
        self.patience = ml_config.get('lstm_patience', 10)
        
        # Risk parameters for trade simulation
        risk_config = config.get('risk', {})
        self.rr_ratio = risk_config.get('take_profit_rr_ratio', 2.0)
        self.sl_atr_mult = risk_config.get('stop_loss_atr_multiplier', 1.5)
        self.sl_pct_fallback = risk_config.get('stop_loss_percent', 0.02)
        
        # Model paths
        if symbol:
            symbol_clean = symbol.replace('/', '_')
            model_dir = f"models/saved_models/{symbol_clean}"
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = f"{model_dir}/lstm_model.pt"
            self.scaler_path = f"{model_dir}/lstm_scaler.pkl"
        else:
            os.makedirs("models/saved_models", exist_ok=True)
            self.model_path = "models/saved_models/lstm_model.pt"
            self.scaler_path = "models/saved_models/lstm_scaler.pkl"
        
        self.is_trained = False
        
        # Feature list: temporal (18) + SMC context (10) = 28 features
        self.feature_names = [
            # Multi-period returns (momentum)
            'return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_24h',
            # Price position
            'price_zscore', 'bb_position', 'dist_ema50', 'dist_ema200', 'ema_spread',
            # Oscillators
            'rsi', 'rsi_change', 'macd_norm', 'macd_hist',
            # Volume
            'volume_ratio', 'volume_trend',
            # Volatility
            'atr_pct', 'volatility_regime',
            # === SMC Setup Context ===
            'in_ote_zone',          # price in OTE (0.618-0.786) zone
            'in_discount',          # price below equilibrium
            'ob_nearby',            # order block within 1% of price
            'choch_recency',        # how recent is last CHoCH (0=old, 1=just happened)
            'bos_recency',          # how recent is last BOS
            'fvg_active',           # fresh FVG near price
            'sweep_recent',         # liquidity sweep in last 10 candles
            'rsi_divergence',       # RSI divergence active (-1=bearish, 0=none, 1=bullish)
            'market_structure',     # {1: bullish, 0: neutral, -1: bearish}
            'body_ratio',           # candle body / range
        ]
        
        logger.info(f"LSTM model initialized for {symbol or 'default'}")
        
        # Try to load existing model
        self.load_model()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal + SMC setup context features.
        
        Temporal features capture market state (momentum, volatility, volume).
        SMC features capture setup quality (OTE, OB, CHoCH recency, divergence).
        """
        features = pd.DataFrame(index=df.index)
        
        # === Multi-period returns (momentum) ===
        for period in [1, 3, 6, 12, 24]:
            features[f'return_{period}h'] = df['close'].pct_change(period)
        
        # === Price position features ===
        roll_mean = df['close'].rolling(20).mean()
        roll_std = df['close'].rolling(20).std()
        features['price_zscore'] = (df['close'] - roll_mean) / (roll_std + 1e-8)
        
        bb_upper = roll_mean + 2 * roll_std
        bb_lower = roll_mean - 2 * roll_std
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        if 'ema_50' in df.columns:
            features['dist_ema50'] = (df['close'] - df['ema_50']) / df['close']
        else:
            ema50 = df['close'].ewm(span=50).mean()
            features['dist_ema50'] = (df['close'] - ema50) / df['close']
        
        if 'ema_200' in df.columns:
            features['dist_ema200'] = (df['close'] - df['ema_200']) / df['close']
        else:
            ema200 = df['close'].ewm(span=200).mean()
            features['dist_ema200'] = (df['close'] - ema200) / df['close']
        
        features['ema_spread'] = features['dist_ema50'] - features['dist_ema200']
        
        # === Oscillators ===
        features['rsi'] = (df['rsi'] / 100 - 0.5) * 2 if 'rsi' in df.columns else 0
        features['rsi_change'] = features['rsi'].diff(3) if isinstance(features['rsi'], pd.Series) else 0
        
        if 'macd' in df.columns:
            features['macd_norm'] = df['macd'] / (df['close'] + 1e-8)
            features['macd_hist'] = (df['macd'] - df.get('macd_signal', df['macd'])) / (df['close'] + 1e-8)
        else:
            # Compute MACD on the fly
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9).mean()
            features['macd_norm'] = macd_line / (df['close'] + 1e-8)
            features['macd_hist'] = (macd_line - macd_signal) / (df['close'] + 1e-8)
        
        # === Volume ===
        vol_mean = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / (vol_mean + 1e-8)
        features['volume_trend'] = (
            df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-8)
        )
        
        # === Volatility ===
        features['atr_pct'] = (df['atr'] / df['close']) if 'atr' in df.columns else df['close'].pct_change().abs().rolling(14).mean()
        features['volatility_regime'] = (
            df['close'].pct_change().rolling(10).std() / 
            (df['close'].pct_change().rolling(50).std() + 1e-8)
        )
        
        # === SMC Setup Context Features ===
        # OTE zone: price in 0.618-0.786 fib retracement
        features['in_ote_zone'] = 0.0
        if 'in_ote_long' in df.columns and 'in_ote_short' in df.columns:
            features['in_ote_zone'] = (df['in_ote_long'].astype(float) + df['in_ote_short'].astype(float)).clip(0, 1)
        
        # Discount/premium
        features['in_discount'] = df['in_discount'].astype(float) if 'in_discount' in df.columns else 0.0
        
        # Order block proximity: is there an OB within 1% of current price?
        features['ob_nearby'] = 0.0
        if 'ob_bullish_high' in df.columns and 'ob_bearish_low' in df.columns:
            ob_bull_dist = abs(df['close'] - df['ob_bullish_high'].ffill()) / (df['close'] + 1e-8)
            ob_bear_dist = abs(df['close'] - df['ob_bearish_low'].ffill()) / (df['close'] + 1e-8)
            features['ob_nearby'] = ((ob_bull_dist < 0.01) | (ob_bear_dist < 0.01)).astype(float)
        
        # CHoCH recency: rolling window — 1.0 if CHoCH in last 5 candles, decays to 0
        features['choch_recency'] = 0.0
        if 'choch_bullish' in df.columns and 'choch_bearish' in df.columns:
            choch_any = df['choch_bullish'].astype(float) + df['choch_bearish'].astype(float)
            features['choch_recency'] = choch_any.rolling(10, min_periods=1).max()
        
        # BOS recency
        features['bos_recency'] = 0.0
        if 'bos_bullish' in df.columns and 'bos_bearish' in df.columns:
            bos_any = df['bos_bullish'].astype(float) + df['bos_bearish'].astype(float)
            features['bos_recency'] = bos_any.rolling(10, min_periods=1).max()
        
        # Fresh FVG near price
        features['fvg_active'] = 0.0
        if 'fvg_fresh' in df.columns:
            features['fvg_active'] = df['fvg_fresh'].astype(float)
        elif 'fvg_bullish' in df.columns and 'fvg_bearish' in df.columns:
            fvg_any = df['fvg_bullish'].astype(float) + df['fvg_bearish'].astype(float)
            features['fvg_active'] = fvg_any.rolling(5, min_periods=1).max()
        
        # Recent liquidity sweep
        features['sweep_recent'] = 0.0
        if 'sweep_bullish' in df.columns and 'sweep_bearish' in df.columns:
            sweep_any = df['sweep_bullish'].astype(float) + df['sweep_bearish'].astype(float)
            features['sweep_recent'] = sweep_any.rolling(10, min_periods=1).max()
        
        # RSI divergence: -1 bearish, 0 none, +1 bullish
        features['rsi_divergence'] = 0.0
        if 'bullish_divergence' in df.columns:
            features['rsi_divergence'] += df['bullish_divergence'].astype(float)
        if 'bearish_divergence' in df.columns:
            features['rsi_divergence'] -= df['bearish_divergence'].astype(float)
        
        # Market structure
        structure_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        if 'market_structure' in df.columns:
            features['market_structure'] = df['market_structure'].map(structure_map).fillna(0)
        else:
            features['market_structure'] = 0
        
        # Candle body ratio
        features['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        
        # Fill NaN and infinite values
        features = features.fillna(0)
        features = features.replace([float('inf'), float('-inf')], 0)
        
        return features
    
    def create_labels(self, df: pd.DataFrame, lookahead: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Create setup-success labels by simulating trades on historical data.
        
        For each candle, simulates both a LONG and SHORT trade:
        - SL = ATR * multiplier away from entry
        - TP = SL distance * R:R ratio
        - Checks future candles to see if TP or SL is hit first
        
        The label indicates whether the dominant direction trade succeeds.
        Also returns a mask of valid setup candles (where a clear setup exists).
        
        Returns:
            labels: Series of 1.0 (TP hit) or 0.0 (SL hit) per candle
            valid_mask: Series of True/False — only train on candles with clear setups
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Compute ATR for SL sizing
        if 'atr' in df.columns:
            atr = df['atr'].values
        else:
            # Manual ATR calculation
            tr = np.maximum(
                high - low,
                np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1)))
            )
            atr = pd.Series(tr).rolling(14).mean().values
        
        n = len(df)
        max_hold = max(lookahead * 4, 20)  # max candles to wait for outcome
        labels = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        
        # Determine dominant direction from market structure / momentum
        # Positive momentum → simulate long, negative → simulate short
        returns_5 = pd.Series(close).pct_change(5).values
        
        for i in range(max(50, self.sequence_length), n - max_hold):
            entry = close[i]
            current_atr = atr[i]
            
            if np.isnan(current_atr) or current_atr <= 0 or entry <= 0:
                continue
            
            sl_dist = current_atr * self.sl_atr_mult
            
            # Ensure minimum SL distance (0.3% of price)
            min_sl = entry * 0.003
            sl_dist = max(sl_dist, min_sl)
            
            tp_dist = sl_dist * self.rr_ratio
            
            # Determine direction from local context
            r5 = returns_5[i] if not np.isnan(returns_5[i]) else 0
            
            # Use market structure if available
            is_long = r5 > 0  # Default: trade with momentum
            
            if is_long:
                sl_price = entry - sl_dist
                tp_price = entry + tp_dist
            else:
                sl_price = entry + sl_dist
                tp_price = entry - tp_dist
            
            # Simulate: check future candles for SL or TP hit
            outcome = None
            for j in range(1, max_hold + 1):
                idx = i + j
                if idx >= n:
                    break
                
                if is_long:
                    if low[idx] <= sl_price:
                        outcome = 0.0  # SL hit
                        break
                    if high[idx] >= tp_price:
                        outcome = 1.0  # TP hit
                        break
                else:
                    if high[idx] >= sl_price:
                        outcome = 0.0  # SL hit
                        break
                    if low[idx] <= tp_price:
                        outcome = 1.0  # TP hit
                        break
            
            if outcome is not None:
                labels[i] = outcome
                valid[i] = True
        
        labels_series = pd.Series(labels, index=df.index)
        valid_series = pd.Series(valid, index=df.index)
        
        tp_count = int(np.nansum(labels == 1.0))
        sl_count = int(np.nansum(labels == 0.0))
        total = tp_count + sl_count
        
        if total > 0:
            logger.info(
                f"Setup labels: {total} valid setups, "
                f"{tp_count} TP hits ({tp_count/total*100:.1f}%), "
                f"{sl_count} SL hits ({sl_count/total*100:.1f}%)"
            )
        
        return labels_series, valid_series
    
    def create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            X: Feature array (samples, features)
            y: Label array (samples,)
            
        Returns:
            X_seq: (samples - seq_len, seq_len, features)
            y_seq: (samples - seq_len,)
        """
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i - self.sequence_length:i])
            labels.append(y[i])
        
        return np.array(sequences), np.array(labels)
    
    def train(
        self, 
        df: pd.DataFrame, 
        lookahead: int = 5,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the LSTM model on setup-success labels.
        
        Args:
            df: DataFrame with OHLCV and indicators (SMC + TA)
            lookahead: Base periods for trade outcome simulation
            validation_split: Fraction for validation
            
        Returns:
            Training results dict
        """
        logger.info(f"Starting LSTM training for {self.symbol}...")
        
        # Prepare features and labels
        X_df = self.prepare_features(df)
        labels, valid_mask = self.create_labels(df, lookahead)
        
        # Only keep rows with valid setup outcomes
        valid_idx = valid_mask & ~X_df.isna().any(axis=1) & ~labels.isna()
        X_df = X_df[valid_idx]
        labels = labels[valid_idx]
        
        if len(X_df) < self.sequence_length + 100:
            logger.error(f"Insufficient valid setups for training ({len(X_df)} samples)")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Convert to numpy
        X = X_df.values
        y = labels.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        logger.info(f"Created {len(X_seq)} sequences of length {self.sequence_length}")
        
        # Train/validation split (no shuffle - time series!)
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_seq.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(DEVICE)
        
        # Loss — binary cross entropy with class balancing
        # Dataset is imbalanced (fewer TP hits), so upweight the positive class
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        raw_weight = n_neg / max(n_pos, 1)
        capped_weight = min(raw_weight, 1.5)  # Cap to avoid over-correction
        pos_weight = torch.tensor([capped_weight], dtype=torch.float32).to(DEVICE)
        logger.info(f"Class balance: pos_weight={capped_weight:.2f} (raw={raw_weight:.2f}, {n_pos:.0f} TP vs {n_neg:.0f} SL)")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Log label balance
        tp_pct = y_train.mean() * 100
        logger.info(f"Label balance: {tp_pct:.1f}% TP hits, {100-tp_pct:.1f}% SL hits")
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Setup success prediction accuracy
            preds_arr = np.array(all_preds)
            targets_arr = np.array(all_targets)
            pred_success = (preds_arr > 0).astype(float)
            accuracy = np.mean(pred_success == targets_arr)
            
            # Mean confidence (distance from 0.5)
            probs = 1.0 / (1.0 + np.exp(-preds_arr.clip(-10, 10)))
            mean_confidence = np.mean(np.abs(probs - 0.5))
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Success Pred Acc: {accuracy:.3f}, "
                    f"Conf: {mean_confidence:.3f}"
                )
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.load_model()
        
        # Final evaluation
        self.model.eval()
        
        def eval_setup_success(loader):
            preds, targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(DEVICE)
                    outputs = self.model(X_batch)
                    preds.extend(outputs.cpu().numpy())
                    targets.extend(y_batch.numpy())
            preds, targets = np.array(preds), np.array(targets)
            pred_success = (preds > 0).astype(float)
            accuracy = np.mean(pred_success == targets)
            probs = 1.0 / (1.0 + np.exp(-preds.clip(-10, 10)))
            mean_conf = np.mean(np.abs(probs - 0.5))
            return accuracy, mean_conf
        
        train_acc, train_conf = eval_setup_success(train_loader)
        test_acc, test_conf = eval_setup_success(val_loader)
        
        self.is_trained = True
        
        logger.info(f"LSTM training complete!")
        logger.info(f"Train setup-success accuracy: {train_acc:.3f}, avg confidence: {train_conf:.3f}")
        logger.info(f"Test  setup-success accuracy: {test_acc:.3f}, avg confidence: {test_conf:.3f}")
        
        return {
            'success': True,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'samples_trained': len(X_train),
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict setup quality for current market state.
        
        Returns:
            p_success: probability that a trade setup succeeds (0.0 to 1.0)
                        > 0.55 = good setup, < 0.45 = bad setup
            confidence: how certain the model is (0.0 = uncertain, 0.5 = very certain)
        """
        if not self.is_trained or self.model is None:
            logger.warning(f"LSTM model for {self.symbol} not trained")
            return 0.5, 0.0  # neutral: 50% success, 0 confidence
        
        try:
            # Prepare features
            X_df = self.prepare_features(df)
            
            if len(X_df) < self.sequence_length:
                logger.warning(f"Not enough data for prediction (need {self.sequence_length})")
                return 0.5, 0.0
            
            # Get last sequence
            X = X_df.iloc[-self.sequence_length:].values
            X = X.reshape(1, -1)  # Flatten for scaler
            X_scaled = self.scaler.transform(X.reshape(-1, X_df.shape[1]))
            X_scaled = X_scaled.reshape(1, self.sequence_length, -1)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logit = self.model(X_tensor).cpu().item()
            
            # Sigmoid → P(setup succeeds)
            p_success = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
            
            # Confidence: distance from 0.5 (ranges 0 to 0.5)
            confidence = abs(p_success - 0.5)
            
            # Score for ensemble: maps P(success) to quality metric
            # 0.5 = neutral, 0.7 = good, 0.3 = bad
            quality_label = "GOOD" if p_success > 0.55 else ("BAD" if p_success < 0.45 else "NEUTRAL")
            
            logger.info(
                f"LSTM {self.symbol}: P(success)={p_success:.3f} logit={logit:+.3f} "
                f"→ {quality_label}, confidence={confidence:.3f}"
            )
            
            return p_success, confidence
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.5, 0.0
    
    def save_model(self):
        """Save model and scaler"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            }, self.model_path)
            
            # Save scaler
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info(f"LSTM model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
    
    def load_model(self):
        """Load saved model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                # Load scaler
                self.scaler = joblib.load(self.scaler_path)
                
                # Load model
                checkpoint = torch.load(self.model_path, map_location=DEVICE)
                
                self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
                
                self.model = LSTMNetwork(
                    input_size=checkpoint['input_size'],
                    hidden_size=checkpoint['hidden_size'],
                    num_layers=checkpoint['num_layers'],
                    dropout=checkpoint['dropout']
                ).to(DEVICE)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.is_trained = True
                logger.info(f"✅ LSTM model loaded from {self.model_path}")
            else:
                logger.info(f"No saved LSTM model found at {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            self.is_trained = False
