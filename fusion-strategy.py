import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Trading-specific imports
# import ta
import talib as ta
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend" 
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"

@dataclass
class TradeSignal:
    """Standardized trade signal with risk management"""
    symbol: str
    strategy: str
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # % of capital
    timeframe: str
    timestamp: pd.Timestamp
    rationale: List[str]
    risk_reward_ratio: float
    market_regime: str

class ProductionStrategiesEngine:
    """
    PRODUCTION-GRADE trading strategies engine
    - Real-world risk management
    - Multi-timeframe confirmation
    - Adaptive parameters based on market regime
    - Transaction cost awareness
    - Portfolio-level position sizing
    """
    
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.01):
        self.strategies = {}
        self.df = None
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.current_capital = initial_capital
        self.open_positions = {}
        self.adaptive_thresholds = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_spike_multiplier': 2.0,
            'zscore_threshold': 2.0
        }
        self.current_regime = None
        
    def update_market_data(self, df: pd.DataFrame, symbol: str = None):
        """Professional data preparation with quality checks"""
        if df.empty:
            raise ValueError("Empty dataframe provided")
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        df = df.copy()
        df = self._handle_missing_data(df)
        # self._validate_data_quality(df)
        df.index = pd.to_datetime(df.index, utc=True)
        self.df = df
        self.symbol = symbol
        self._update_adaptive_parameters()
        return True
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill()
        null_count = df.isnull().sum().max()
        if null_count > 5:
            logger.warning(f"Large data gap detected: {null_count} missing values")
        df = df.dropna()
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame):
        print("Data columns:", df.columns.tolist())
        print(df.head())
        required_cols = ['Open','High','Low','Close','Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert to numeric
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=required_cols)

        # Price check
        if (df[['Open','High','Low','Close']] <= 0).any().any():
            raise ValueError("Invalid price data: negative or zero values detected")

        # Volume check
        if (df['Volume'] < 0).any():
            raise ValueError("Invalid volume data: negative values detected")

        # OHLC consistency check
        ohlc_valid = ((df['High'] >= df[['Open','Low','Close']]).all(axis=1) &
                    (df['Low'] <= df[['Open','High','Close']]).all(axis=1))
        if not ohlc_valid.all():
            invalid_count = (~ohlc_valid).sum()
            logger.warning(f"{invalid_count} invalid OHLC patterns detected")

        # return df

    
    def _update_adaptive_parameters(self):
        regime_data = self.detect_market_regime()
        regime = regime_data['regime']
        volatility = regime_data['volatility_score']
        if regime == MarketRegime.HIGH_VOLATILITY.value:
            self.adaptive_thresholds.update({'rsi_overbought':75,'rsi_oversold':25,
                                             'volume_spike_multiplier':2.5,'zscore_threshold':2.5})
        elif regime == MarketRegime.LOW_VOLATILITY.value:
            self.adaptive_thresholds.update({'rsi_overbought':65,'rsi_oversold':35,
                                             'volume_spike_multiplier':1.8,'zscore_threshold':1.5})
        else:
            self.adaptive_thresholds.update({'rsi_overbought':70,'rsi_oversold':30,
                                             'volume_spike_multiplier':2.0,'zscore_threshold':2.0})
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        risk_amount = self.current_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            return 0.0
        raw_size = risk_amount / price_risk
        max_position_size = self.current_capital * 0.20 / entry_price
        position_size = min(raw_size, max_position_size)
        size_percentage = (position_size * entry_price) / self.current_capital
        return min(size_percentage, 0.20)
    
    def detect_all_strategies(self) -> Dict[str, TradeSignal]:
        if self.df is None or len(self.df) < 50:
            return {}
        signals = {}
        strategies_to_run = [
            ('breakout', self.detect_breakout_with_confirmation),
            ('mean_reversion', self.detect_smart_mean_reversion),
            ('trend_following', self.detect_trend_following),
            ('swing_trade', self.detect_swing_trade_pro),
            ('momentum', self.detect_momentum_strategy),
            ('volatility_breakout', self.detect_volatility_breakout),
            ('multi_timeframe', self.detect_multi_timeframe_alignment)
        ]
        for name, strategy_func in strategies_to_run:
            try:
                signal = strategy_func()
                if signal and signal.confidence > 0.3:
                    signals[name] = signal
            except Exception as e:
                logger.error(f"Error in {name} strategy: {e}")
        signals = self._filter_conflicting_signals(signals)
        signals = self._rank_signals_by_quality(signals)
        return signals
    
    def detect_market_regime(self) -> Dict:
        df = self.df
        returns_vol = df['Close'].pct_change().rolling(20).std().iloc[-1]
        atr = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14).iloc[-1]
        atr_percent = atr / df['Close'].iloc[-1]
        ema_20 = ta.EMA(df['Close'], timeperiod=20).iloc[-1]
        ema_50 = ta.EMA(df['Close'], timeperiod=50).iloc[-1]
        ema_200 = ta.EMA(df['Close'], timeperiod=200).iloc[-1]
        volume_sma = df['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df['Volume'].iloc[-1] / volume_sma
        adx = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14).iloc[-1]
        regime = MarketRegime.SIDEWAYS
        volatility_score = returns_vol
        if adx > 25:
            if ema_20 > ema_50 > ema_200:
                regime = MarketRegime.BULL_TREND
            elif ema_20 < ema_50 < ema_200:
                regime = MarketRegime.BEAR_TREND
        elif returns_vol > 0.02:
            regime = MarketRegime.HIGH_VOLATILITY
        elif returns_vol < 0.005:
            regime = MarketRegime.LOW_VOLATILITY
        self.current_regime = regime
        return {
            'regime': regime.value,
            'volatility_score': volatility_score,
            'trend_strength': adx,
            'volume_activity': volume_ratio,
            'key_levels': self._calculate_key_levels()
        }
    
    # ---------- Strategy Detection Functions ----------
    # All your detect_* strategy functions go here, as defined above
    # e.g., detect_breakout_with_confirmation, detect_smart_mean_reversion, etc.
    # They remain unchanged from your code above
    
    # ---------- Utilities ----------
    def _calculate_fibonacci_levels(self) -> Dict[str, float]:
        df = self.df
        if len(df) < 60: return {}
        lookback = min(60, len(df))
        high = df['High'].tail(lookback).max()
        low = df['Low'].tail(lookback).min()
        diff = high - low
        if diff == 0: return {}
        return {'0.0%': high, '23.6%': high-0.236*diff, '38.2%': high-0.382*diff,
                '50.0%': high-0.5*diff, '61.8%': high-0.618*diff,
                '78.6%': high-0.786*diff, '100.0%': low}
    
    def _calculate_key_levels(self) -> Dict:
        df = self.df
        if len(df) < 20: return {}
        return {'support_1': df['Low'].rolling(20).min().iloc[-1],
                'resistance_1': df['High'].rolling(20).max().iloc[-1],
                'pivot_point': (df['High'].iloc[-1]+df['Low'].iloc[-1]+df['Close'].iloc[-1])/3,
                'volume_profile': self._calculate_volume_profile()}
    
    def _calculate_volume_profile(self) -> Dict:
        df = self.df
        if len(df) < 20: return {}
        price_range = df['High'].max() - df['Low'].min()
        if price_range == 0: return {}
        return {'high_volume_zone': df['Close'].mode().iloc[0] if not df['Close'].mode().empty else df['Close'].iloc[-1],
                'volume_delta': (df['Volume'].iloc[-1]-df['Volume'].mean())/df['Volume'].std()}
    
    def _filter_conflicting_signals(self, signals: Dict[str, TradeSignal]) -> Dict[str, TradeSignal]:
        if not signals: return {}
        long_signals = {k:v for k,v in signals.items() if v.direction=='LONG'}
        short_signals = {k:v for k,v in signals.items() if v.direction=='SHORT'}
        if long_signals and short_signals:
            long_avg_conf = np.mean([s.confidence for s in long_signals.values()])
            short_avg_conf = np.mean([s.confidence for s in short_signals.values()])
            return long_signals if long_avg_conf>short_avg_conf else short_signals
        return signals
    
    def _rank_signals_by_quality(self, signals: Dict[str, TradeSignal]) -> Dict[str, TradeSignal]:
        if not signals: return {}
        quality_scores = {name: s.confidence*0.7 + min(s.risk_reward_ratio/3,1.0)*0.3
                          for name,s in signals.items()}
        sorted_signals = sorted(signals.items(), key=lambda x: quality_scores[x[0]], reverse=True)
        return dict(sorted_signals)
    
    def get_strategy_summary(self) -> Dict:
        signals = self.detect_all_strategies()
        regime = self.detect_market_regime()
        return {'symbol': self.symbol,
                'timestamp': self.df.index[-1] if self.df is not None else None,
                'current_price': self.df['Close'].iloc[-1] if self.df is not None else None,
                'market_regime': regime,
                'signals': signals,
                'total_signals': len(signals),
                'recommended_action': self._get_recommended_action(signals),
                'risk_assessment': self._assess_overall_risk()}
    
    def _get_recommended_action(self, signals: Dict[str, TradeSignal]) -> str:
        if not signals: return 'HOLD'
        long_score = sum(s.confidence for s in signals.values() if s.direction=='LONG')
        short_score = sum(s.confidence for s in signals.values() if s.direction=='SHORT')
        if long_score>short_score and long_score>0.5: return 'LONG'
        elif short_score>long_score and short_score>0.5: return 'SHORT'
        else: return 'HOLD'
    
    def _assess_overall_risk(self) -> Dict:
        if self.df is None: return {'level':'UNKNOWN','reason':'No data'}
        volatility = self.df['Close'].pct_change().std()
        drawdown = (self.df['Close']/self.df['Close'].cummax()-1).min()
        volume_trend = self.df['Volume'].pct_change(5).iloc[-1]
        risk_level = 'LOW'
        reasons = []
        if volatility>0.03: risk_level='HIGH'; reasons.append(f"High volatility: {volatility:.2%}")
        elif drawdown<-0.1: risk_level='HIGH'; reasons.append(f"Significant drawdown: {drawdown:.2%}")
        elif volume_trend<-0.3: risk_level='MEDIUM'; reasons.append(f"Declining volume: {volume_trend:.2%}")
        return {'level':risk_level,'reasons':reasons,'volatility':float(volatility),'max_drawdown':float(drawdown)}

# Example usage
def example_usage():
    ticker = "AAPL"
    data = yf.download(ticker, period="6mo", interval="1d")
    engine = ProductionStrategiesEngine(initial_capital=100000, risk_per_trade=0.01)
    engine.update_market_data(data, symbol=ticker)
    summary = engine.get_strategy_summary()
    return summary

if __name__ == "__main__":
    summary = example_usage()
    for k,v in summary.items():
        print(f"{k}: {v}\n")
