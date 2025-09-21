# AlphaFusion Changelog

All notable changes to **AlphaFusion** are documented in this file.  
The project follows [Semantic Versioning](https://semver.org/).

---

## [v1.2.0-beta] - YYYY-MM-DD
**Status:** Beta (Under Development)

**Features (Current / Active):**
- Computes **80+ technical indicators** from OHLCV data.
- Computes **30+ trading strategies** (breakout, mean reversion, swing, scalping, volume, price action, regime detection, etc.).
- Supports **stocks, ETFs, and cryptocurrencies** across multiple timeframes.
- Prints weighted strategy signals with bullish/bearish confidence scores and reasons.
- Outputs JSON per candle with all indicators, strategies, and signal summaries.
- Console output enhanced with Rich tables and panels for easy reading.
- Designed for **live analysis**, ML model integration (SCSO-LSTM, HMM, XGBoost, RandomForest, etc.) will be added in future updates.
- Multi-threading / parallel processing for strategy calculation and future model training.

**Notes:**
- Still under active development.
- ML/forecasting models from previous versions are not yet integrated.
- Focused on indicator and strategy engine for robust signal calculation.

---

## [v1.2.0] - YYYY-MM-DD
**Status:** Planned Release  

**Features / Improvements:**
- Removed all forecasting models except **SCSO-LSTM**.
- Optimized codebase for speed and memory.
- Computes **80+ indicators** for stocks and crypto.
- Production-ready **SCSO-LSTM forecasting engine** with feature engineering.
- Maintains all previously supported strategies.
- Supports multi-timeframe analysis and JSON/CSV export.

---

## [v1.1.0] - YYYY-MM-DD
**Optimized Release**

**Features:**
- Unified forecasting system:
  - Supports **multiple models**: LSTM, BiLSTM, PSO-LSTM, PSO-BiLSTM, TFT, SARIMAX, Prophet, XGBoost.
  - Patched datetime handling for tz-aware → tz-naive conversions (StatsModels & Prophet warnings fixed).
- Strategy signals engine:
  - Breakout, mean reversion, fibonacci, price action, swing, scalping, pairs trading.
  - Volume-based strategies (OBV, ADL, CMF, MFI, VPT, Force Index, EOM).
  - Momentum oscillators (RSI, StochRSI, MACD, TRIX, Ultimate Oscillator, TSI, Williams %R, ROC, CCI, Awesome Oscillator).
  - Trend indicators (ADX, Ichimoku, SuperTrend, PSAR, HMA slope, KAMA slope).
- Neural signal fusion (ensemble of strategies with weighted scoring).
- Confidence scoring: bullish/bearish, confidence %, and reason logging.
- Multi-model forecasting pipeline with **parallel execution via ThreadPoolExecutor**.
- Automatic feature engineering (OHLCV + technical indicators).
- PSO/SCSO hyperparameter optimization hooks.
- Logging:
  - Structured console + file logging.
  - Rich-powered live epoch logging for LSTM/BiLSTM.
  - JSON exports of per-candle analysis.
- CLI usage for predictions, strategies, and ensemble forecasts.
- Lazy imports for heavy dependencies (TensorFlow, Prophet, XGBoost).

---

## [v1.0.0] - YYYY-MM-DD
**Base Version - Single-file Production-Grade AI Trading CLI**

**Features:**
- Core LSTM and BiLSTM forecasting.
- Strategy signals: breakout, mean reversion, fibonacci, price action, swing, scalping.
- Multi-indicator analysis (RSI, MACD, ADX, OBV, etc.).
- Confidence scoring for bullish/bearish signals.
- JSON output for candle-level analysis.
- CLI usage:
  - Predict, analyze strategies, forecast.
- Designed as a **production-ready single-file CLI**.

---
