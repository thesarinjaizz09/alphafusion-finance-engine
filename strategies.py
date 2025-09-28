class StrategiesEngineer:
    """Compute advanced trading strategies using TA indicators + custom rules"""

    def __init__(self):
        self.strategies = {}

    def detect_all_strategies(self, df: pd.DataFrame, cfg: CLIConfig, target_name: str = 'Close'):
        target_df = df.copy()
        
        if console and not cfg.quiet:
            with Progress(SpinnerOrTickColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                t = prog.add_task("Initiating strategies engineer...", total=None)
                try:
                    self.strategies["breakout"] = self.detect_breakout(target_df, target_col=target_name)
                    # Mean reversion
                    self.strategies["mean_reversion"] = self.detect_mean_reversion(target_df, target_col=target_name)
                    # Fibonacci
                    self.strategies["fibonacci"] = self.detect_fibonacci_pullback(target_df, target_col=target_name)
                    # Price action
                    self.strategies["price_action"] = self.detect_price_action(target_df)
                    # Swing trade
                    self.strategies["swing"] = self.detect_swing_trade(target_df, target_col=target_name)
                    # Scalping helper (advisory)
                    self.strategies["scalping_helper"] = self.detect_scalping_opportunity(target_df, target_col=target_name)
                    # Market regime
                    self.strategies["regime"] = self.detect_market_regime(target_df)
                    # Options summary (light)
                    self.strategies["options_summary"] = self.fetch_options_flow_stub(cfg.ticker) if hasattr(cfg, 'ticker') else {}
                    # News sentiment (stub)
                    self.strategies["news_sentiment"] = self.fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, 'ticker') else None
                    # Trend strength
                    self.strategies["trend_strength"] = self.detect_trend_strength(target_df) if hasattr(cfg, 'ticker') else None
                    # Volume spike
                    self.strategies["volume_spike"] = self.detect_volume_spike(target_df) if hasattr(cfg, 'ticker') else None
                    # MACD strategy
                    self.strategies["macd_strategy"] = self.detect_macd_strategy(target_df) if hasattr(cfg, 'ticker') else None
                    # Bollinger strategy
                    self.strategies["bollinger_strategy"] = self.detect_bollinger_strategy(target_df) if hasattr(cfg, 'ticker') else None
                    # Extra pro indicators
                    self.detect_extra_strategies(target_df)
                    prog.stop_task(t)
                    prog.update(t, description=f"Engineered 30+ strategies.")
                    return self.strategies
                except Exception as e:
                    prog.stop()
                    logger.error(f"Failed to detect strategies: {e}")
                    raise
        else:
            self.strategies["breakout"] = self.detect_breakout(target_df, target_col=target_name)
            # Mean reversion
            self.strategies["mean_reversion"] = self.detect_mean_reversion(target_df, target_col=target_name)
            # Fibonacci
            self.strategies["fibonacci"] = self.detect_fibonacci_pullback(target_df, target_col=target_name)
            # Price action
            self.strategies["price_action"] = self.detect_price_action(target_df)
            # Swing trade
            self.strategies["swing"] = self.detect_swing_trade(target_df, target_col=target_name)
            # Scalping helper (advisory)
            self.strategies["scalping_helper"] = self.detect_scalping_opportunity(target_df, target_col=target_name)
            # Market regime
            self.strategies["regime"] = self.detect_market_regime(target_df)
            # Options summary (light)
            self.strategies["options_summary"] = self.fetch_options_flow_stub(cfg.ticker) if hasattr(cfg, 'ticker') else {}
            # News sentiment (stub)
            self.strategies["news_sentiment"] = self.fetch_news_sentiment_stub(cfg.ticker) if hasattr(cfg, 'ticker') else None
            # Trend strength
            self.strategies["trend_strength"] = self.detect_trend_strength(target_df) if hasattr(cfg, 'ticker') else None
            # Volume spike
            self.strategies["volume_spike"] = self.detect_volume_spike(target_df) if hasattr(cfg, 'ticker') else None
            # MACD strategy
            self.strategies["macd_strategy"] = self.detect_macd_strategy(target_df) if hasattr(cfg, 'ticker') else None
            # Bollinger strategy
            self.strategies["bollinger_strategy"] = self.detect_bollinger_strategy(target_df) if hasattr(cfg, 'ticker') else None
            # Extra pro indicators
            self.detect_extra_strategies(target_df)
            return self.strategies

    # ----------------------------- INDICATORS ----------------------------- #
    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14
        )
        return vwap.vwap

    def _zscore(self, series: pd.Series, window: int = 20):
        m = series.rolling(window=window, min_periods=1).mean()
        s = series.rolling(window=window, min_periods=1).std(ddof=0).replace(0, np.nan)
        return (series - m) / (s + 1e-12)

    def _volume_spike(self, volume: pd.Series, window: int = 20, mult: float = 2.0):
        if len(volume) < window:
            return pd.Series([False] * len(volume), index=volume.index)
        mv = volume.rolling(window=window, min_periods=1).mean()
        return volume > (mv * mult)

    def _fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        lookback = min(len(df), 200)
        s = df["Close"].iloc[-lookback:]
        hi, lo = s.max(), s.min()
        diff = hi - lo
        if diff == 0:
            return {}
        return {
            "0%": hi,
            "23.6%": hi - 0.236 * diff,
            "38.2%": hi - 0.382 * diff,
            "50%": hi - 0.5 * diff,
            "61.8%": hi - 0.618 * diff,
            "100%": lo,
        }

    # -------------------------- CANDLE PATTERNS --------------------------- #
    def _is_bullish_engulfing(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 1 or idx >= len(df):
            return False
        return tlb.CDLENGULFING(df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values)[idx] > 0

    def _is_bearish_engulfing(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 1 or idx >= len(df):
            return False
        return tlb.CDLENGULFING(df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values)[idx] < 0

    def _is_pin_bar(self, df: pd.DataFrame, idx: int, tail_ratio: float = 2.0) -> bool:
        if idx < 0 or idx >= len(df):
            return False
        high, low, open_, close = df["High"].iat[idx], df["Low"].iat[idx], df["Open"].iat[idx], df["Close"].iat[idx]
        body = abs(close - open_)
        tail_top = high - max(open_, close)
        tail_bot = min(open_, close) - low
        tail = max(tail_top, tail_bot)
        if body <= 0:
            return False
        return tail / (body + 1e-12) >= tail_ratio

    # ----------------------------- STRATEGIES ----------------------------- #
    def detect_market_regime(self, df: pd.DataFrame, window: int = 20) -> dict:
        """Detects volatility regime using ATR, returns, volume, and EMA trend."""
        if len(df) < window:
            return {
                "signal": "none",
                "reason": "not enough data",
                "vol_spike": False,
                "confidence": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        atr = tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=window)
        returns_vol = df["Close"].pct_change().rolling(window).std()
        avg_vol = df["Volume"].rolling(window).mean()

        ema20 = tlb.EMA(df["Close"], timeperiod=20)
        ema50 = tlb.EMA(df["Close"], timeperiod=50)
        diff = (ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]

        if abs(diff) < 0.002:   # <0.2% difference → sideways
            trend = "sideways"
        elif ema20.iloc[-1] > ema50.iloc[-1]:
            trend = "bullish"
        else:
            trend = "bearish"

        vol_spike = df["Volume"].iloc[-1] > 1.5 * avg_vol.iloc[-1]

        if returns_vol.iloc[-1] < 0.005 and atr.iloc[-1] < 0.01 * df["Close"].iloc[-1]:
            signal = "low_volatility"
            reason = "Sideways regime / accumulation"
        elif returns_vol.iloc[-1] < 0.02:
            signal = "medium_volatility"
            reason = "Trending regime with controlled volatility"
        else:
            signal = "high_volatility"
            reason = "Explosive volatility regime"

        # Confidence: higher if volatility is low
        confidence = max(0.0, min(1.0, 1 - (returns_vol.iloc[-1] / 0.05)))

        # Stop loss and take profit based on ATR
        stop_loss = df["Close"].iloc[-1] - 1.5 * atr.iloc[-1] if trend == "bullish" else df["Close"].iloc[-1] + 1.5 * atr.iloc[-1]
        take_profit = df["Close"].iloc[-1] + 3 * atr.iloc[-1] if trend == "bullish" else df["Close"].iloc[-1] - 3 * atr.iloc[-1]

        return {
            "volatility": signal,
            "reason": reason,
            "signal": trend,
            "vol_spike": vol_spike,
            "confidence": round(confidence, 2),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit)
        }

    def detect_breakout(self, df: pd.DataFrame, target_col: str = "Close", vol_mult: float = 2.0) -> dict:
        """Detect breakout using Bollinger Bands, Donchian Channels, ATR, volume, and EMA trend (TA-Lib version)."""
        if len(df) < 50:
            return {
                "signal": "none",
                "reason": "not enough data",
                "volume_spike": False,
                "stop_loss": None,
                "take_profit": None,
                "confidence": 0.0
            }

        # Bollinger Bands (20, 2)
        upper_bb, middle_bb, lower_bb = tlb.BBANDS(df[target_col], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # Donchian Channels (20)
        upper_dc = df["High"].rolling(window=20).max()
        lower_dc = df["Low"].rolling(window=20).min()
        middle_dc = (upper_dc + lower_dc) / 2

        # ATR (14)
        atr = tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

        # EMA (20)
        ema20 = tlb.EMA(df[target_col], timeperiod=20)

        # Last values
        last = df[target_col].iloc[-1]
        last_upper_bb = upper_bb.iloc[-1]
        last_lower_bb = lower_bb.iloc[-1]
        last_upper_dc = upper_dc.iloc[-1]
        last_lower_dc = lower_dc.iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_atr = atr.iloc[-1]

        # Volume spike check
        vol_spike = df["Volume"].iloc[-1] > vol_mult * df["Volume"].rolling(20).mean().iloc[-1]

        # Rolling volatility baseline (std dev of returns, 14)
        vol_std = df[target_col].pct_change().rolling(14).std().iloc[-1]

        reasons, signal, confidence = [], "none", 0.0

        # Bullish breakout
        if (
            last > last_upper_bb
            and last > last_upper_dc
            and last > last_ema20
            and last_atr > vol_std
        ):
            signal = "bullish"
            reasons = ["Price > BB upper", "Price > Donchian high", "Above EMA20", "ATR confirms volatility"]
            confidence = min(1.0, (last - last_ema20) / last_atr + (0.2 if vol_spike else 0))

        # Bearish breakout
        elif (
            last < last_lower_bb
            and last < last_lower_dc
            and last < last_ema20
            and last_atr > vol_std
        ):
            signal = "bearish"
            reasons = ["Price < BB lower", "Price < Donchian low", "Below EMA20", "ATR confirms volatility"]
            confidence = min(1.0, (last_ema20 - last) / last_atr + (0.2 if vol_spike else 0))

        # Risk management
        stop_loss = None
        take_profit = None
        if signal == "bullish":
            stop_loss = last - 2 * last_atr
            take_profit = last + 3 * last_atr
        elif signal == "bearish":
            stop_loss = last + 2 * last_atr
            take_profit = last - 3 * last_atr

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "reason": reasons if reasons else "No resistance/support band breakout",
            "volume_spike": vol_spike,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def detect_mean_reversion(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect mean reversion with RSI, Bollinger Bands, Z-score, and volume (TA-Lib version)."""

        # RSI (14)
        rsi = tlb.RSI(df[target_col], timeperiod=14).iloc[-1]

        # Bollinger Bands (20, 2)
        upper_bb, middle_bb, lower_bb = tlb.BBANDS(df[target_col], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # Mean & Std
        mean = df[target_col].rolling(20).mean().iloc[-1]
        std = df[target_col].rolling(20).std().iloc[-1]

        # Last price
        last = df[target_col].iloc[-1]

        # Z-Score (custom)
        z = ((df[target_col] - df[target_col].rolling(20).mean()) / df[target_col].rolling(20).std()).iloc[-1]

        # Volume spike
        vol_spike = df['Volume'].iloc[-1] > 1.5 * df['Volume'].rolling(20).mean().iloc[-1]

        # Last band values
        upper = upper_bb.iloc[-1]
        lower = lower_bb.iloc[-1]

        # Conditions
        reasons, signal, confidence = [], "none", 0.0

        if rsi > 70:
            reasons.append("RSI_overbought")
        elif rsi < 30:
            reasons.append("RSI_oversold")

        if last >= upper:
            reasons.append("BB_upper_touch")
        if last <= lower:
            reasons.append("BB_lower_touch")

        if z > 2:
            reasons.append("Zscore_high")
        elif z < -2:
            reasons.append("Zscore_low")

        if any(x in reasons for x in ["RSI_oversold", "BB_lower_touch", "Zscore_low"]):
            signal = "buy_revert"
            confidence = min(1.0, abs(z) / 3 + 0.2)

        elif any(x in reasons for x in ["RSI_overbought", "BB_upper_touch", "Zscore_high"]):
            signal = "sell_revert"
            confidence = min(1.0, z / 3 + 0.2)

        # Risk management
        stop_loss = last + 2 * std if signal == "buy_revert" else last - 2 * std
        take_profit = mean

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "volume_spike": vol_spike,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def detect_price_action(self, df: pd.DataFrame, idx_offset: int = 0) -> dict:
        """Detect candlestick patterns with volume confirmation."""
        idx = len(df) - 1 - idx_offset
        last = df.iloc[idx]

        vol_spike = df["Volume"].iloc[-1] > 1.5 * df["Volume"].rolling(20).mean().iloc[-1]

        bullish_engulfing = last["Close"] > last["Open"] and df["Close"].iloc[idx - 1] < df["Open"].iloc[idx - 1] and last["Close"] > df["Open"].iloc[idx - 1]
        bearish_engulfing = last["Close"] < last["Open"] and df["Close"].iloc[idx - 1] > df["Open"].iloc[idx - 1] and last["Close"] < df["Open"].iloc[idx - 1]
        pinbar = abs(last["Close"] - last["Open"]) < (last["High"] - last["Low"]) * 0.25
        
        body_size = abs(last['Close'] - last['Open'])
        range_size = last['High'] - last['Low']
        confidence = round(body_size / range_size if range_size != 0 else 0, 2)

        return {"bullish_engulfing": bullish_engulfing, "bearish_engulfing": bearish_engulfing, "pin_bar": pinbar, "volume_confirmation": vol_spike, "confidence": round(confidence, 2)}

    def detect_swing_trade(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect swing trades with EMA cross, MACD, RSI, ATR, and volume, plus metrics."""
        if len(df) < 50:
            return {
                "signal": "none",
                "reasons": [],
                "confidence": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        ema12 = tlb.EMA(df[target_col], timeperiod=12)
        ema26 = tlb.EMA(df[target_col], timeperiod=26)
        macd, macd_signal, macd_hist = tlb.MACD(df[target_col], fastperiod=12, slowperiod=26, signalperiod=9)
        rsi = tlb.RSI(df[target_col], timeperiod=14)
        atr = tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
        avg_vol = df["Volume"].rolling(20).mean()

        last_ema12 = ema12.iloc[-1]
        last_ema26 = ema26.iloc[-1]
        last_macd_hist = macd_hist[-1]
        last_rsi = rsi.iloc[-1]
        last_atr = atr.iloc[-1]
        last_vol = df["Volume"].iloc[-1]
        avg_vol_last = avg_vol.iloc[-1]

        reasons, signal = [], "none"

        # Bullish swing
        if (
            last_ema12 > last_ema26
            and last_macd_hist > 0
            and last_rsi < 70
            and last_atr > 0
            and last_vol > avg_vol_last
        ):
            signal = "bullish"
            reasons = ["EMA cross up", "MACD positive", "RSI < 70", "ATR valid", "Volume confirmed"]

        # Bearish swing
        elif (
            last_ema12 < last_ema26
            and last_macd_hist < 0
            and last_rsi > 30
            and last_atr > 0
            and last_vol > avg_vol_last
        ):
            signal = "bearish"
            reasons = ["EMA cross down", "MACD negative", "RSI > 30", "ATR valid", "Volume confirmed"]

        # Confidence: proportion of conditions met
        conditions = [
            last_ema12 > last_ema26 if signal == "bullish" else last_ema12 < last_ema26,
            last_macd_hist > 0 if signal == "bullish" else last_macd_hist < 0,
            (last_rsi < 70 if signal == "bullish" else last_rsi > 30),
            last_atr > 0,
            last_vol > avg_vol_last
        ]
        confidence = sum(conditions) / len(conditions) if signal != "none" else 0.0

        # Stop loss and take profit
        stop_loss = df[target_col].iloc[-1] - 1.5 * last_atr if signal == "bullish" else (
                    df[target_col].iloc[-1] + 1.5 * last_atr if signal == "bearish" else None)
        take_profit = df[target_col].iloc[-1] + 3 * last_atr if signal == "bullish" else (
                    df[target_col].iloc[-1] - 3 * last_atr if signal == "bearish" else None)

        return {
            "signal": signal,
            "reasons": reasons,
            "confidence": float(confidence),
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "take_profit": float(take_profit) if take_profit is not None else None
        }

    def detect_scalping_opportunity(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect scalping opportunities using VWAP, RSI, StochRSI, and volume, with SL/TP & confidence."""
        if "VWAP" not in df.columns:
            df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
            
        sma5 = tlb.SMA(df['Close'], timeperiod=5)[-1]
        sma20 = tlb.SMA(df['Close'], timeperiod=20)[-1]
        last, vwap_last = df[target_col].iloc[-1], df["VWAP"].iloc[-1]
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        rsi = tlb.RSI(df[target_col], timeperiod=14)[-1]

        # STOCHRSI returns (fastk, fastd), take the last %K
        fastk, fastd = tlb.STOCHRSI(df[target_col], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        stochrsi = fastk[-1]

        # Risk-reward parameters
        risk_pct = 0.003  # 0.3%
        reward_multiple = 2.0

        # --- LONG SETUP ---
        if last > vwap_last and df["Volume"].iloc[-1] > avg_vol and rsi < 70 and stochrsi > 0.5:
            stop_loss = round(last * (1 - risk_pct), 2)
            take_profit = round(last * (1 + risk_pct * reward_multiple), 2)

            # Confidence score (scaled 0–1)
            confidence = 0
            confidence += 0.25 if last > vwap_last else 0
            confidence += 0.25 if df["Volume"].iloc[-1] > avg_vol else 0
            confidence += 0.25 if rsi < 70 else 0
            confidence += 0.25 if stochrsi > 0.5 else 0

            return {
                "signal": "long_momentum",
                "confidence": round(confidence, 2),
                "reason": "VWAP above + volume high + RSI < 70 + StochRSI rising",
                "entry": round(last, 2),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        
        # --- SHORT SETUP ---
        elif last < vwap_last and df["Volume"].iloc[-1] > avg_vol and rsi > 30 and stochrsi < 0.5:
            stop_loss = round(last * (1 + risk_pct), 2)
            take_profit = round(last * (1 - risk_pct * reward_multiple), 2)

            confidence = 0
            confidence += 0.25 if last < vwap_last else 0
            confidence += 0.25 if df["Volume"].iloc[-1] > avg_vol else 0
            confidence += 0.25 if rsi > 30 else 0
            confidence += 0.25 if stochrsi < 0.5 else 0

            return {
                "signal": "short_momentum",
                "confidence": round(confidence, 2),
                "reason": "VWAP below + volume high + RSI > 30 + StochRSI falling",
                "entry": round(last, 2),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

        # --- NO SETUP ---
        return {"signal": "none", "confidence": 0, "reason": None}

    def fetch_news_sentiment_stub(self, ticker: str):
        logger.debug(f"No news API configured; skipping sentiment for {ticker}")
        return None

    def fetch_options_flow_stub(self, ticker: str):
        out = {}
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            out["expiries"] = list(exps) if exps is not None else []
            if out["expiries"]:
                chain = tk.option_chain(out["expiries"][0])
                calls = chain.calls
                puts = chain.puts
                out["calls_oi_sum"] = int(calls["openInterest"].sum()) if "openInterest" in calls else None
                out["puts_oi_sum"] = int(puts["openInterest"].sum()) if "openInterest" in puts else None
        except Exception as e:
            logger.debug(f"Options scan failed (yfinance): {e}")
        return out

        # ---------------- Trend Strength ----------------
    
    def detect_trend_strength(self, df: pd.DataFrame):
        ema10 = tlb.EMA(df['Close'], timeperiod=10).iloc[-1]
        ema50 = tlb.EMA(df['Close'], timeperiod=50).iloc[-1]
        diff = ema10 - ema50

        signal = 'bull_trend' if diff > 0 else 'bear_trend'
        confidence = round(abs(diff)/ema50, 2)

        return {
            'signal': signal,
            'confidence': confidence
        }

    def detect_fibonacci_pullback(self, df: pd.DataFrame, target_col: str = "Close") -> dict:
        """Detect Fibonacci pullbacks with ATR, trend, and EMA confirmation (TA-Lib version)."""

        # High/Low Range
        high, low = df[target_col].max(), df[target_col].min()
        diff = high - low

        # Fibonacci retracement levels
        levels = {
            "0.236": high - 0.236 * diff,
            "0.382": high - 0.382 * diff,
            "0.5": high - 0.5 * diff,
            "0.618": high - 0.618 * diff,
            "0.786": high - 0.786 * diff,
        }

        # Last price
        last = df[target_col].iloc[-1]

        # Nearest Fibonacci level
        nearest = min(levels.items(), key=lambda x: abs(last - x[1]))

        # ATR (14)
        atr = tlb.ATR(df["High"], df["Low"], df["Close"], timeperiod=14).iloc[-1]

        # EMA (50)
        ema50 = tlb.EMA(df[target_col], timeperiod=50).iloc[-1]

        # Trend confirmation
        trend = "bullish" if last > ema50 else "bearish"

        return {
            "levels": levels,
            "near_level": nearest[0],
            "distance": float(abs(last - nearest[1])),
            "atr": atr,
            "trend": trend,
        }

    # ---------------- Volume Spike ----------------
    def detect_volume_spike(self, df):
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        last_vol = df['Volume'].iloc[-1]

        signal = 'none'
        confidence = 0.0
        if last_vol > 1.5 * avg_vol:
            signal = 'volume_spike'
            confidence = round((last_vol - avg_vol)/avg_vol, 2)

        return {
            'signal': signal,
            'confidence': confidence,
            'last_volume': last_vol,
            'avg_volume': avg_vol
        }

    # ---------------- RSI Strategy ----------------
    def detect_rsi_strategy(self, df):
        rsi = tlb.RSI(df['Close'], timeperiod=14).iloc[-1]

        signal = 'none'
        confidence = 0.0
        if rsi > 70:
            signal = 'overbought'
            confidence = round((rsi-70)/30, 2)
        elif rsi < 30:
            signal = 'oversold'
            confidence = round((30-rsi)/30, 2)

        return {'signal': signal, 'rsi': rsi, 'confidence': confidence}

    # ---------------- MACD Strategy ----------------
    def detect_macd_strategy(self, df):
        macd, signal_line, _ = tlb.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        macd_last = macd.iloc[-1]
        signal_last = signal_line.iloc[-1]

        signal = 'none'
        confidence = 0.0
        if macd_last > signal_last:
            signal = 'bull_macd'
            confidence = round((macd_last - signal_last)/abs(signal_last+1e-12), 2)
        elif macd_last < signal_last:
            signal = 'bear_macd'
            confidence = round((signal_last - macd_last)/abs(signal_last+1e-12), 2)

        return {'signal': signal, 'confidence': confidence}

    # ---------------- Bollinger Strategy ----------------
    def detect_bollinger_strategy(self, df):
        bb_upper, bb_middle, bb_lower = tlb.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        last = df['Close'].iloc[-1]

        signal = 'none'
        confidence = 0.0
        if last > bb_upper.iloc[-1]:
            signal = 'sell_bb'
            confidence = round((last - bb_upper.iloc[-1])/bb_upper.iloc[-1], 2)
        elif last < bb_lower.iloc[-1]:
            signal = 'buy_bb'
            confidence = round((bb_lower.iloc[-1] - last)/bb_lower.iloc[-1], 2)

        return {'signal': signal, 'confidence': confidence}

    # -------------------------- PRO ADD-ONS ------------------------------- #
    def detect_extra_strategies(self, df: pd.DataFrame):
        """Calculates all the pro strategies with full multi-condition analysis"""

        # ---------------- TREND STRATEGIES ---------------- #
        try:
            ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"], window1=9, window2=26, window3=52)
            close_last = df["Close"].iat[-1]
            ichi_a = ichi.ichimoku_a().iat[-1]
            ichi_b = ichi.ichimoku_b().iat[-1]
            if close_last > ichi_a and close_last > ichi_b:
                self.strategies["ichimoku"] = {"signal": "bullish", "reason": "above ichimoku cloud"}
            elif close_last < ichi_a and close_last < ichi_b:
                self.strategies["ichimoku"] = {"signal": "bearish", "reason": "below ichimoku cloud"}
            else:
                self.strategies["ichimoku"] = {"signal": "neutral", "reason": "inside cloud"}
        except Exception:
            self.strategies["ichimoku"] = None

        try:
            psar = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"])
            psar_val = psar.psar().iat[-1]
            signal = "bullish" if close_last > psar_val else "bearish"
            self.strategies["psar_trend"] = {"signal": signal, "psar_value": float(psar_val)}
        except Exception:
            self.strategies["psar_trend"] = None

        try:
            kama = ta.momentum.KAMAIndicator(df["Close"]).kama()
            slope = np.sign(kama.diff().iat[-1])
            self.strategies["kama_slope"] = {"signal": "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral",
                                            "value": float(kama.iat[-1])}
        except Exception:
            self.strategies["kama_slope"] = None

        try:
            trix = ta.trend.TRIXIndicator(df["Close"]).trix()
            slope = np.sign(trix.diff().iat[-1])
            self.strategies["trix_momentum"] = {"signal": "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral",
                                                "value": float(trix.iat[-1])}
        except Exception:
            self.strategies["trix_momentum"] = None

        # ---------------- VOLUME STRATEGIES ---------------- #
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
            slope = np.sign(obv.diff().iat[-1])
            self.strategies["obv"] = {"signal": "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral",
                                    "value": float(obv.iat[-1])}
        except Exception:
            self.strategies["obv"] = None

        try:
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"])
            cmf_val = cmf.chaikin_money_flow().iat[-1]
            self.strategies["cmf"] = {"signal": "bullish" if cmf_val > 0 else "bearish", "value": float(cmf_val)}
        except Exception:
            self.strategies["cmf"] = None

        try:
            mfi = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index()
            signal = "bullish" if mfi.iat[-1] > 50 else "bearish"
            self.strategies["mfi"] = {"signal": signal, "value": float(mfi.iat[-1])}
        except Exception:
            self.strategies["mfi"] = None

        try:
            eom = ta.volume.EaseOfMovementIndicator(df["High"], df["Low"], df["Volume"]).ease_of_movement()
            signal = "bullish" if eom.iat[-1] > 0 else "bearish"
            self.strategies["eom"] = {"signal": signal, "value": float(eom.iat[-1])}
        except Exception:
            self.strategies["eom"] = None

        try:
            fi = ta.volume.ForceIndexIndicator(df["Close"], df["Volume"]).force_index()
            signal = "bullish" if fi.iat[-1] > 0 else "bearish"
            self.strategies["force_index"] = {"signal": signal, "value": float(fi.iat[-1])}
        except Exception:
            self.strategies["force_index"] = None

        try:
            adl = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
            slope = np.sign(adl.diff().iat[-1])
            self.strategies["adl"] = {"signal": "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral",
                                    "value": float(adl.iat[-1])}
        except Exception:
            self.strategies["adl"] = None

        try:
            vpt = ta.volume.VolumePriceTrendIndicator(df["Close"], df["Volume"]).volume_price_trend()
            slope = np.sign(vpt.diff().iat[-1])
            self.strategies["vpt"] = {"signal": "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral",
                                    "value": float(vpt.iat[-1])}
        except Exception:
            self.strategies["vpt"] = None

        # ---------------- VOLATILITY STRATEGIES ---------------- #
        try:
            kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"])
            last = df["Close"].iat[-1]
            if last > kc.keltner_channel_hband().iat[-1]:
                signal = "bullish"
            elif last < kc.keltner_channel_lband().iat[-1]:
                signal = "bearish"
            else:
                signal = "neutral"
            self.strategies["keltner"] = {"signal": signal,
                                        "upper": float(kc.keltner_channel_hband().iat[-1]),
                                        "lower": float(kc.keltner_channel_lband().iat[-1]),
                                        "close": float(last)}
        except Exception:
            self.strategies["keltner"] = None

        try:
            donchian = ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"])
            last = df["Close"].iat[-1]
            if last > donchian.donchian_channel_hband().iat[-1]:
                signal = "bullish"
            elif last < donchian.donchian_channel_lband().iat[-1]:
                signal = "bearish"
            else:
                signal = "neutral"
            self.strategies["donchian"] = {"signal": signal,
                                        "upper": float(donchian.donchian_channel_hband().iat[-1]),
                                        "lower": float(donchian.donchian_channel_lband().iat[-1]),
                                        "close": float(last)}
        except Exception:
            self.strategies["donchian"] = None

        # ---------------- MOMENTUM STRATEGIES ---------------- #
        momentum_indicators = {
            "stoch_rsi": ta.momentum.StochRSIIndicator(df["Close"]).stochrsi,
            "ultimate_osc": ta.momentum.UltimateOscillator(df["High"], df["Low"], df["Close"]).ultimate_oscillator,
            "awesome_osc": ta.momentum.AwesomeOscillatorIndicator(df["High"], df["Low"]).awesome_oscillator,
            "tsi": ta.momentum.TSIIndicator(df["Close"]).tsi,
            "cci": ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci,
            "williams_r": ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r,
            "roc": ta.momentum.ROCIndicator(df["Close"]).roc,
            "adx": ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx,
            "adx_pos": ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx_pos,
            "adx_neg": ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx_neg,
        }

        for key, func in momentum_indicators.items():
            try:
                val = func().iat[-1]
                self.strategies[key] = {"value": float(val),
                                        "signal": ("bullish" if val > 0 else "bearish" if val < 0 else "neutral")}
            except Exception:
                self.strategies[key] = None

        # ---------------- SUPER TREND ---------------- #
        try:
            atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            factor = 3.0
            hl2 = (df["High"] + df["Low"]) / 2
            supertrend = hl2 - (factor * atr)
            last = df["Close"].iat[-1]
            self.strategies["supertrend"] = {"signal": "bullish" if last > supertrend.iat[-1] else "bearish",
                                            "value": float(supertrend.iat[-1]),
                                            "close": float(last)}
        except Exception:
            self.strategies["supertrend"] = None

        # ---------------- HULL MA ---------------- #
        try:
            def hull_moving_average(series, window=14):
                half_length = int(window / 2)
                sqrt_length = int(np.sqrt(window))
                wma = series.rolling(window, min_periods=1).mean()
                wma_half = series.rolling(half_length, min_periods=1).mean()
                raw_hma = (2 * wma_half - wma).rolling(sqrt_length, min_periods=1).mean()
                return raw_hma

            hma = hull_moving_average(df["Close"], 20)
            slope = np.sign(hma.diff().iat[-1])
            self.strategies["hma"] = {"signal": "bullish" if slope > 0 else "bearish" if slope < 0 else "neutral",
                                    "value": float(hma.iat[-1])}
        except Exception:
            self.strategies["hma"] = None

