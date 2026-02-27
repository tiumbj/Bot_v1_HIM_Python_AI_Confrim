"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 2.10.5

Changelog:
- 2.10.5 (2026-02-27):
  - FEATURE: proximity_score_min gate (quality filter) for sideway_scalp
      - Read from config: sideway_scalp.proximity_score_min (default 0.70)
      - Applied mainly to SOFT-ZONE attempts (adaptive trigger). Hard-touch stays allowed.
      - If in soft-zone but score < min => blocked_by includes "low_proximity_score"
  - KEEP: adaptive trigger knobs (near_trigger_atr, allow_soft_trigger, rsi_soft_band)
  - KEEP: sideway_scalp NEUTRAL (bias_source="SIDEWAY_MODE", direction_bias="NEUTRAL")
  - KEEP: regime gate = ADX low + BB width normalized by ATR
  - KEEP: debug bag from v2.10.4 to diagnose rr_too_low/tick anomalies (only when attempting trade)

Notes:
- This version is intended to reduce false signals on M5 by requiring "close enough" proximity.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import MetaTrader5 as mt5

from config_resolver import resolve_effective_config


ENGINE_VERSION = "2.10.5"


class TradingEngine:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.cfg = self.load_config()

    @staticmethod
    def safe_float(v: Any, default: float = 0.0) -> float:
        if v is None:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def safe_int(v: Any, default: int = 0) -> int:
        if v is None:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return resolve_effective_config(raw)
        except Exception:
            return {}

    def reload_config(self) -> Dict[str, Any]:
        self.cfg = self.load_config()
        return self.cfg

    @staticmethod
    def ensure_mt5() -> None:
        if mt5.initialize():
            return
        time.sleep(0.2)
        if not mt5.initialize():
            raise RuntimeError("MT5 initialize failed")

    @staticmethod
    def get_tick(symbol: str):
        TradingEngine.ensure_mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"no tick for symbol={symbol}")
        return t

    @staticmethod
    def tf(tf_str: str) -> int:
        t = (tf_str or "").strip().upper()
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M2": mt5.TIMEFRAME_M2,
            "M3": mt5.TIMEFRAME_M3,
            "M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5,
            "M6": mt5.TIMEFRAME_M6,
            "M10": mt5.TIMEFRAME_M10,
            "M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15,
            "M20": mt5.TIMEFRAME_M20,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H2": mt5.TIMEFRAME_H2,
            "H3": mt5.TIMEFRAME_H3,
            "H4": mt5.TIMEFRAME_H4,
            "H6": mt5.TIMEFRAME_H6,
            "H8": mt5.TIMEFRAME_H8,
            "H12": mt5.TIMEFRAME_H12,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        if t not in mapping:
            raise ValueError(f"unknown timeframe: {tf_str}")
        return mapping[t]

    def get_data(self, symbol: str, timeframe: int, n: int) -> Dict[str, np.ndarray]:
        TradingEngine.ensure_mt5()
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(n))
        if rates is None or len(rates) < 50:
            raise RuntimeError(f"no rates (symbol={symbol}, timeframe={timeframe})")
        return {
            "time": np.asarray(rates["time"], dtype=np.int64),
            "open": np.asarray(rates["open"], dtype=float),
            "high": np.asarray(rates["high"], dtype=float),
            "low": np.asarray(rates["low"], dtype=float),
            "close": np.asarray(rates["close"], dtype=float),
            "tick_volume": np.asarray(rates["tick_volume"], dtype=np.int64),
            "spread": np.asarray(rates["spread"], dtype=np.int64),
        }

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n < period + 1 or period <= 0:
            return out

        tr = np.maximum.reduce(
            [
                h[1:] - l[1:],
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ]
        )
        out[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i - 1]) / period
        return out

    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, multiplier: float = 3.0):
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)

        atr = TradingEngine.atr(h, l, c, period)
        hl2 = (h + l) / 2.0
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        n = len(c)
        st = np.full(n, np.nan, dtype=float)
        direction = np.full(n, 0, dtype=int)

        for i in range(1, n):
            if np.isnan(atr[i]) or np.isnan(upperband[i]) or np.isnan(lowerband[i]):
                continue

            if np.isnan(st[i - 1]):
                st[i] = lowerband[i]
                direction[i] = 1
                continue

            prev_st = st[i - 1]
            prev_dir = direction[i - 1] if direction[i - 1] != 0 else 1

            fub = upperband[i] if (upperband[i] < upperband[i - 1] or c[i - 1] > upperband[i - 1]) else upperband[i - 1]
            flb = lowerband[i] if (lowerband[i] > lowerband[i - 1] or c[i - 1] < lowerband[i - 1]) else lowerband[i - 1]

            if prev_dir == 1:
                if c[i] <= fub:
                    direction[i] = -1
                    st[i] = fub
                else:
                    direction[i] = 1
                    st[i] = flb
            else:
                if c[i] >= flb:
                    direction[i] = 1
                    st[i] = flb
                else:
                    direction[i] = -1
                    st[i] = fub

            if np.isnan(st[i]):
                st[i] = prev_st

        direction = np.where(direction == 0, 1, direction)
        direction = np.where(direction > 0, 1, -1)
        return st, direction

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        c = np.asarray(close, dtype=float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n < period + 1 or period <= 0:
            return out

        delta = np.diff(c)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.full(n, np.nan, dtype=float)
        avg_loss = np.full(n, np.nan, dtype=float)

        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])

        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
        out = 100.0 - (100.0 / (1.0 + rs))
        return out

    @staticmethod
    def bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.asarray(close, dtype=float)
        n = len(c)
        upper = np.full(n, np.nan, dtype=float)
        mid = np.full(n, np.nan, dtype=float)
        lower = np.full(n, np.nan, dtype=float)
        if n < period or period <= 1:
            return upper, mid, lower

        for i in range(period - 1, n):
            window = c[i - period + 1 : i + 1]
            mu = float(np.mean(window))
            sd = float(np.std(window, ddof=0))
            mid[i] = mu
            upper[i] = mu + sd * float(std_mult)
            lower[i] = mu - sd * float(std_mult)
        return upper, mid, lower

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n < period + 2 or period <= 0:
            return out

        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = np.maximum.reduce(
            [
                h[1:] - l[1:],
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ]
        )

        atr = np.full(n, np.nan, dtype=float)
        p_dm = np.full(n, np.nan, dtype=float)
        m_dm = np.full(n, np.nan, dtype=float)

        atr[period] = np.sum(tr[:period])
        p_dm[period] = np.sum(plus_dm[:period])
        m_dm[period] = np.sum(minus_dm[:period])

        for i in range(period + 1, n):
            atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i - 1]
            p_dm[i] = p_dm[i - 1] - (p_dm[i - 1] / period) + plus_dm[i - 1]
            m_dm[i] = m_dm[i - 1] - (m_dm[i - 1] / period) + minus_dm[i - 1]

        plus_di = 100.0 * (p_dm / np.where(atr == 0, np.nan, atr))
        minus_di = 100.0 * (m_dm / np.where(atr == 0, np.nan, atr))
        dx = 100.0 * (np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di)))

        start = period * 2
        if start < n and np.any(np.isfinite(dx[period + 1 : start + 1])):
            out[start] = np.nanmean(dx[period + 1 : start + 1])
            for i in range(start + 1, n):
                out[i] = (out[i - 1] * (period - 1) + dx[i]) / period
        return out

    def structure(self, data, sens: int) -> Tuple[str, Optional[float], Optional[float]]:
        high = data["high"]
        low = data["low"]
        sens = max(1, int(sens))

        piv_hi = []
        piv_lo = []
        for i in range(sens, len(high) - sens):
            if high[i] == max(high[i - sens : i + sens + 1]):
                piv_hi.append(float(high[i]))
            if low[i] == min(low[i - sens : i + sens + 1]):
                piv_lo.append(float(low[i]))

        last_hi = piv_hi[-1] if piv_hi else None
        last_lo = piv_lo[-1] if piv_lo else None

        trend = "ranging"
        if len(piv_hi) >= 2 and len(piv_lo) >= 2:
            if piv_hi[-1] > piv_hi[-2] and piv_lo[-1] > piv_lo[-2]:
                trend = "bullish"
            elif piv_hi[-1] < piv_hi[-2] and piv_lo[-1] < piv_lo[-2]:
                trend = "bearish"

        return trend, last_hi, last_lo

    def _get_sideway_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        c = cfg.get("sideway_scalp", {}) or {}
        adx_period = self.safe_int(c.get("adx_period", 14), 14)
        bb_period = self.safe_int(c.get("bb_period", 20), 20)
        rsi_period = self.safe_int(c.get("rsi_period", 14), 14)

        return {
            "enabled": bool(c.get("enabled", True)),
            "adx_period": max(adx_period, 2),
            "adx_max": max(self.safe_float(c.get("adx_max", 22.0), 22.0), 0.0),
            "bb_period": max(bb_period, 5),
            "bb_std": max(self.safe_float(c.get("bb_std", 2.0), 2.0), 0.1),
            "bb_width_atr_max": max(self.safe_float(c.get("bb_width_atr_max", 6.0), 6.0), 0.1),
            "rsi_period": max(rsi_period, 2),
            "rsi_overbought": self.clamp(self.safe_float(c.get("rsi_overbought", 70.0), 70.0), 50.0, 95.0),
            "rsi_oversold": self.clamp(self.safe_float(c.get("rsi_oversold", 30.0), 30.0), 5.0, 50.0),
            "require_confirmation": bool(c.get("require_confirmation", True)),
            "touch_buffer_atr": self.clamp(self.safe_float(c.get("touch_buffer_atr", 0.10), 0.10), 0.0, 0.50),

            # Adaptive trigger
            "near_trigger_atr": self.clamp(self.safe_float(c.get("near_trigger_atr", 0.50), 0.50), 0.0, 2.0),
            "allow_soft_trigger": bool(c.get("allow_soft_trigger", True)),
            "rsi_soft_band": self.clamp(self.safe_float(c.get("rsi_soft_band", 10.0), 10.0), 0.0, 25.0),

            # Proximity scoring
            "proximity_window_atr": self.clamp(self.safe_float(c.get("proximity_window_atr", 1.00), 1.00), 0.10, 5.0),

            # NEW: minimum proximity score to allow SOFT-ZONE triggers (quality gate)
            "proximity_score_min": self.clamp(self.safe_float(c.get("proximity_score_min", 0.70), 0.70), 0.0, 1.0),
        }

    def generate_signal_package(self) -> Dict[str, Any]:
        cfg = self.reload_config()

        symbol = str(cfg.get("symbol", "GOLD"))
        mode = str(cfg.get("mode", "balanced")).strip().lower()
        tf_cfg = cfg.get("timeframes", {}) or {}

        # Risk contract: prefer cfg.risk.* ; fallback to legacy cfg.atr.*
        risk_cfg = cfg.get("risk", {}) or {}
        if not risk_cfg:
            legacy_atr = cfg.get("atr", {}) or {}
            risk_cfg = {"atr_period": legacy_atr.get("period", 14), "atr_sl_mult": legacy_atr.get("sl_mult", 1.8)}

        atr_period = self.safe_int(risk_cfg.get("atr_period", 14), 14)
        atr_sl_mult = self.safe_float(risk_cfg.get("atr_sl_mult", 1.8), 1.8)
        min_rr = self.safe_float(cfg.get("min_rr", 2.0), 2.0)

        st_cfg = cfg.get("supertrend", {}) or {}
        st_period = self.safe_int(st_cfg.get("period", 10), 10)
        st_mult = self.safe_float(st_cfg.get("multiplier", st_cfg.get("mult", 3.0)), 3.0)

        sens_cfg = cfg.get("structure_sensitivity", {}) or {}
        sens_htf = self.safe_int(sens_cfg.get("htf", 5), 5)
        sens_mtf = self.safe_int(sens_cfg.get("mtf", 4), 4)
        sens_ltf = self.safe_int(sens_cfg.get("ltf", 3), 3)

        htf = str(tf_cfg.get("htf", "H4"))
        mtf = str(tf_cfg.get("mtf", "H1"))
        ltf = str(tf_cfg.get("ltf", "M15"))

        blocked_reasons = []
        watch_state = "NONE"
        debug: Dict[str, Any] = {}

        try:
            htf_data = self.get_data(symbol, self.tf(htf), 600)
            mtf_data = self.get_data(symbol, self.tf(mtf), 800)
            ltf_data = self.get_data(symbol, self.tf(ltf), 1200)
        except Exception as e:
            return {
                "symbol": symbol,
                "direction": "NONE",
                "entry_candidate": None,
                "stop_candidate": None,
                "tp_candidate": None,
                "rr": 0.0,
                "score": 0.0,
                "confidence_py": 0,
                "bos": False,
                "supertrend_ok": False,
                "context": {"blocked_by": "no_data", "error": str(e), "engine_version": ENGINE_VERSION},
            }

        htf_trend, _, _ = self.structure(htf_data, sens_htf)
        mtf_trend, _, _ = self.structure(mtf_data, sens_mtf)
        ltf_trend, bos_hi, bos_lo = self.structure(ltf_data, sens_ltf)

        close = np.asarray(ltf_data["close"], dtype=float)
        high = np.asarray(ltf_data["high"], dtype=float)
        low = np.asarray(ltf_data["low"], dtype=float)
        open_ = np.asarray(ltf_data["open"], dtype=float)

        mtf_close = np.asarray(mtf_data["close"], dtype=float)
        mtf_high = np.asarray(mtf_data["high"], dtype=float)
        mtf_low = np.asarray(mtf_data["low"], dtype=float)

        ltf_close = float(close[-1])

        atr_arr = self.atr(high, low, close, atr_period)
        atr_val = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else 0.0

        st_line, st_dir_arr = self.supertrend(high, low, close, st_period, st_mult)
        st_dir = "bullish" if int(st_dir_arr[-1]) == 1 else "bearish"
        st_value = float(st_line[-1]) if len(st_line) else float("nan")

        if mode == "sideway_scalp":
            direction_bias = "NEUTRAL"
            bias_source = "SIDEWAY_MODE"
        else:
            direction_bias = "NONE"
            bias_source = "NO_CLEAR_BIAS"
            if htf_trend == "bullish":
                direction_bias = "BUY"
                bias_source = "HTF"
            elif htf_trend == "bearish":
                direction_bias = "SELL"
                bias_source = "HTF"
            elif mtf_trend == "bullish":
                direction_bias = "BUY"
                bias_source = "MTF_FALLBACK"
            elif mtf_trend == "bearish":
                direction_bias = "SELL"
                bias_source = "MTF_FALLBACK"
            else:
                blocked_reasons.append("no_clear_bias")

        supertrend_ok = True if mode == "sideway_scalp" else False
        if mode != "sideway_scalp":
            if direction_bias == "BUY" and st_dir == "bullish":
                supertrend_ok = True
            elif direction_bias == "SELL" and st_dir == "bearish":
                supertrend_ok = True
            else:
                if direction_bias in ("BUY", "SELL"):
                    blocked_reasons.append("supertrend_conflict")

        direction_out = "NONE"
        entry_candidate = None
        stop_candidate = None
        tp_candidate = None
        rr = 0.0

        sideway_ctx: Dict[str, Any] = {}

        distance_buy = float("nan")
        distance_sell = float("nan")
        proximity_side = "NONE"
        proximity_best_dist = float("nan")
        proximity_score = 0.0

        if mode == "sideway_scalp":
            k = self._get_sideway_knobs(cfg)

            adx_arr = self.adx(mtf_high, mtf_low, mtf_close, k["adx_period"])
            adx_val = float(adx_arr[-1]) if (len(adx_arr) and np.isfinite(adx_arr[-1])) else float("nan")

            bb_u, bb_m, bb_l = self.bollinger(close, k["bb_period"], k["bb_std"])
            bb_upper = float(bb_u[-1]) if (len(bb_u) and np.isfinite(bb_u[-1])) else float("nan")
            bb_mid = float(bb_m[-1]) if (len(bb_m) and np.isfinite(bb_m[-1])) else float("nan")
            bb_lower = float(bb_l[-1]) if (len(bb_l) and np.isfinite(bb_l[-1])) else float("nan")

            bb_width = float(bb_upper - bb_lower) if (np.isfinite(bb_upper) and np.isfinite(bb_lower)) else float("nan")
            bb_width_atr = float(bb_width / atr_val) if (atr_val > 0 and np.isfinite(bb_width)) else float("nan")

            rsi_arr = self.rsi(close, k["rsi_period"])
            rsi_val = float(rsi_arr[-1]) if (len(rsi_arr) and np.isfinite(rsi_arr[-1])) else float("nan")

            last_open = float(open_[-1])
            last_close = float(close[-1])
            prev_open = float(open_[-2]) if len(open_) >= 2 else last_open
            prev_close = float(close[-2]) if len(close) >= 2 else last_close

            bullish_reversal = (last_close > last_open) and (prev_close < prev_open)
            bearish_reversal = (last_close < last_open) and (prev_close > prev_open)

            if not (np.isfinite(adx_val) and adx_val <= k["adx_max"]):
                blocked_reasons.append("not_sideway_adx")
            if not (np.isfinite(bb_width_atr) and bb_width_atr <= k["bb_width_atr_max"]):
                blocked_reasons.append("not_sideway_bbwidth")

            touch_buffer_points = float(k["touch_buffer_atr"] * atr_val) if atr_val > 0 else 0.0
            near_points = float(k["near_trigger_atr"] * atr_val) if atr_val > 0 else 0.0

            buy_trigger_level = (bb_lower + touch_buffer_points) if np.isfinite(bb_lower) else float("nan")
            sell_trigger_level = (bb_upper - touch_buffer_points) if np.isfinite(bb_upper) else float("nan")

            if np.isfinite(buy_trigger_level):
                distance_buy = float(ltf_close - buy_trigger_level)
            if np.isfinite(sell_trigger_level):
                distance_sell = float(sell_trigger_level - ltf_close)

            d_buy_abs = abs(distance_buy) if np.isfinite(distance_buy) else float("inf")
            d_sell_abs = abs(distance_sell) if np.isfinite(distance_sell) else float("inf")

            if d_buy_abs < d_sell_abs:
                proximity_side = "BUY"
                proximity_best_dist = float(d_buy_abs)
            else:
                proximity_side = "SELL"
                proximity_best_dist = float(d_sell_abs)

            window = float(max(k["proximity_window_atr"] * atr_val, 1e-9))
            if np.isfinite(proximity_best_dist) and window > 0:
                proximity_score = float(max(0.0, 1.0 - (proximity_best_dist / window)))
            else:
                proximity_score = 0.0

            # Trigger evaluation
            if len(blocked_reasons) == 0 and k["enabled"]:
                buy_touch = np.isfinite(buy_trigger_level) and (ltf_close <= buy_trigger_level)
                sell_touch = np.isfinite(sell_trigger_level) and (ltf_close >= sell_trigger_level)

                buy_soft = np.isfinite(buy_trigger_level) and (ltf_close <= (buy_trigger_level + near_points))
                sell_soft = np.isfinite(sell_trigger_level) and (ltf_close >= (sell_trigger_level - near_points))

                buy_confirm_hard = (np.isfinite(rsi_val) and rsi_val <= k["rsi_oversold"]) or bullish_reversal
                sell_confirm_hard = (np.isfinite(rsi_val) and rsi_val >= k["rsi_overbought"]) or bearish_reversal

                buy_soft_rsi = (np.isfinite(rsi_val) and rsi_val <= (k["rsi_oversold"] + k["rsi_soft_band"]))
                sell_soft_rsi = (np.isfinite(rsi_val) and rsi_val >= (k["rsi_overbought"] - k["rsi_soft_band"]))
                buy_confirm_soft = bullish_reversal or buy_soft_rsi
                sell_confirm_soft = bearish_reversal or sell_soft_rsi

                # Hard-touch stays allowed (no proximity_score_min gate needed)
                if buy_touch:
                    if (not k["require_confirmation"]) or buy_confirm_hard:
                        direction_out = "BUY"
                    else:
                        watch_state = "WATCH_PROXIMITY"
                        blocked_reasons.append("no_confirmation")
                elif sell_touch:
                    if (not k["require_confirmation"]) or sell_confirm_hard:
                        direction_out = "SELL"
                    else:
                        watch_state = "WATCH_PROXIMITY"
                        blocked_reasons.append("no_confirmation")
                else:
                    # Soft-zone path (apply proximity_score_min as quality gate)
                    in_soft_zone = bool(buy_soft or sell_soft)
                    if in_soft_zone:
                        watch_state = "WATCH_PROXIMITY"

                        if proximity_score < float(k["proximity_score_min"]):
                            blocked_reasons.append("low_proximity_score")
                        else:
                            if k["allow_soft_trigger"]:
                                if buy_soft and ((not k["require_confirmation"]) or buy_confirm_soft):
                                    direction_out = "BUY"
                                elif sell_soft and ((not k["require_confirmation"]) or sell_confirm_soft):
                                    direction_out = "SELL"
                                else:
                                    blocked_reasons.append("no_confirmation")
                            else:
                                blocked_reasons.append("no_mean_reversion_trigger")
                    else:
                        blocked_reasons.append("no_mean_reversion_trigger")
            elif not k["enabled"]:
                blocked_reasons.append("sideway_disabled")

            # Build SL/TP
            if direction_out in ("BUY", "SELL"):
                tick = self.get_tick(symbol)

                debug["debug_direction_attempted"] = direction_out
                debug["debug_tick_bid"] = float(getattr(tick, "bid", float("nan")))
                debug["debug_tick_ask"] = float(getattr(tick, "ask", float("nan")))
                spread = float(getattr(tick, "ask", 0.0)) - float(getattr(tick, "bid", 0.0))
                debug["debug_spread"] = float(spread)

                entry_candidate = float(tick.ask if direction_out == "BUY" else tick.bid)

                sl_dist = float(max(atr_val * atr_sl_mult, 1e-9))
                debug["debug_sl_dist"] = float(sl_dist)
                debug["debug_min_rr_used"] = float(min_rr)
                debug["debug_min_rr_check"] = float(min_rr)

                if direction_out == "BUY":
                    stop_candidate = float(entry_candidate - sl_dist)
                    tp_candidate = float(entry_candidate + sl_dist * float(max(min_rr, 0.1)))
                    risk = entry_candidate - stop_candidate
                    reward = tp_candidate - entry_candidate
                    rr = float(reward / risk) if risk > 1e-9 else 0.0
                else:
                    stop_candidate = float(entry_candidate + sl_dist)
                    tp_candidate = float(entry_candidate - sl_dist * float(max(min_rr, 0.1)))
                    risk = stop_candidate - entry_candidate
                    reward = entry_candidate - tp_candidate
                    rr = float(reward / risk) if risk > 1e-9 else 0.0

                debug["debug_entry"] = float(entry_candidate)
                debug["debug_stop"] = float(stop_candidate)
                debug["debug_tp"] = float(tp_candidate)
                debug["debug_rr"] = float(rr)

                if rr < min_rr:
                    debug["debug_rr_fail_reason"] = "rr < min_rr"
                    direction_out = "NONE"
                    entry_candidate = stop_candidate = tp_candidate = None
                    rr = 0.0
                    blocked_reasons.append("rr_too_low")

            sideway_ctx = {
                "adx_period": k["adx_period"],
                "adx": adx_val,
                "adx_max": k["adx_max"],
                "bb_period": k["bb_period"],
                "bb_std": k["bb_std"],
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "bb_width": bb_width,
                "bb_width_atr": bb_width_atr,
                "bb_width_atr_max": k["bb_width_atr_max"],
                "rsi_period": k["rsi_period"],
                "rsi": rsi_val,
                "rsi_overbought": k["rsi_overbought"],
                "rsi_oversold": k["rsi_oversold"],
                "bullish_reversal": bool(bullish_reversal),
                "bearish_reversal": bool(bearish_reversal),
                "touch_buffer_atr": k["touch_buffer_atr"],
                "touch_buffer_points": touch_buffer_points,
                "near_trigger_atr": k["near_trigger_atr"],
                "near_trigger_points": near_points,
                "allow_soft_trigger": bool(k["allow_soft_trigger"]),
                "require_confirmation": bool(k["require_confirmation"]),
                "rsi_soft_band": k["rsi_soft_band"],
                "proximity_window_atr": k["proximity_window_atr"],
                "proximity_score_min": k["proximity_score_min"],
                "buy_trigger_level": buy_trigger_level,
                "sell_trigger_level": sell_trigger_level,
                "distance_buy": distance_buy,
                "distance_sell": distance_sell,
                "proximity_side": proximity_side,
                "proximity_best_dist": proximity_best_dist,
                "proximity_score": proximity_score,
            }

        blocked_by = ",".join(blocked_reasons) if blocked_reasons else None

        if debug:
            sideway_ctx["debug"] = debug

        return {
            "symbol": symbol,
            "direction": direction_out,
            "entry_candidate": entry_candidate,
            "stop_candidate": stop_candidate,
            "tp_candidate": tp_candidate,
            "rr": float(rr),
            "score": float(proximity_score) if mode == "sideway_scalp" else 0.0,
            "confidence_py": 0,
            "bos": False,
            "supertrend_ok": bool(supertrend_ok),
            "context": {
                "engine_version": ENGINE_VERSION,
                "mode": mode,
                "HTF_trend": htf_trend,
                "MTF_trend": mtf_trend,
                "LTF_trend": ltf_trend,
                "bias_source": bias_source,
                "direction_bias": direction_bias,
                "watch_state": watch_state,
                "supertrend_dir": st_dir,
                "supertrend_value": st_value,
                "bos_ref_high": bos_hi,
                "bos_ref_low": bos_lo,
                "atr": atr_val,
                "atr_period": atr_period,
                "atr_sl_mult": atr_sl_mult,
                "min_rr": min_rr,
                **sideway_ctx,
                "blocked_by": blocked_by,
            },
        }