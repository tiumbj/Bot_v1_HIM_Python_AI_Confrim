"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 2.11.1

Changelog:
- 2.11.1 (2026-03-01):
  - FIX: Restore MT5 data integration while adding Breakout mode (BOS + buffer + retest + Hybrid SL)
  - ADD: breakout config knobs with sane defaults (no more NotImplemented get_rates)
  - KEEP: sideway_scalp logic unchanged
- 2.10.6 (2026-02-28):
  - FIX: Prevent validator E_RR_FLOOR due to floating-point precision
      - Use rr_eps to compute TP with target_rr = min_rr + rr_eps (strictly above RR floor)
      - Keep fail-closed RR gate, but compare with epsilon tolerance: rr < (min_rr - eps)
      - Add debug fields: debug_target_rr, debug_rr_eps
  - KEEP: proximity_score_min gate (quality filter) for sideway_scalp
  - KEEP: adaptive trigger knobs (near_trigger_atr, allow_soft_trigger, rsi_soft_band)
  - KEEP: sideway_scalp NEUTRAL (bias_source="SIDEWAY_MODE", direction_bias="NEUTRAL")
  - KEEP: regime gate = ADX low + BB width normalized by ATR
  - KEEP: debug bag to diagnose rr/tick anomalies (only when attempting trade)

Notes:
- This version is intended to preserve strict RR floor at validator while avoiding float artifacts.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import MetaTrader5 as mt5

from config_resolver import resolve_effective_config


ENGINE_VERSION = "2.11.1"


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
            raise RuntimeError(f"no rates (symbol={symbol}, timeframe={timeframe}, n={n})")

        # mt5 rates: time, open, high, low, close, tick_volume, spread, real_volume
        return {
            "time": np.array([r["time"] for r in rates], dtype=np.int64),
            "open": np.array([r["open"] for r in rates], dtype=float),
            "high": np.array([r["high"] for r in rates], dtype=float),
            "low": np.array([r["low"] for r in rates], dtype=float),
            "close": np.array([r["close"] for r in rates], dtype=float),
            "volume": np.array([r["tick_volume"] for r in rates], dtype=float),
        }

    # -----------------------------
    # Indicators
    # -----------------------------
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        tr = np.zeros_like(close, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        atr = np.full_like(close, np.nan, dtype=float)
        if len(close) < period + 1:
            return atr

        atr[period] = np.nanmean(tr[1 : period + 1])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        close = np.asarray(close, dtype=float)
        delta = np.diff(close, prepend=close[0])

        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        rsi = np.full_like(close, np.nan, dtype=float)
        if len(close) < period + 1:
            return rsi

        avg_gain = np.nanmean(gain[1 : period + 1])
        avg_loss = np.nanmean(loss[1 : period + 1])

        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period + 1, len(close)):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        close = np.asarray(close, dtype=float)
        upper = np.full_like(close, np.nan, dtype=float)
        mid = np.full_like(close, np.nan, dtype=float)
        lower = np.full_like(close, np.nan, dtype=float)

        if len(close) < period:
            return upper, mid, lower

        for i in range(period - 1, len(close)):
            window = close[i - period + 1 : i + 1]
            m = float(np.nanmean(window))
            s = float(np.nanstd(window))
            mid[i] = m
            upper[i] = m + std_mult * s
            lower[i] = m - std_mult * s

        return upper, mid, lower

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        adx = np.full_like(close, np.nan, dtype=float)
        if len(close) < period + 2:
            return adx

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = np.zeros(len(close) - 1, dtype=float)
        for i in range(1, len(close)):
            tr[i - 1] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        atr = np.full_like(tr, np.nan, dtype=float)
        atr[period - 1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        plus_di = np.full_like(tr, np.nan, dtype=float)
        minus_di = np.full_like(tr, np.nan, dtype=float)

        plus_dm_sm = np.full_like(plus_dm, np.nan, dtype=float)
        minus_dm_sm = np.full_like(minus_dm, np.nan, dtype=float)

        plus_dm_sm[period - 1] = np.nanmean(plus_dm[:period])
        minus_dm_sm[period - 1] = np.nanmean(minus_dm[:period])

        for i in range(period, len(plus_dm)):
            plus_dm_sm[i] = (plus_dm_sm[i - 1] * (period - 1) + plus_dm[i]) / period
            minus_dm_sm[i] = (minus_dm_sm[i - 1] * (period - 1) + minus_dm[i]) / period

        for i in range(period - 1, len(tr)):
            if atr[i] and atr[i] > 0:
                plus_di[i] = 100.0 * (plus_dm_sm[i] / atr[i])
                minus_di[i] = 100.0 * (minus_dm_sm[i] / atr[i])

        dx = np.full_like(tr, np.nan, dtype=float)
        for i in range(period - 1, len(tr)):
            p = plus_di[i]
            m = minus_di[i]
            denom = p + m
            if denom and denom > 0:
                dx[i] = 100.0 * (abs(p - m) / denom)

        adx_tr = np.full_like(tr, np.nan, dtype=float)
        start = (period - 1) + (period - 1)
        if start < len(tr):
            adx_tr[start] = np.nanmean(dx[period - 1 : start + 1])
            for i in range(start + 1, len(tr)):
                adx_tr[i] = (adx_tr[i - 1] * (period - 1) + dx[i]) / period

        adx[1:] = adx_tr
        return adx

    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, mult: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        atr = TradingEngine.atr(high, low, close, period)
        hl2 = (high + low) / 2.0

        upperband = hl2 + mult * atr
        lowerband = hl2 - mult * atr

        st = np.full_like(close, np.nan, dtype=float)
        direction = np.full_like(close, 1, dtype=int)  # 1 bullish, -1 bearish

        for i in range(1, len(close)):
            if not np.isfinite(atr[i]):
                continue

            if np.isfinite(st[i - 1]):
                upperband[i] = min(upperband[i], upperband[i - 1]) if close[i - 1] <= upperband[i - 1] else upperband[i]
                lowerband[i] = max(lowerband[i], lowerband[i - 1]) if close[i - 1] >= lowerband[i - 1] else lowerband[i]

            if np.isfinite(st[i - 1]):
                if st[i - 1] == upperband[i - 1]:
                    if close[i] <= upperband[i]:
                        st[i] = upperband[i]
                        direction[i] = -1
                    else:
                        st[i] = lowerband[i]
                        direction[i] = 1
                else:
                    if close[i] >= lowerband[i]:
                        st[i] = lowerband[i]
                        direction[i] = 1
                    else:
                        st[i] = upperband[i]
                        direction[i] = -1
            else:
                st[i] = lowerband[i] if close[i] >= hl2[i] else upperband[i]
                direction[i] = 1 if close[i] >= hl2[i] else -1

        dir01 = np.where(direction == 1, 1, 0).astype(int)
        return st, dir01

    # -----------------------------
    # Structure
    # -----------------------------
    @staticmethod
    def structure(data: Dict[str, Any], sensitivity: int = 3) -> Tuple[str, float, float]:
        high = np.asarray(data["high"], dtype=float)
        low = np.asarray(data["low"], dtype=float)
        close = np.asarray(data["close"], dtype=float)

        if len(close) < (sensitivity * 2 + 5):
            return "ranging", float("nan"), float("nan")

        piv_hi = []
        piv_lo = []
        for i in range(sensitivity, len(close) - sensitivity):
            window_hi = high[i - sensitivity : i + sensitivity + 1]
            window_lo = low[i - sensitivity : i + sensitivity + 1]
            if high[i] == np.nanmax(window_hi):
                piv_hi.append((i, float(high[i])))
            if low[i] == np.nanmin(window_lo):
                piv_lo.append((i, float(low[i])))

        bos_hi = piv_hi[-1][1] if len(piv_hi) >= 1 else float("nan")
        bos_lo = piv_lo[-1][1] if len(piv_lo) >= 1 else float("nan")

        trend = "ranging"
        if len(piv_hi) >= 2 and len(piv_lo) >= 2:
            last_hi = piv_hi[-1][1]
            prev_hi = piv_hi[-2][1]
            last_lo = piv_lo[-1][1]
            prev_lo = piv_lo[-2][1]
            if last_hi > prev_hi and last_lo > prev_lo:
                trend = "bullish"
            elif last_hi < prev_hi and last_lo < prev_lo:
                trend = "bearish"
            else:
                trend = "ranging"

        return trend, float(bos_hi), float(bos_lo)

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

    def _get_breakout_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Breakout mode knobs (mode=breakout)

        - confirm_buffer_atr: ต้องปิดเหนือ/ใต้ BOS level ด้วย buffer*ATR เพื่อกันไส้เทียนหลอก
        - retest_required: ต้องมีการ retest level ภายใน lookback bars
        - retest_band_atr: ระยะยอมรับ retest รอบ level (เป็น ATR multiple)
        - sl_buffer_atr: บัฟเฟอร์ SL เพิ่มจากโครงสร้าง (structure) ด้วยความผันผวน (ATR)

        - bb_width_atr_min: volatility expansion gate (BBWidth/ATR ต้อง >= ค่านี้)
        """
        b = cfg.get("breakout", {}) or {}

        confirm_buffer_atr = self.clamp(self.safe_float(b.get("confirm_buffer_atr", 0.05), 0.05), 0.0, 1.0)
        retest_required = bool(b.get("retest_required", True))
        retest_band_atr = self.clamp(self.safe_float(b.get("retest_band_atr", 0.30), 0.30), 0.0, 3.0)
        retest_lookback_bars = max(self.safe_int(b.get("retest_lookback_bars", 5), 5), 2)

        sl_buffer_atr = self.clamp(
            self.safe_float(
                b.get("sl_buffer_atr", max(confirm_buffer_atr, retest_band_atr, 0.10)),
                max(confirm_buffer_atr, retest_band_atr, 0.10),
            ),
            0.0,
            5.0,
        )

        bb_width_atr_min = self.clamp(self.safe_float(b.get("bb_width_atr_min", 5.0), 5.0), 0.1, 50.0)

        proximity_threshold_atr = self.clamp(self.safe_float(b.get("proximity_threshold_atr", 1.50), 1.50), 0.1, 10.0)
        proximity_min_score = self.clamp(self.safe_float(b.get("proximity_min_score", 0.0), 0.0), 0.0, 100.0)

        return {
            "confirm_buffer_atr": confirm_buffer_atr,
            "retest_required": retest_required,
            "retest_band_atr": retest_band_atr,
            "retest_lookback_bars": retest_lookback_bars,
            "sl_buffer_atr": sl_buffer_atr,
            "bb_width_atr_min": bb_width_atr_min,
            "proximity_threshold_atr": proximity_threshold_atr,
            "proximity_min_score": proximity_min_score,
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

                # Hard-touch stays allowed (no proximity gate)
                buy_allowed = bool(buy_touch and buy_confirm_hard)
                sell_allowed = bool(sell_touch and sell_confirm_hard)

                # Soft-zone requires proximity score gate
                if not buy_allowed and k["allow_soft_trigger"]:
                    if buy_soft and buy_confirm_soft and (proximity_score >= float(k["proximity_score_min"])):
                        buy_allowed = True

                if not sell_allowed and k["allow_soft_trigger"]:
                    if sell_soft and sell_confirm_soft and (proximity_score >= float(k["proximity_score_min"])):
                        sell_allowed = True

                # Mutual exclusion fail-closed
                if buy_allowed and not sell_allowed:
                    direction_out = "BUY"
                elif sell_allowed and not buy_allowed:
                    direction_out = "SELL"

                # Candidate prices
                if direction_out in ("BUY", "SELL") and atr_val > 0:
                    entry_candidate = float(ltf_close)
                    if direction_out == "BUY":
                        stop_candidate = float(entry_candidate - (atr_sl_mult * atr_val))
                    else:
                        stop_candidate = float(entry_candidate + (atr_sl_mult * atr_val))

                    rr_eps = 1e-6
                    target_rr = float(min_rr) + rr_eps
                    debug["debug_target_rr"] = target_rr
                    debug["debug_rr_eps"] = rr_eps

                    if direction_out == "BUY":
                        risk = float(entry_candidate - stop_candidate)
                        if risk > 0:
                            tp_candidate = float(entry_candidate + (risk * target_rr))
                            rr = float((tp_candidate - entry_candidate) / risk)
                    else:
                        risk = float(stop_candidate - entry_candidate)
                        if risk > 0:
                            tp_candidate = float(entry_candidate - (risk * target_rr))
                            rr = float((entry_candidate - tp_candidate) / risk)

                    eps = 1e-9
                    if rr < (float(min_rr) - eps):
                        blocked_reasons.append("rr_below_floor")

            sideway_ctx = {
                "sideway": True,
                "adx_val": adx_val,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "bb_width": bb_width,
                "bb_width_atr": bb_width_atr,
                "rsi_val": rsi_val,
                "touch_buffer_atr": k["touch_buffer_atr"],
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

        elif mode == "breakout":
            kb = self._get_breakout_knobs(cfg)

            adx_arr = self.adx(
                mtf_high,
                mtf_low,
                mtf_close,
                self.safe_int((cfg.get("sideway_scalp", {}) or {}).get("adx_period", 14), 14),
            )
            adx_val = float(adx_arr[-1]) if (len(adx_arr) and np.isfinite(adx_arr[-1])) else float("nan")

            sw = cfg.get("sideway_scalp", {}) or {}
            bb_period = self.safe_int(sw.get("bb_period", 20), 20)
            bb_std = self.safe_float(sw.get("bb_std", 2.0), 2.0)

            bb_u, bb_m, bb_l = self.bollinger(close, bb_period, bb_std)
            bb_upper = float(bb_u[-1]) if (len(bb_u) and np.isfinite(bb_u[-1])) else float("nan")
            bb_lower = float(bb_l[-1]) if (len(bb_l) and np.isfinite(bb_l[-1])) else float("nan")
            bb_width = float(bb_upper - bb_lower) if (np.isfinite(bb_upper) and np.isfinite(bb_lower)) else float("nan")
            bb_width_atr = float(bb_width / atr_val) if (atr_val > 0 and np.isfinite(bb_width)) else float("nan")

            if direction_bias not in ("BUY", "SELL"):
                blocked_reasons.append("no_clear_bias")
            if direction_bias in ("BUY", "SELL") and not supertrend_ok:
                blocked_reasons.append("supertrend_conflict")

            if not (np.isfinite(bb_width_atr) and bb_width_atr >= float(kb["bb_width_atr_min"])):
                blocked_reasons.append("no_vol_expansion")

            bos_ref_high = float(bos_hi) if np.isfinite(bos_hi) else float("nan")
            bos_ref_low = float(bos_lo) if np.isfinite(bos_lo) else float("nan")

            confirm_buf = float(kb["confirm_buffer_atr"] * atr_val) if atr_val > 0 else 0.0
            retest_band = float(kb["retest_band_atr"] * atr_val) if atr_val > 0 else 0.0
            sl_buf = float(kb["sl_buffer_atr"] * atr_val) if atr_val > 0 else 0.0

            lookback = int(kb["retest_lookback_bars"])
            lookback = max(2, min(lookback, len(close)))

            proximity_score_bk = 0.0

            if len(blocked_reasons) == 0:
                if direction_bias == "BUY":
                    if not (np.isfinite(bos_ref_high) and np.isfinite(bos_ref_low)):
                        blocked_reasons.append("no_bos_refs")
                    else:
                        level = bos_ref_high
                        confirm_level = level + confirm_buf
                        if not (ltf_close > confirm_level):
                            blocked_reasons.append("no_bos_break")
                        else:
                            if bool(kb["retest_required"]):
                                recent_low = float(np.nanmin(low[-lookback:]))
                                if not (np.isfinite(recent_low) and recent_low <= (level + retest_band) and ltf_close > level):
                                    blocked_reasons.append("no_retest")

                            if atr_val > 0 and float(kb["proximity_threshold_atr"]) > 0:
                                dist_atr = float(abs(ltf_close - level) / atr_val)
                                th = float(kb["proximity_threshold_atr"])
                                proximity_score_bk = float(max(0.0, (th - dist_atr) / th) * 100.0)
                            if proximity_score_bk < float(kb["proximity_min_score"]):
                                blocked_reasons.append("low_proximity_score")

                            if len(blocked_reasons) == 0:
                                direction_out = "BUY"
                                entry_candidate = float(ltf_close)

                                sl_base = bos_ref_low if np.isfinite(bos_ref_low) else float(entry_candidate - (atr_sl_mult * atr_val))
                                stop_candidate = float(sl_base - sl_buf)

                                risk = float(entry_candidate - stop_candidate)
                                if risk <= 0:
                                    blocked_reasons.append("bad_risk")
                                else:
                                    rr_eps = 1e-6
                                    target_rr = float(min_rr) + rr_eps
                                    tp_candidate = float(entry_candidate + (risk * target_rr))
                                    rr = float((tp_candidate - entry_candidate) / risk)

                else:  # SELL
                    if not (np.isfinite(bos_ref_low) and np.isfinite(bos_ref_high)):
                        blocked_reasons.append("no_bos_refs")
                    else:
                        level = bos_ref_low
                        confirm_level = level - confirm_buf
                        if not (ltf_close < confirm_level):
                            blocked_reasons.append("no_bos_break")
                        else:
                            if bool(kb["retest_required"]):
                                recent_high = float(np.nanmax(high[-lookback:]))
                                if not (np.isfinite(recent_high) and recent_high >= (level - retest_band) and ltf_close < level):
                                    blocked_reasons.append("no_retest")

                            if atr_val > 0 and float(kb["proximity_threshold_atr"]) > 0:
                                dist_atr = float(abs(ltf_close - level) / atr_val)
                                th = float(kb["proximity_threshold_atr"])
                                proximity_score_bk = float(max(0.0, (th - dist_atr) / th) * 100.0)
                            if proximity_score_bk < float(kb["proximity_min_score"]):
                                blocked_reasons.append("low_proximity_score")

                            if len(blocked_reasons) == 0:
                                direction_out = "SELL"
                                entry_candidate = float(ltf_close)

                                sl_base = bos_ref_high if np.isfinite(bos_ref_high) else float(entry_candidate + (atr_sl_mult * atr_val))
                                stop_candidate = float(sl_base + sl_buf)

                                risk = float(stop_candidate - entry_candidate)
                                if risk <= 0:
                                    blocked_reasons.append("bad_risk")
                                else:
                                    rr_eps = 1e-6
                                    target_rr = float(min_rr) + rr_eps
                                    tp_candidate = float(entry_candidate - (risk * target_rr))
                                    rr = float((entry_candidate - tp_candidate) / risk)

            sideway_ctx = {
                "breakout": True,
                "adx_val": adx_val,
                "bb_width_atr": bb_width_atr,
                "bb_width_atr_min": float(kb["bb_width_atr_min"]),
                "confirm_buffer_atr": float(kb["confirm_buffer_atr"]),
                "retest_required": bool(kb["retest_required"]),
                "retest_band_atr": float(kb["retest_band_atr"]),
                "retest_lookback_bars": int(kb["retest_lookback_bars"]),
                "sl_buffer_atr": float(kb["sl_buffer_atr"]),
                "proximity_score": float(proximity_score_bk),
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
            "score": float(proximity_score),
            "confidence_py": 0,
            "bos": bool(direction_out in ("BUY", "SELL")),
            "supertrend_ok": bool(supertrend_ok),
            "context": {
                "blocked_by": blocked_by,
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
                "bos_ref_high": float(bos_hi) if np.isfinite(bos_hi) else None,
                "bos_ref_low": float(bos_lo) if np.isfinite(bos_lo) else None,
                "atr": float(atr_val),
                "atr_period": int(atr_period),
                "atr_sl_mult": float(atr_sl_mult),
                "min_rr": float(min_rr),
                **sideway_ctx,
            },
        }