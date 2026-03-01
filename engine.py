"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 2.11.0

Changelog:
- 2.11.0 (2026-03-01):
  - ADD: Breakout mode v1.0 (BOS + confirmation buffer + optional retest + supertrend confirm)
  - ADD: Hybrid SL (structure level + ATR volatility buffer)
  - ADD: BBWidth/ATR expansion gate for breakout to reduce fake breaks
  - KEEP: Sideway scalping logic unchanged
- 2.10.6 (2026-02-28):
  - FIX: Prevent validator E_RR_FLOOR due to floating-point precision
      - Use rr_eps to compute TP with target_rr = min_rr + rr_eps (strictly above RR floor)
      - Keep fail-closed RR gate, but compare with epsilon tolerance: rr < (min_rr - eps)
      - Add debug fields: debug_target_rr, debug_rr_eps
  - KEEP: proximity_score_min gate (quality filter) for sideway_scalp
  - KEEP: adaptive trigger knobs (near_trigger_atr, allow_soft_trigger, rsi_soft_band)
  - KEEP: sideway_scalp NEUTRAL bias (mean-reversion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import numpy as np

ENGINE_VERSION = "2.11.0"


@dataclass
class OHLCV:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class TradingEngine:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)

    # -----------------------------
    # Utils
    # -----------------------------
    def _load_config(self, path: str) -> Dict[str, Any]:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    def clamp(self, x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    # -----------------------------
    # Indicators
    # -----------------------------
    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
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

    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
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

    def bollinger(self, close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
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

        # Wilder smoothing
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

        # ADX smoothing
        adx_tr = np.full_like(tr, np.nan, dtype=float)
        start = (period - 1) + (period - 1)
        if start < len(tr):
            adx_tr[start] = np.nanmean(dx[period - 1 : start + 1])
            for i in range(start + 1, len(tr)):
                adx_tr[i] = (adx_tr[i - 1] * (period - 1) + dx[i]) / period

        # align to close length
        adx[1:] = adx_tr
        return adx

    def supertrend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, mult: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        atr = self.atr(high, low, close, period)
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

        # convert to 1 bullish / 0 bearish for earlier code compatibility
        dir01 = np.where(direction == 1, 1, 0).astype(int)
        return st, dir01

    # -----------------------------
    # Structure
    # -----------------------------
    def structure(self, data: Dict[str, Any], sensitivity: int = 3) -> Tuple[str, float, float]:
        """
        Returns:
          trend: 'bullish' | 'bearish' | 'ranging'
          bos_ref_high: float (reference high for BOS)
          bos_ref_low: float (reference low for BOS)
        """
        high = np.asarray(data["high"], dtype=float)
        low = np.asarray(data["low"], dtype=float)
        close = np.asarray(data["close"], dtype=float)

        if len(close) < (sensitivity * 2 + 5):
            return "ranging", float("nan"), float("nan")

        # crude pivot detection
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

        # trend heuristic using last two pivots
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

    # -----------------------------
    # Config knobs
    # -----------------------------
    def _get_sideway_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        c = cfg.get("sideway_scalp", {}) if isinstance(cfg.get("sideway_scalp", {}), dict) else {}
        return {
            "enabled": bool(c.get("enabled", True)),

            "adx_period": int(self.clamp(self.safe_float(c.get("adx_period", 14), 14), 5, 50)),
            "adx_max": self.clamp(self.safe_float(c.get("adx_max", 25.0), 25.0), 10.0, 60.0),

            "bb_period": int(self.clamp(self.safe_float(c.get("bb_period", 20), 20), 5, 200)),
            "bb_std": self.clamp(self.safe_float(c.get("bb_std", 2.0), 2.0), 0.5, 5.0),
            "bb_width_atr_max": self.clamp(self.safe_float(c.get("bb_width_atr_max", 6.0), 6.0), 0.5, 50.0),

            "rsi_period": int(self.clamp(self.safe_float(c.get("rsi_period", 14), 14), 5, 100)),
            "rsi_overbought": self.clamp(self.safe_float(c.get("rsi_overbought", 70), 70), 50.0, 95.0),
            "rsi_oversold": self.clamp(self.safe_float(c.get("rsi_oversold", 30), 30), 5.0, 50.0),

            "touch_buffer_atr": self.clamp(self.safe_float(c.get("touch_buffer_atr", 0.20), 0.20), 0.0, 2.0),
            "near_trigger_atr": self.clamp(self.safe_float(c.get("near_trigger_atr", 0.90), 0.90), 0.0, 5.0),

            "allow_soft_trigger": bool(c.get("allow_soft_trigger", True)),
            "require_confirmation": bool(c.get("require_confirmation", False)),
            "rsi_soft_band": self.clamp(self.safe_float(c.get("rsi_soft_band", 10.0), 10.0), 0.0, 25.0),

            # Proximity scoring
            "proximity_window_atr": self.clamp(self.safe_float(c.get("proximity_window_atr", 1.00), 1.00), 0.10, 5.0),

            # NEW: minimum proximity score to allow SOFT-ZONE triggers (quality gate)
            "proximity_score_min": self.clamp(self.safe_float(c.get("proximity_score_min", 0.70), 0.70), 0.0, 1.0),
        }

    def _get_breakout_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Breakout knobs (mode=breakout)

        Config sources:
        - cfg["breakout"] (top-level)
          - confirm_buffer_atr: ต้องทะลุระดับ BOS ด้วยบัฟเฟอร์ ATR เพื่อกันไส้เทียนหลอก
          - retest_required: ต้องมี retest หลัง breakout (ตรวจจาก window ย้อนหลัง)
          - retest_band_atr: แบนด์สำหรับ retest รอบระดับ breakout
          - proximity_threshold_atr / proximity_min_score: ใช้เป็น quality gate เพิ่ม (optional)
        - cfg["sideway_scalp"] ใช้ bb_period/bb_std/adx_period เป็นค่า default เพื่อคำนวณ ADX/BBWidth/ATR gate
        """
        b = cfg.get("breakout", {}) if isinstance(cfg.get("breakout", {}), dict) else {}
        s = cfg.get("sideway_scalp", {}) if isinstance(cfg.get("sideway_scalp", {}), dict) else {}

        confirm_buffer_atr = self.clamp(self.safe_float(b.get("confirm_buffer_atr", 0.05), 0.05), 0.0, 1.0)
        retest_band_atr = self.clamp(self.safe_float(b.get("retest_band_atr", 0.30), 0.30), 0.0, 2.0)

        # Hybrid SL buffer (structure + ATR buffer)
        default_sl_buffer = max(confirm_buffer_atr, retest_band_atr, 0.10)

        return {
            "confirm_buffer_atr": confirm_buffer_atr,
            "retest_required": bool(b.get("retest_required", True)),
            "retest_band_atr": retest_band_atr,

            # Window (bars) for retest detection (not in config yet; keep internal default)
            "retest_lookback_bars": int(self.clamp(self.safe_float(b.get("retest_lookback_bars", 5), 5), 2, 50)),

            # Optional proximity score gate for breakout (quality)
            "proximity_threshold_atr": self.clamp(self.safe_float(b.get("proximity_threshold_atr", 1.50), 1.50), 0.1, 10.0),
            "proximity_min_score": self.clamp(self.safe_float(b.get("proximity_min_score", 10.0), 10.0), 0.0, 100.0),

            # Use same indicator params as sideway defaults (stable)
            "adx_period": int(self.clamp(self.safe_float(s.get("adx_period", 14), 14), 5, 50)),
            "bb_period": int(self.clamp(self.safe_float(s.get("bb_period", 20), 20), 5, 200)),
            "bb_std": self.clamp(self.safe_float(s.get("bb_std", 2.0), 2.0), 0.5, 5.0),

            # Expansion gate: require BBWidth/ATR >= this threshold (reuse sideway max as boundary)
            "bb_width_atr_min": self.clamp(self.safe_float(s.get("bb_width_atr_max", 6.0), 6.0), 0.5, 50.0),

            # SL buffer ATR (hybrid): structure +/- buffer
            "sl_buffer_atr": self.clamp(self.safe_float(b.get("sl_buffer_atr", default_sl_buffer), default_sl_buffer), 0.0, 3.0),
        }

    # -----------------------------
    # Data fetching (placeholder)
    # -----------------------------
    def get_rates(self, symbol: str, tf: str, bars: int = 500) -> Dict[str, Any]:
        """
        Project-specific: in your repo this likely reads from MT5 adapter.
        Here assume implemented elsewhere and injected in runtime.
        """
        raise NotImplementedError("get_rates must be provided by integration layer")

    # -----------------------------
    # Signal generation
    # -----------------------------
    def generate_signal_package(self, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = self.cfg

        symbol = str(cfg.get("symbol", "GOLD"))
        mode = str(cfg.get("mode", "manual"))

        tfs = cfg.get("timeframes", {}) if isinstance(cfg.get("timeframes", {}), dict) else {}
        tf_htf = str(tfs.get("HTF", "H4"))
        tf_mtf = str(tfs.get("MTF", "M15"))
        tf_ltf = str(tfs.get("LTF", "M5"))

        sens = cfg.get("structure_sensitivity", {}) if isinstance(cfg.get("structure_sensitivity", {}), dict) else {}
        sens_htf = int(sens.get("HTF", 4))
        sens_mtf = int(sens.get("MTF", 3))
        sens_ltf = int(sens.get("LTF", 2))

        atr_cfg = cfg.get("atr", {}) if isinstance(cfg.get("atr", {}), dict) else {}
        atr_period = int(atr_cfg.get("period", 14))
        atr_sl_mult = float(atr_cfg.get("sl_mult", 1.5))

        st_cfg = cfg.get("supertrend", {}) if isinstance(cfg.get("supertrend", {}), dict) else {}
        st_period = int(st_cfg.get("period", 10))
        st_mult = float(st_cfg.get("mult", 3.0))

        min_rr = float(cfg.get("min_rr", 1.5))

        blocked_reasons: List[str] = []
        watch_state = "NONE"

        try:
            htf_data = self.get_rates(symbol, tf_htf, bars=500)
            mtf_data = self.get_rates(symbol, tf_mtf, bars=500)
            ltf_data = self.get_rates(symbol, tf_ltf, bars=500)
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

            # Proximity score: closeness to trigger level (normalized by ATR window)
            prox_window_points = float(k["proximity_window_atr"] * atr_val) if atr_val > 0 else 0.0

            def proximity(dist_abs: float) -> float:
                if prox_window_points <= 0:
                    return 0.0
                if not np.isfinite(dist_abs):
                    return 0.0
                if dist_abs >= prox_window_points:
                    return 0.0
                return float(1.0 - (dist_abs / prox_window_points))

            buy_score = proximity(d_buy_abs)
            sell_score = proximity(d_sell_abs)

            if buy_score >= sell_score:
                proximity_side = "BUY"
                proximity_score = buy_score
                proximity_best_dist = float(distance_buy) if np.isfinite(distance_buy) else float("nan")
            else:
                proximity_side = "SELL"
                proximity_score = sell_score
                proximity_best_dist = float(distance_sell) if np.isfinite(distance_sell) else float("nan")

            # Strict/Soft trigger logic
            allow_soft = bool(k["allow_soft_trigger"])
            require_confirm = bool(k["require_confirmation"])
            rsi_soft_band = float(k["rsi_soft_band"])

            hard_buy = bool(np.isfinite(distance_buy) and distance_buy <= 0.0)
            hard_sell = bool(np.isfinite(distance_sell) and distance_sell <= 0.0)

            soft_buy = bool(np.isfinite(distance_buy) and 0.0 < distance_buy <= near_points)
            soft_sell = bool(np.isfinite(distance_sell) and 0.0 < distance_sell <= near_points)

            rsi_buy_ok = bool(np.isfinite(rsi_val) and rsi_val <= (k["rsi_oversold"] + rsi_soft_band))
            rsi_sell_ok = bool(np.isfinite(rsi_val) and rsi_val >= (k["rsi_overbought"] - rsi_soft_band))

            # quality gate for soft triggers
            soft_quality_ok = bool(proximity_score >= float(k["proximity_score_min"]))

            buy_signal = False
            sell_signal = False

            if hard_buy:
                buy_signal = True
            elif allow_soft and soft_buy and rsi_buy_ok and soft_quality_ok:
                buy_signal = True
            if hard_sell:
                sell_signal = True
            elif allow_soft and soft_sell and rsi_sell_ok and soft_quality_ok:
                sell_signal = True

            if require_confirm:
                if buy_signal and not bullish_reversal:
                    buy_signal = False
                if sell_signal and not bearish_reversal:
                    sell_signal = False

            if buy_signal and not sell_signal:
                direction_out = "BUY"
            elif sell_signal and not buy_signal:
                direction_out = "SELL"
            else:
                # Either no signal or conflicting; fail-closed
                direction_out = "NONE"

            if direction_out in ("BUY", "SELL") and atr_val > 0:
                entry_candidate = float(ltf_close)

                # SL: ATR multiple
                if direction_out == "BUY":
                    stop_candidate = float(entry_candidate - (atr_sl_mult * atr_val))
                else:
                    stop_candidate = float(entry_candidate + (atr_sl_mult * atr_val))

                # --- RR floor safety (validator strict RR floor) ---
                rr_eps = 1e-6
                target_rr = float(min_rr) + rr_eps

                if direction_out == "BUY":
                    risk = float(entry_candidate - stop_candidate)
                    tp_candidate = float(entry_candidate + (risk * target_rr))
                    rr = float((tp_candidate - entry_candidate) / risk) if risk > 0 else 0.0
                else:
                    risk = float(stop_candidate - entry_candidate)
                    tp_candidate = float(entry_candidate - (risk * target_rr))
                    rr = float((entry_candidate - tp_candidate) / risk) if risk > 0 else 0.0

                # fail-closed RR check (epsilon)
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
            # --- Breakout v1.0 (BOS + confirm buffer + optional retest + supertrend confirm) ---
            k = self._get_breakout_knobs(cfg)

            # Trend strength (MTF ADX) + volatility expansion (LTF BBWidth/ATR)
            adx_arr = self.adx(mtf_high, mtf_low, mtf_close, k["adx_period"])
            adx_val = float(adx_arr[-1]) if (len(adx_arr) and np.isfinite(adx_arr[-1])) else float("nan")

            bb_u, bb_m, bb_l = self.bollinger(close, k["bb_period"], k["bb_std"])
            bb_upper = float(bb_u[-1]) if (len(bb_u) and np.isfinite(bb_u[-1])) else float("nan")
            bb_lower = float(bb_l[-1]) if (len(bb_l) and np.isfinite(bb_l[-1])) else float("nan")
            bb_width = float(bb_upper - bb_lower) if (np.isfinite(bb_upper) and np.isfinite(bb_lower)) else float("nan")
            bb_width_atr = float(bb_width / atr_val) if (atr_val > 0 and np.isfinite(bb_width)) else float("nan")

            # Gate: must have clear bias + supertrend confirmation (per requirement)
            if direction_bias not in ("BUY", "SELL"):
                blocked_reasons.append("no_clear_bias")
            if not bool(supertrend_ok):
                blocked_reasons.append("supertrend_conflict")

            # Gate: ADX must be finite (basic sanity; router already confirmed TREND with hysteresis)
            if not (np.isfinite(adx_val) and adx_val >= 10.0):
                blocked_reasons.append("bad_adx")

            # Gate: volatility expansion to reduce fake breaks
            if not (np.isfinite(bb_width_atr) and bb_width_atr >= k["bb_width_atr_min"]):
                blocked_reasons.append("no_vol_expansion")

            # BOS reference levels from structure()
            bos_ref_high = float(bos_hi) if np.isfinite(bos_hi) else float("nan")
            bos_ref_low = float(bos_lo) if np.isfinite(bos_lo) else float("nan")

            confirm_buf = float(k["confirm_buffer_atr"] * atr_val) if atr_val > 0 else 0.0
            retest_band = float(k["retest_band_atr"] * atr_val) if atr_val > 0 else 0.0
            sl_buf = float(k["sl_buffer_atr"] * atr_val) if atr_val > 0 else 0.0

            lookback = int(k["retest_lookback_bars"])
            lookback = min(max(2, lookback), len(close))

            prox_score = 0.0

            if not blocked_reasons:
                if direction_bias == "BUY":
                    if not np.isfinite(bos_ref_high) or not np.isfinite(bos_ref_low):
                        blocked_reasons.append("no_bos_refs")
                    else:
                        level = bos_ref_high
                        confirm_level = level + confirm_buf
                        broke = bool(ltf_close > confirm_level)

                        if not broke:
                            blocked_reasons.append("no_bos_break")
                        else:
                            if k["retest_required"]:
                                recent_low = float(np.nanmin(low[-lookback:]))
                                # retest: dipped near breakout level but still closes above it
                                if not (np.isfinite(recent_low) and recent_low <= (level + retest_band) and ltf_close > level):
                                    blocked_reasons.append("no_retest")

                            dist_atr = float(abs(ltf_close - level) / atr_val) if atr_val > 0 else float("inf")
                            if np.isfinite(dist_atr) and dist_atr >= 0:
                                th = float(k["proximity_threshold_atr"])
                                prox_score = float(max(0.0, (th - dist_atr) / th) * 100.0) if th > 0 else 0.0
                            if prox_score < float(k["proximity_min_score"]):
                                blocked_reasons.append("low_proximity_score")

                            if not blocked_reasons:
                                direction_out = "BUY"
                                entry_candidate = float(ltf_close)

                                # Hybrid SL: structure invalidation (bos_ref_low) + ATR buffer
                                sl_base = bos_ref_low
                                if not np.isfinite(sl_base):
                                    sl_base = float(entry_candidate - (atr_sl_mult * atr_val))
                                stop_candidate = float(sl_base - sl_buf)

                                # TP from RR (fail-closed)
                                risk = float(entry_candidate - stop_candidate)
                                if risk <= 0:
                                    blocked_reasons.append("bad_risk")
                                else:
                                    rr_eps = 1e-6
                                    target_rr = float(min_rr) + rr_eps
                                    tp_candidate = float(entry_candidate + (risk * target_rr))
                                    rr = float((tp_candidate - entry_candidate) / risk)

                elif direction_bias == "SELL":
                    if not np.isfinite(bos_ref_low) or not np.isfinite(bos_ref_high):
                        blocked_reasons.append("no_bos_refs")
                    else:
                        level = bos_ref_low
                        confirm_level = level - confirm_buf
                        broke = bool(ltf_close < confirm_level)

                        if not broke:
                            blocked_reasons.append("no_bos_break")
                        else:
                            if k["retest_required"]:
                                recent_high = float(np.nanmax(high[-lookback:]))
                                # retest: spiked near breakout level but still closes below it
                                if not (np.isfinite(recent_high) and recent_high >= (level - retest_band) and ltf_close < level):
                                    blocked_reasons.append("no_retest")

                            dist_atr = float(abs(ltf_close - level) / atr_val) if atr_val > 0 else float("inf")
                            if np.isfinite(dist_atr) and dist_atr >= 0:
                                th = float(k["proximity_threshold_atr"])
                                prox_score = float(max(0.0, (th - dist_atr) / th) * 100.0) if th > 0 else 0.0
                            if prox_score < float(k["proximity_min_score"]):
                                blocked_reasons.append("low_proximity_score")

                            if not blocked_reasons:
                                direction_out = "SELL"
                                entry_candidate = float(ltf_close)

                                # Hybrid SL: structure invalidation (bos_ref_high) + ATR buffer
                                sl_base = bos_ref_high
                                if not np.isfinite(sl_base):
                                    sl_base = float(entry_candidate + (atr_sl_mult * atr_val))
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
                "confirm_buffer_atr": k["confirm_buffer_atr"],
                "retest_required": bool(k["retest_required"]),
                "retest_band_atr": k["retest_band_atr"],
                "retest_lookback_bars": k["retest_lookback_bars"],
                "bb_width_atr_min": k["bb_width_atr_min"],
                "sl_buffer_atr": k["sl_buffer_atr"],
                "proximity_score": prox_score,
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
            "bos": bool(direction_out in ("BUY", "SELL")),
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