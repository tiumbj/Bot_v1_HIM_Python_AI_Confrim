"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 2.8.0

Changelog:
- 2.8.0 (2026-02-26):
  - ADD: Intelligent Dashboard knobs (breakout.*) to control strategy by mode buttons:
      breakout.confirm_buffer_atr
      breakout.require_retest
      breakout.retest_band_atr
      breakout.proximity_threshold_atr
      breakout.proximity_min_score
  - FIX: BOS confirmation uses buffer ATR (optional) instead of strict pivot break only
  - FIX: Proximity threshold now supports ATR-based default via breakout.proximity_threshold_atr
  - ENFORCE: Option C completeness (if require_retest=True -> block when bos=True but retest_ok=False)
  - COMPAT: fallback to legacy keys:
      breakout_proximity.{threshold_atr, threshold, min_score}
      entry_model.retest_atr_mult
      supertrend.{multiplier,mult}
  - KEEP: Real SuperTrend (ATR-based) on LTF
  - KEEP: counter_breakout block + watch_state
  - KEEP: contract gating (blocked -> direction NONE, candidates None)

SuperTrend (ศัพท์):
- ATR (Average True Range) = ค่าแกว่งตัวเฉลี่ย
- SuperTrend = เส้นตามเทรนด์ (ATR*multiplier) ใช้บอก bullish/bearish
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import MetaTrader5 as mt5

from config_resolver import resolve_effective_config


class TradingEngine:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.cfg = self.load_config()

    # -------------------------
    # Safe utilities
    # -------------------------
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

    # -------------------------
    # Config
    # -------------------------
    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict):
                return {}
            return resolve_effective_config(cfg)
        except Exception:
            return {}

    def reload_config(self) -> Dict[str, Any]:
        self.cfg = self.load_config()
        return self.cfg

    # -------------------------
    # MT5 + Data
    # -------------------------
    def ensure_mt5(self) -> Tuple[bool, str]:
        try:
            if mt5.terminal_info() is not None:
                return True, "already_initialized"
            ok = mt5.initialize()
            if not ok:
                return False, f"initialize_failed: {mt5.last_error()}"
            return True, "initialized"
        except Exception as e:
            return False, f"initialize_exception: {e}"

    def get_tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"tick_none: {symbol} last_error={mt5.last_error()}")
        return tick

    def tf(self, name: str):
        mapping = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        key = str(name).upper().strip()
        return mapping.get(key, mt5.TIMEFRAME_M15)

    def get_data(self, symbol: str, timeframe, bars: int):
        ok, msg = self.ensure_mt5()
        if not ok:
            raise RuntimeError(msg)

        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select_failed: {symbol} last_error={mt5.last_error()}")

        bars = int(bars)

        def fetch_once():
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        rates = fetch_once()
        if rates is None or len(rates) == 0:
            time.sleep(0.2)
            rates = fetch_once()

        if rates is None:
            raise RuntimeError(
                f"copy_rates_none: symbol={symbol} tf={timeframe} bars={bars} last_error={mt5.last_error()}"
            )
        if len(rates) < 200:
            raise RuntimeError(f"not_enough_bars: got={len(rates)} need>=200 symbol={symbol} tf={timeframe}")
        return rates

    # -------------------------
    # ATR (EMA-style)
    # -------------------------
    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        period = max(1, int(period))
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        out = np.zeros(len(close), dtype=float)
        alpha = 1.0 / period
        for i in range(1, len(close)):
            out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
        return out

    # -------------------------
    # SuperTrend
    # -------------------------
    def supertrend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float):
        """
        Returns:
        - st: SuperTrend line values
        - dir_arr: +1 bullish, -1 bearish
        """
        n = len(close)
        period = max(1, int(period))
        mult = float(max(mult, 0.1))

        atr = self.atr(high, low, close, period)
        hl2 = (high + low) / 2.0

        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        f_upper = np.copy(upper)
        f_lower = np.copy(lower)

        dir_arr = np.ones(n, dtype=int)  # start bullish
        st = np.zeros(n, dtype=float)

        for i in range(1, n):
            if upper[i] < f_upper[i - 1] or close[i - 1] > f_upper[i - 1]:
                f_upper[i] = upper[i]
            else:
                f_upper[i] = f_upper[i - 1]

            if lower[i] > f_lower[i - 1] or close[i - 1] < f_lower[i - 1]:
                f_lower[i] = lower[i]
            else:
                f_lower[i] = f_lower[i - 1]

            if dir_arr[i - 1] == 1:
                dir_arr[i] = -1 if close[i] < f_lower[i] else 1
            else:
                dir_arr[i] = 1 if close[i] > f_upper[i] else -1

            st[i] = f_lower[i] if dir_arr[i] == 1 else f_upper[i]

        st[0] = hl2[0]
        return st, dir_arr

    # -------------------------
    # STRUCTURE (pivots)
    # -------------------------
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

    # -------------------------
    # Proximity Score
    # -------------------------
    def proximity_score(self, dist_buy: Optional[float], dist_sell: Optional[float], threshold_points: float):
        thr = max(self.safe_float(threshold_points, 0.0), 1e-9)
        best: Optional[float] = None
        side: Optional[str] = None

        if dist_buy is not None:
            try:
                db = float(dist_buy)
                if db >= 0:
                    best = db
                    side = "BUY"
            except Exception:
                pass

        if dist_sell is not None:
            try:
                ds = float(dist_sell)
                if ds >= 0:
                    if best is None or ds < best:
                        best = ds
                        side = "SELL"
            except Exception:
                pass

        if best is None:
            return 0.0, None, None

        score = 100.0 * max(0.0, min(1.0, (thr - best) / thr))
        return float(score), side, float(best)

    # -------------------------
    # Score + Confidence
    # -------------------------
    def compute_score_0_10(
        self,
        direction: str,
        htf_trend: str,
        bos: bool,
        prox_score_0_100: float,
        retest_ok: bool,
        overshoot_points: Optional[float],
        threshold_points: float,
        blocked: bool,
    ) -> float:
        s = 0.0

        align = 0.0
        if direction == "BUY" and htf_trend == "bullish":
            align = 1.0
        elif direction == "SELL" and htf_trend == "bearish":
            align = 1.0
        s += 2.0 * align

        s += 2.5 if bos else 0.0

        ps = self.clamp(self.safe_float(prox_score_0_100, 0.0) / 100.0, 0.0, 1.0)
        s += 2.0 * ps

        s += 1.5 if retest_ok else 0.0

        og = 0.0
        if overshoot_points is not None:
            thr = max(self.safe_float(threshold_points, 0.0), 1e-9)
            ratio = self.clamp(float(overshoot_points) / thr, 0.0, 2.0)
            og = 1.0 - self.clamp(ratio / 2.0, 0.0, 1.0)
        s += 1.0 * og

        if blocked:
            s -= 3.0

        return self.clamp(s, 0.0, 10.0)

    def compute_confidence_0_100(self, score_0_10: float, bos: bool, retest_ok: bool, align: bool) -> int:
        score100 = self.clamp(score_0_10 * 10.0, 0.0, 100.0)
        base = score100 * 0.85
        bonus = 0.0
        if bos:
            bonus += 8.0
        if retest_ok:
            bonus += 5.0
        if align:
            bonus += 4.0
        return int(round(self.clamp(base + bonus, 0.0, 100.0)))

    # -------------------------
    # Config access helpers (compat)
    # -------------------------
    def _get_breakout_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Priority: breakout.* (new) -> legacy breakout_proximity / entry_model fallbacks
        """
        breakout = cfg.get("breakout", {}) or {}
        prox_legacy = cfg.get("breakout_proximity", {}) or {}
        entry_legacy = cfg.get("entry_model", {}) or {}

        confirm_buffer_atr = self.safe_float(breakout.get("confirm_buffer_atr"), 0.0)

        require_retest = breakout.get("require_retest")
        if require_retest is None:
            require_retest = True
        require_retest = bool(require_retest)

        retest_band_atr = breakout.get("retest_band_atr")
        if retest_band_atr is None:
            # legacy name in old engine
            retest_band_atr = entry_legacy.get("retest_atr_mult")
        retest_band_atr = self.safe_float(retest_band_atr, 0.25)

        # proximity knobs
        prox_thr_atr = breakout.get("proximity_threshold_atr")
        if prox_thr_atr is None:
            prox_thr_atr = prox_legacy.get("threshold_atr")
        prox_thr_atr = self.safe_float(prox_thr_atr, 0.8)

        prox_min_score = breakout.get("proximity_min_score")
        if prox_min_score is None:
            prox_min_score = prox_legacy.get("min_score")
        prox_min_score = self.safe_float(prox_min_score, 50.0)

        # legacy absolute points fallback (rare)
        prox_thr_points = prox_legacy.get("threshold")
        prox_thr_points = self.safe_float(prox_thr_points, 0.0)

        return {
            "confirm_buffer_atr": float(max(confirm_buffer_atr, 0.0)),
            "require_retest": bool(require_retest),
            "retest_band_atr": float(max(retest_band_atr, 0.0)),
            "proximity_threshold_atr": float(max(prox_thr_atr, 0.0)),
            "proximity_min_score": float(self.clamp(prox_min_score, 0.0, 100.0)),
            "legacy_threshold_points": float(max(prox_thr_points, 0.0)),
        }

    # -------------------------
    # MAIN
    # -------------------------
    def generate_signal_package(self) -> Dict[str, Any]:
        cfg = self.reload_config()

        symbol = str(cfg.get("symbol", "GOLD"))
        tf_cfg = cfg.get("timeframes", {}) or {}
        sens_cfg = cfg.get("structure_sensitivity", {}) or {}
        risk_cfg = cfg.get("risk", {}) or {}
        st_cfg = cfg.get("supertrend", {}) or {}

        knobs = self._get_breakout_knobs(cfg)

        min_rr = self.safe_float(cfg.get("min_rr", 2.0), 2.0)
        atr_sl_mult = self.safe_float(risk_cfg.get("atr_sl_mult", 1.8), 1.8)

        htf = str(tf_cfg.get("htf", "H4"))
        mtf = str(tf_cfg.get("mtf", "H1"))
        ltf = str(tf_cfg.get("ltf", "M15"))

        sens_htf = self.safe_int(sens_cfg.get("htf"), 5)
        sens_mtf = self.safe_int(sens_cfg.get("mtf"), 4)
        sens_ltf = self.safe_int(sens_cfg.get("ltf"), 3)

        atr_period = self.safe_int(risk_cfg.get("atr_period", 14), 14)

        # SuperTrend params (compat: multiplier/mult)
        st_period = self.safe_int(st_cfg.get("period", 10), 10)
        st_mult = st_cfg.get("multiplier", None)
        if st_mult is None:
            st_mult = st_cfg.get("mult", None)
        st_mult = self.safe_float(st_mult, 3.0)

        blocked_reasons = []
        watch_state = "NONE"

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
                "context": {
                    "blocked_by": "no_data",
                    "error": str(e),
                    "timeframes": {"htf": htf, "mtf": mtf, "ltf": ltf},
                },
            }

        htf_trend, _, _ = self.structure(htf_data, sens_htf)
        mtf_trend, _, _ = self.structure(mtf_data, sens_mtf)
        ltf_trend, bos_hi, bos_lo = self.structure(ltf_data, sens_ltf)

        # Bias (owner: engine)
        direction = "NONE"
        bias_source = "NO_CLEAR_BIAS"
        if htf_trend == "bullish":
            direction = "BUY"
            bias_source = "HTF"
        elif htf_trend == "bearish":
            direction = "SELL"
            bias_source = "HTF"
        elif mtf_trend == "bullish":
            direction = "BUY"
            bias_source = "MTF_FALLBACK"
        elif mtf_trend == "bearish":
            direction = "SELL"
            bias_source = "MTF_FALLBACK"
        else:
            blocked_reasons.append("no_clear_bias")

        close = np.asarray(ltf_data["close"], dtype=float)
        high = np.asarray(ltf_data["high"], dtype=float)
        low = np.asarray(ltf_data["low"], dtype=float)
        ltf_close = float(close[-1])

        atr_arr = self.atr(high, low, close, atr_period)
        atr_val = float(atr_arr[-1]) if len(atr_arr) else 0.0

        # SuperTrend (REAL)
        st_line, st_dir_arr = self.supertrend(high, low, close, st_period, st_mult)
        st_dir = "bullish" if int(st_dir_arr[-1]) == 1 else "bearish"
        st_value = float(st_line[-1])

        supertrend_ok = False
        if direction == "BUY" and st_dir == "bullish":
            supertrend_ok = True
        elif direction == "SELL" and st_dir == "bearish":
            supertrend_ok = True

        if direction in ("BUY", "SELL") and not supertrend_ok:
            blocked_reasons.append("supertrend_conflict")

        # Proximity threshold points:
        # - Prefer ATR-based breakout.proximity_threshold_atr
        # - If ATR not available or knob=0, fallback legacy absolute points if provided
        threshold_mode = "ATR"
        threshold_atr_out = None
        if atr_val > 0 and knobs["proximity_threshold_atr"] > 0:
            threshold_points = float(atr_val * float(knobs["proximity_threshold_atr"]))
            threshold_atr_out = float(knobs["proximity_threshold_atr"])
        else:
            threshold_points = float(max(knobs["legacy_threshold_points"], 2.5))
            threshold_mode = "POINTS"
        threshold_points = max(float(threshold_points), 1e-9)

        # Distances to structure refs
        dist_buy = (float(bos_hi) - ltf_close) if bos_hi is not None else None
        dist_sell = (ltf_close - float(bos_lo)) if bos_lo is not None else None

        # Breakout confirm with ATR buffer:
        # BUY breakout when close > bos_hi + buffer
        # SELL breakout when close < bos_lo - buffer
        confirm_buffer_points = float(max(atr_val * knobs["confirm_buffer_atr"], 0.0)) if atr_val > 0 else 0.0

        breakout_state = "NONE"
        breakout_side = None
        breakout_overshoot_points = None

        # Overshoot measured beyond pivot+buffer (>=0 means breakout)
        buy_overshoot = None
        sell_overshoot = None

        if bos_hi is not None:
            buy_overshoot = float(ltf_close - (float(bos_hi) + confirm_buffer_points))
        if bos_lo is not None:
            sell_overshoot = float((float(bos_lo) - confirm_buffer_points) - ltf_close)

        if buy_overshoot is not None and buy_overshoot > 0:
            breakout_state = "BREAKOUT_BUY_CONFIRMED"
            breakout_side = "BUY"
            breakout_overshoot_points = float(buy_overshoot)
        elif sell_overshoot is not None and sell_overshoot > 0:
            breakout_state = "BREAKOUT_SELL_CONFIRMED"
            breakout_side = "SELL"
            breakout_overshoot_points = float(sell_overshoot)

        # Conflict block: opposite breakout exists vs bias direction
        if breakout_side in ("BUY", "SELL") and direction in ("BUY", "SELL") and breakout_side != direction:
            blocked_reasons.append("counter_breakout")
            watch_state = "WATCH_COUNTER_BREAKOUT"

        bos = bool(breakout_side == direction and breakout_state.startswith("BREAKOUT_"))

        # Proximity score (distance to pivot refs, not buffer)
        prox_score, prox_side, prox_best_dist = self.proximity_score(dist_buy, dist_sell, threshold_points)
        prox_min_score = float(knobs["proximity_min_score"])
        if watch_state == "NONE" and prox_score >= prox_min_score and prox_side in ("BUY", "SELL"):
            watch_state = "WATCH_PROXIMITY"

        # Retest reference
        bos_ref = None
        if direction == "BUY":
            bos_ref = bos_hi
        elif direction == "SELL":
            bos_ref = bos_lo

        # Retest band points (ATR based)
        retest_band = float(max(atr_val * knobs["retest_band_atr"], 1e-9)) if atr_val > 0 else 1e-9
        retest_ok = False
        if bos_ref is not None and direction in ("BUY", "SELL"):
            retest_ok = bool(abs(ltf_close - float(bos_ref)) <= retest_band)

        # -------------------------
        # Option C enforcement (BOS + Proximity + Retest)
        # -------------------------
        if direction not in ("BUY", "SELL"):
            blocked_reasons.append("direction_none")

        # BOS required
        if direction in ("BUY", "SELL") and not bos:
            blocked_reasons.append("no_bos")

        # Proximity required (must be near structure in the direction context)
        # - enforce prox_score >= prox_min_score and prox_side aligns to direction
        if direction in ("BUY", "SELL"):
            if not (prox_side == direction and prox_score >= prox_min_score):
                blocked_reasons.append("no_proximity")

        # Retest required (if configured)
        if direction in ("BUY", "SELL") and knobs["require_retest"]:
            if bos and not retest_ok:
                blocked_reasons.append("no_retest")

        blocked = bool(len(blocked_reasons) > 0)

        align = bool((direction == "BUY" and htf_trend == "bullish") or (direction == "SELL" and htf_trend == "bearish"))
        score_0_10 = self.compute_score_0_10(
            direction=direction,
            htf_trend=htf_trend,
            bos=bos,
            prox_score_0_100=prox_score,
            retest_ok=retest_ok,
            overshoot_points=breakout_overshoot_points,
            threshold_points=threshold_points,
            blocked=blocked,
        )
        confidence_py = self.compute_confidence_0_100(score_0_10, bos=bos, retest_ok=retest_ok, align=align)

        if blocked:
            direction_out = "NONE"
            entry_candidate = None
            stop_candidate = None
            tp_candidate = None
            rr = 0.0
        else:
            direction_out = direction
            tick = self.get_tick(symbol)
            entry_candidate = float(tick.ask if direction_out == "BUY" else tick.bid)

            sl_dist = float(max(atr_val * atr_sl_mult, 1e-9))
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

        return {
            "symbol": symbol,
            "direction": direction_out,
            "entry_candidate": entry_candidate,
            "stop_candidate": stop_candidate,
            "tp_candidate": tp_candidate,
            "rr": float(rr),
            "score": float(score_0_10),
            "confidence_py": int(confidence_py),
            "bos": bool(bos),
            "supertrend_ok": bool(supertrend_ok),
            "context": {
                "engine_version": "2.8.0",
                "HTF_trend": htf_trend,
                "MTF_trend": mtf_trend,
                "LTF_trend": ltf_trend,
                "bias_source": bias_source,
                "direction_bias": direction,

                "bos_ref_high": bos_hi,
                "bos_ref_low": bos_lo,
                "bos_ref": bos_ref,

                "distance_buy": dist_buy,
                "distance_sell": dist_sell,

                "atr": atr_val,
                "atr_period": atr_period,
                "atr_sl_mult": atr_sl_mult,
                "min_rr": min_rr,

                "breakout_confirm_buffer_atr": knobs["confirm_buffer_atr"],
                "breakout_confirm_buffer_points": confirm_buffer_points,
                "breakout_proximity_min_score": prox_min_score,
                "breakout_proximity_threshold_atr": threshold_atr_out,
                "breakout_proximity_threshold_mode": threshold_mode,
                "breakout_proximity_threshold_points": threshold_points,
                "proximity_score": prox_score,
                "proximity_side": prox_side,
                "proximity_best_dist": prox_best_dist,

                "breakout_state": breakout_state,
                "breakout_side": breakout_side,
                "breakout_overshoot_points": breakout_overshoot_points,

                "retest_required": bool(knobs["require_retest"]),
                "retest_band_atr": knobs["retest_band_atr"],
                "retest_band": retest_band,
                "retest_ok": retest_ok,

                "supertrend_period": st_period,
                "supertrend_mult": st_mult,
                "supertrend_dir": st_dir,
                "supertrend_value": st_value,
                "supertrend_ok": bool(supertrend_ok),

                "watch_state": watch_state,
                "blocked_by": ",".join(blocked_reasons) if blocked_reasons else None,

                "timeframes": {"htf": htf, "mtf": mtf, "ltf": ltf},
                "structure_sensitivity": {"htf": sens_htf, "mtf": sens_mtf, "ltf": sens_ltf},
            },
        }


if __name__ == "__main__":
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        raise SystemExit(1)

    e = TradingEngine("config.json")
    pkg = e.generate_signal_package()
    print(json.dumps(pkg, indent=2, ensure_ascii=False))