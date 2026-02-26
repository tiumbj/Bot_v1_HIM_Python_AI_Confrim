"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 2.6.1

Changelog:
- 2.6.1 (2026-02-26):
  - P0 FIX: proximity_score() no longer calls safe_float(..., None) which could crash (float(None))
  - Use MT5 tick ask/bid for entry_candidate (market execution pricing owner)
  - Keep v2.6.0 contract fields and config_resolver integration

Design Rules (Commissioning):
- Engine is the pricing owner: computes entry/SL/TP candidates
- AI is confirm-only (must not change prices) [enforced elsewhere]
- Executor enforces safety + risk (later steps)
- supertrend_ok remains placeholder True (explicit)
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

    # =========================
    # SAFE CONVERSION
    # =========================
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

    # =========================
    # CONFIG
    # =========================
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

    # =========================
    # MT5 READY
    # =========================
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

    # =========================
    # TF MAP
    # =========================
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

    # =========================
    # DATA
    # =========================
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

    # =========================
    # ATR (EMA-style)
    # =========================
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

    # =========================
    # STRUCTURE (pivots)
    # =========================
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

    # =========================
    # PROXIMITY (P0 FIX)
    # =========================
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

    # =========================
    # SCORING (0-10) + CONF (0-100)
    # =========================
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

        # Trend alignment (HTF only in this version)
        align = 0.0
        if direction == "BUY" and htf_trend == "bullish":
            align = 1.0
        elif direction == "SELL" and htf_trend == "bearish":
            align = 1.0
        s += 2.0 * align

        # BOS
        s += 2.5 if bos else 0.0

        # Proximity
        ps = self.clamp(self.safe_float(prox_score_0_100, 0.0) / 100.0, 0.0, 1.0)
        s += 2.0 * ps

        # Retest
        s += 1.5 if retest_ok else 0.0

        # Overshoot grading
        og = 0.0
        if overshoot_points is not None:
            thr = max(self.safe_float(threshold_points, 0.0), 1e-9)
            ratio = self.clamp(float(overshoot_points) / thr, 0.0, 2.0)
            og = 1.0 - self.clamp(ratio / 2.0, 0.0, 1.0)
        s += 1.0 * og

        # Penalty
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

    # =========================
    # MAIN
    # =========================
    def generate_signal_package(self) -> Dict[str, Any]:
        cfg = self.reload_config()

        symbol = str(cfg.get("symbol", "GOLD"))
        tf_cfg = cfg.get("timeframes", {}) or {}
        sens_cfg = cfg.get("structure_sensitivity", {}) or {}
        risk_cfg = cfg.get("risk", {}) or {}
        prox_cfg = cfg.get("breakout_proximity", {}) or {}

        entry_cfg = cfg.get("entry_model", {}) or {}
        retest_atr_mult = self.safe_float(entry_cfg.get("retest_atr_mult"), 0.35)

        min_rr = self.safe_float(cfg.get("min_rr", 1.7), 1.7)
        atr_sl_mult = self.safe_float(risk_cfg.get("atr_sl_mult"), 1.8)

        htf = str(tf_cfg.get("htf", "H1"))
        mtf = str(tf_cfg.get("mtf", "M30"))
        ltf = str(tf_cfg.get("ltf", "M15"))

        sens_htf = self.safe_int(sens_cfg.get("htf"), 4)
        sens_mtf = self.safe_int(sens_cfg.get("mtf"), 3)
        sens_ltf = self.safe_int(sens_cfg.get("ltf"), 1)

        atr_period = self.safe_int(risk_cfg.get("atr_period"), 14)
        prox_min_score = self.safe_float(prox_cfg.get("min_score"), 40.0)

        threshold_atr_raw = prox_cfg.get("threshold_atr", None)
        threshold_points_fallback = prox_cfg.get("threshold", None)

        blocked_reasons = []

        try:
            htf_data = self.get_data(symbol, self.tf(htf), 600)
            mtf_data = self.get_data(symbol, self.tf(mtf), 800)
            ltf_data = self.get_data(symbol, self.tf(ltf), 1200)
        except Exception as e:
            return {
                "symbol": symbol,
                "direction": "NONE",
                "score": 0.0,
                "rr": 0.0,
                "confidence_py": 0,
                "bos": False,
                "supertrend_ok": True,
                "entry_candidate": 0.0,
                "stop_candidate": 0.0,
                "tp_candidate": 0.0,
                "context": {
                    "blocked_by": "no_data",
                    "error": str(e),
                    "timeframes": {"htf": htf, "mtf": mtf, "ltf": ltf},
                },
            }

        htf_trend, _, _ = self.structure(htf_data, sens_htf)
        mtf_trend, _, _ = self.structure(mtf_data, sens_mtf)
        ltf_trend, bos_hi, bos_lo = self.structure(ltf_data, sens_ltf)

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

        thr_atr = self.safe_float(threshold_atr_raw, default=0.0)
        if thr_atr > 0 and atr_val > 0:
            threshold_points = float(atr_val * thr_atr)
            threshold_mode = "ATR"
            threshold_atr_out = thr_atr
        else:
            threshold_points = self.safe_float(threshold_points_fallback, default=2.5)
            threshold_mode = "POINTS"
            threshold_atr_out = None
        threshold_points = max(float(threshold_points), 1e-9)

        dist_buy = (float(bos_hi) - ltf_close) if bos_hi is not None else None
        dist_sell = (ltf_close - float(bos_lo)) if bos_lo is not None else None

        breakout_state = "NONE"
        breakout_side = None
        breakout_overshoot_points = None

        if dist_buy is not None and dist_buy < 0:
            breakout_state = "BREAKOUT_BUY_CONFIRMED"
            breakout_side = "BUY"
            breakout_overshoot_points = float(abs(dist_buy))
        elif dist_sell is not None and dist_sell < 0:
            breakout_state = "BREAKOUT_SELL_CONFIRMED"
            breakout_side = "SELL"
            breakout_overshoot_points = float(abs(dist_sell))

        bos = bool(breakout_side == direction and breakout_state.startswith("BREAKOUT_"))

        prox_score, prox_side, prox_best_dist = self.proximity_score(dist_buy, dist_sell, threshold_points)

        bos_ref = None
        if direction == "BUY":
            bos_ref = bos_hi
        elif direction == "SELL":
            bos_ref = bos_lo

        retest_band = float(max(atr_val * retest_atr_mult, 1e-9))
        retest_ok = False
        if bos_ref is not None and direction in ("BUY", "SELL"):
            retest_ok = bool(abs(ltf_close - float(bos_ref)) <= retest_band)

        if direction == "BUY":
            if bos_hi is None:
                blocked_reasons.append("no_bos_ref_high")
            if dist_buy is not None and dist_buy >= 0 and dist_buy > threshold_points:
                blocked_reasons.append("no_bos")
        elif direction == "SELL":
            if bos_lo is None:
                blocked_reasons.append("no_bos_ref_low")
            if dist_sell is not None and dist_sell >= 0 and dist_sell > threshold_points:
                blocked_reasons.append("no_bos")

        blocked = bool(len(blocked_reasons) > 0)

        # -------------------------
        # Trade-ready candidates (pricing owner) using tick
        # -------------------------
        tick = self.get_tick(symbol)

        if direction == "BUY":
            entry_candidate = float(tick.ask)
        elif direction == "SELL":
            entry_candidate = float(tick.bid)
        else:
            entry_candidate = float(ltf_close)

        sl_dist = float(max(atr_val * atr_sl_mult, 1e-9))

        if direction == "BUY":
            stop_candidate = float(entry_candidate - sl_dist)
            tp_candidate = float(entry_candidate + sl_dist * float(max(min_rr, 0.1)))
        elif direction == "SELL":
            stop_candidate = float(entry_candidate + sl_dist)
            tp_candidate = float(entry_candidate - sl_dist * float(max(min_rr, 0.1)))
        else:
            stop_candidate = 0.0
            tp_candidate = 0.0

        rr = 0.0
        if direction == "BUY":
            risk = entry_candidate - stop_candidate
            reward = tp_candidate - entry_candidate
            rr = float(reward / risk) if risk > 1e-9 else 0.0
        elif direction == "SELL":
            risk = stop_candidate - entry_candidate
            reward = entry_candidate - tp_candidate
            rr = float(reward / risk) if risk > 1e-9 else 0.0

        align = bool((direction == "BUY" and htf_trend == "bullish") or (direction == "SELL" and htf_trend == "bearish"))

        score_0_10 = self.compute_score_0_10(
            direction=direction if direction in ("BUY", "SELL") else "NONE",
            htf_trend=htf_trend,
            bos=bos,
            prox_score_0_100=prox_score,
            retest_ok=retest_ok,
            overshoot_points=breakout_overshoot_points,
            threshold_points=threshold_points,
            blocked=(direction not in ("BUY", "SELL")) or blocked,
        )

        confidence_py = self.compute_confidence_0_100(score_0_10, bos=bos, retest_ok=retest_ok, align=align)

        supertrend_ok = True
        supertrend_label = "PLACEHOLDER"

        direction_out = direction if direction in ("BUY", "SELL") else "NONE"

        return {
            "symbol": symbol,
            "direction": direction_out,
            "entry_candidate": float(entry_candidate),
            "stop_candidate": float(stop_candidate),
            "tp_candidate": float(tp_candidate),
            "rr": float(rr),
            "score": float(score_0_10),
            "confidence_py": int(confidence_py),
            "bos": bool(bos),
            "supertrend_ok": bool(supertrend_ok),
            "context": {
                "HTF_trend": htf_trend,
                "MTF_trend": mtf_trend,
                "LTF_trend": ltf_trend,
                "bias_source": bias_source,

                "bos_ref_high": bos_hi,
                "bos_ref_low": bos_lo,
                "bos_ref": bos_ref,

                "distance_buy": dist_buy,
                "distance_sell": dist_sell,

                "atr": atr_val,
                "atr_period": atr_period,
                "atr_sl_mult": atr_sl_mult,
                "min_rr": min_rr,

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

                "retest_atr_mult": retest_atr_mult,
                "retest_band": retest_band,
                "retest_ok": retest_ok,

                "bos": bool(bos),
                "supertrend": supertrend_label,
                "supertrend_ok": bool(supertrend_ok),

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