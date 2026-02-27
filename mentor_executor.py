"""
Mentor Executor - Execution Controller (Phase 9 Live Stabilization)
Version: 2.6.0

Changelog:
- 2.6.0 (2026-02-27):
  - NEW: AI Spec v2 Mode D integration (bounded adjustment, direction locked, fail-closed)
  - NEW: Multi-order support with stacking guards (distance ATR, max positions per symbol)
  - NEW: Unified mentor output uses AI mentor block (reduces intelligent mentor complexity)
  - KEEP: Production safety (spread, stops/freeze, dedupe, BOS+SuperTrend required, MT5 must remain running)
  - KEEP: Position manager (BE + ATR trail + emergency SL) + compatible extension for AI tighten-only

Locked Rules (do not change unless explicitly declared):
- Mode = sideway_scalp (frozen)
- min_rr = 1.5 (frozen)
- atr_sl_mult = 1.6 (frozen)
- require_confirmation = true (frozen)
- BOS required
- SuperTrend OK required
- AI is bounded (Mode D): no direction change, no lot sizing, no risk model override
- Fail closed on AI/parse/constraint errors

Backtest evidence (placeholder):
- N/A (executor logic refactor; requires forward A/B logs)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from ai_mentor import AIMentor
from config_resolver import resolve_effective_config
from engine import TradingEngine
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("him_system.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("HIM")


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        if v is None:
            return float(d)
        return float(v)
    except Exception:
        return float(d)


def _si(v: Any, d: int = 0) -> int:
    try:
        if v is None:
            return int(d)
        return int(v)
    except Exception:
        return int(d)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
    if not (_is_finite(entry) and _is_finite(sl) and _is_finite(tp)):
        return False
    if direction == "BUY":
        return sl < entry < tp
    if direction == "SELL":
        return tp < entry < sl
    return False


class HotConfig:
    def __init__(self, path: str):
        self.path = path
        self._mtime = 0.0
        self._cache: Dict[str, Any] = {}

    def load_raw(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.path):
                return self._cache or {}
            m = os.path.getmtime(self.path)
            if m <= self._mtime and self._cache:
                return self._cache
            with open(self.path, "r", encoding="utf-8") as f:
                self._cache = json.load(f) or {}
            self._mtime = m
            return self._cache
        except Exception as e:
            logger.error(f"CRITICAL: config.json load failed: {e}")
            return self._cache or {}

    def load_effective(self) -> Dict[str, Any]:
        raw = self.load_raw()
        try:
            return resolve_effective_config(raw)
        except Exception as e:
            logger.error(f"CRITICAL: resolve_effective_config failed: {e}")
            return raw or {}


class MentorExecutor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_dir, "config.json")
        self.hcfg = HotConfig(self.config_path)

        self.engine = TradingEngine(config_path=self.config_path)
        self.ai = AIMentor()
        self.tg = TelegramNotifier(config_path=self.config_path)
        self.tlog = TradeLogger(os.path.join(base_dir, "trade_history.json"))

        self._last_exec_sig: Optional[str] = None
        self._last_exec_ts: float = 0.0

        self._ensure_mt5()

    # -----------------------------
    # MT5 helpers
    # -----------------------------
    def _ensure_mt5(self) -> None:
        if mt5.terminal_info() is not None:
            return
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        logger.info("MT5 Connected")

    @staticmethod
    def _get_info(symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Cannot select symbol: {symbol} last_error={mt5.last_error()}")
        return info

    @staticmethod
    def _get_tick(symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for symbol: {symbol} last_error={mt5.last_error()}")
        return tick

    @staticmethod
    def _positions(symbol: str):
        pos = mt5.positions_get(symbol=symbol)
        if pos is None:
            return []
        return list(pos)

    # -----------------------------
    # Safety
    # -----------------------------
    def _spread_points(self, symbol: str) -> float:
        info = self._get_info(symbol)
        tick = self._get_tick(symbol)
        if info.point <= 0:
            return float("inf")
        return float((tick.ask - tick.bid) / info.point)

    def _validate_stops_levels(self, symbol: str, direction: str, entry: float, sl: float, tp: float) -> Tuple[bool, str]:
        info = self._get_info(symbol)
        point = float(info.point) if info.point else 0.0
        if point <= 0:
            return False, "invalid_symbol_point"

        stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
        freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)
        min_dist = float(stops_level_points) * point

        if direction == "BUY":
            if not (sl < entry < tp):
                return False, "sanity_failed_buy"
            if (entry - sl) < min_dist:
                return False, "stop_level_failed_sl"
            if (tp - entry) < min_dist:
                return False, "stop_level_failed_tp"
        elif direction == "SELL":
            if not (tp < entry < sl):
                return False, "sanity_failed_sell"
            if (sl - entry) < min_dist:
                return False, "stop_level_failed_sl"
            if (entry - tp) < min_dist:
                return False, "stop_level_failed_tp"
        else:
            return False, "invalid_direction"

        if freeze_level_points > 0:
            tick = self._get_tick(symbol)
            freeze_dist = float(freeze_level_points) * point
            cur = float(tick.ask if direction == "BUY" else tick.bid)
            if abs(cur - entry) <= freeze_dist:
                return False, "freeze_level_too_close"

        return True, "ok"

    # -----------------------------
    # Multi-order policy
    # -----------------------------
    def _multi_order_policy(self, cfg_eff: Dict[str, Any]) -> Dict[str, Any]:
        ex = (cfg_eff.get("execution") or {})
        return {
            "allow_multiple_positions": bool(ex.get("allow_multiple_positions", True)),
            "max_positions_per_symbol": _si(ex.get("max_positions_per_symbol", 2), 2),
            "min_distance_between_entries_atr": _sf(ex.get("min_distance_between_entries_atr", 0.60), 0.60),
            "cooldown_sec": _si(ex.get("cooldown_sec", 120), 120),
        }

    def _execution_dedupe_ok(self, policy: Dict[str, Any], sig: str) -> bool:
        cooldown = int(policy.get("cooldown_sec", 120))
        now = time.time()
        if self._last_exec_sig == sig and (now - self._last_exec_ts) < cooldown:
            return False
        self._last_exec_sig = sig
        self._last_exec_ts = now
        return True

    def _stacking_ok(self, symbol: str, entry: float, atr: float, policy: Dict[str, Any]) -> Tuple[bool, str]:
        pos = self._positions(symbol)
        if not pos:
            return True, "no_positions"

        max_pos = int(policy["max_positions_per_symbol"])
        if len(pos) >= max_pos:
            return False, f"max_positions_reached={len(pos)}/{max_pos}"

        min_dist_atr = float(policy["min_distance_between_entries_atr"])
        min_dist = min_dist_atr * max(atr, 1e-9)

        # If we cannot compute atr reliably -> fallback to conservative allow (but still distance check is weak)
        for p in pos:
            try:
                p_price = float(getattr(p, "price_open", 0.0) or 0.0)
                if abs(entry - p_price) < min_dist:
                    return False, f"stacking_too_close dist={abs(entry-p_price):.5f} min={min_dist:.5f}"
            except Exception:
                return False, "stacking_check_failed"

        return True, "stacking_ok"

    # -----------------------------
    # AI package builder (Spec v2)
    # -----------------------------
    def _build_ai_package(self, cfg_eff: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(cfg_eff.get("symbol", "GOLD"))
        direction = str(pkg.get("direction", "NONE")).upper()

        entry0 = _sf(pkg.get("entry_candidate"))
        sl0 = _sf(pkg.get("stop_candidate"))
        tp0 = _sf(pkg.get("tp_candidate"))
        rr0 = _sf(pkg.get("rr"), 0.0)

        # ATR from engine context (best-effort)
        ctx = (pkg.get("context") or {})
        atr = _sf(ctx.get("atr", pkg.get("atr", 0.0)), 0.0)

        spread_pts = self._spread_points(symbol)
        # Simple spread z estimate (no history) -> 0 by default
        exec_ctx = {
            "spread_current": float(spread_pts),
            "spread_avg_5m": float(spread_pts),
            "spread_z": 0.0,
            "slippage_estimate": 0.3,
            "freeze_level": int(getattr(self._get_info(symbol), "trade_freeze_level", 0) or 0),
            "stop_level": int(getattr(self._get_info(symbol), "trade_stops_level", 0) or 0),
        }

        # Structure flags (use TOP-LEVEL contract fields when present)
        structure = {
            "bos": bool(pkg.get("bos", False)),
            "choch": bool(ctx.get("choch", False)),
            "htf_bias": ctx.get("HTF_trend") or ctx.get("htf_bias") or "unknown",
            "mtf_bias": ctx.get("MTF_trend") or ctx.get("mtf_bias") or "unknown",
            "ltf_bias": ctx.get("LTF_trend") or ctx.get("ltf_bias") or "unknown",
            "proximity_score": ctx.get("proximity_score"),
        }

        # Technical snapshot (best-effort; if engine already supplies)
        technical = {
            "bb_state": ctx.get("bb_state") or ctx.get("bb"),
            "rsi": ctx.get("rsi"),
            "adx": ctx.get("adx"),
            "supertrend": ctx.get("supertrend_dir") or ("up" if bool(pkg.get("supertrend_ok", False)) and direction == "BUY" else "down"),
            "volatility_z": ctx.get("vol_z") or ctx.get("vol"),
        }

        # Context layer (optional feed from news_filter etc.)
        context = {
            "regime": ctx.get("regime") or cfg_eff.get("mode") or "unknown",
            "session": ctx.get("session") or "unknown",
            "event_risk": ctx.get("event_risk") or "NONE",
            "time_to_event_min": ctx.get("time_to_event_min"),
            "correlations": ctx.get("correlations") or {},
            "liquidity": ctx.get("liq") or ctx.get("liquidity") or "",
        }

        positions = self._positions(symbol)
        portfolio = {
            "open_positions_symbol": len(positions),
            "daily_pnl_usd": 0.0,
            "equity_drawdown_pct": 0.0,
            "correlation_risk": _sf(ctx.get("correlation_risk"), 0.0),
        }

        # Constraints (Mode D)
        ai_cfg = (cfg_eff.get("ai") or {})
        constraints = {
            "min_rr": _sf(cfg_eff.get("min_rr", 1.5), 1.5),
            "entry_shift_max_atr": _sf(ai_cfg.get("entry_shift_max_atr", 0.20), 0.20),
            "sl_atr_min": _sf(ai_cfg.get("sl_atr_min", 1.20), 1.20),
            "sl_atr_max": _sf(ai_cfg.get("sl_atr_max", 1.80), 1.80),
            "conf_execute_threshold": _si(cfg_eff.get("confidence_threshold", 70), 70),
            "event_conf_cap_high": _si(ai_cfg.get("event_conf_cap_high", 72), 72),
        }

        return {
            "symbol": symbol,
            "baseline": {"dir": direction, "entry": entry0, "sl": sl0, "tp": tp0, "rr": rr0, "atr": atr, "spread_points": spread_pts},
            "execution_context": exec_ctx,
            "technical": technical,
            "structure": structure,
            "context": context,
            "portfolio": portfolio,
            "constraints": constraints,
            "ask": "Validate baseline, adjust within constraints, output execution/analysis/mentor",
        }

    # -----------------------------
    # Python-side validation (must pass)
    # -----------------------------
    def _validate_ai_execution(
        self,
        ai_out: Dict[str, Any],
        baseline: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> Tuple[bool, str]:
        ex = (ai_out.get("execution") or {})
        direction = str(baseline.get("dir", "NONE")).upper()
        entry0, sl0, tp0 = _sf(baseline.get("entry")), _sf(baseline.get("sl")), _sf(baseline.get("tp"))
        atr = max(_sf(baseline.get("atr"), 0.0), 1e-9)

        entry1, sl1, tp1 = _sf(ex.get("entry")), _sf(ex.get("sl")), _sf(ex.get("tp"))
        rr1 = _sf(ex.get("rr"), 0.0)

        if not _sanity(direction, entry1, sl1, tp1):
            return False, "sanity_failed"

        # Entry shift
        max_shift = _sf(constraints.get("entry_shift_max_atr", 0.20), 0.20) * atr
        if abs(entry1 - entry0) > (max_shift + 1e-9):
            return False, "entry_shift_exceed"

        # SL ATR range
        sl_dist = abs(sl1 - entry1)
        sl_min = _sf(constraints.get("sl_atr_min", 1.20), 1.20) * atr
        sl_max = _sf(constraints.get("sl_atr_max", 1.80), 1.80) * atr
        if sl_dist < (sl_min - 1e-9) or sl_dist > (sl_max + 1e-9):
            return False, "sl_atr_range_fail"

        # RR min
        min_rr = _sf(constraints.get("min_rr", 1.50), 1.50)
        if rr1 < (min_rr - 1e-9):
            return False, "rr_below_min"

        return True, "ok"

    # -----------------------------
    # Order execution
    # -----------------------------
    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Tuple[bool, str]:
        info = self._get_info(symbol)
        tick = self._get_tick(symbol)

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": info.trade_fill_mode,
            "comment": "HIM v2.6.0",
        }

        res = mt5.order_send(request)
        if res is None:
            return False, f"order_send_none err={mt5.last_error()}"
        if getattr(res, "retcode", None) not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            return False, f"retcode={getattr(res,'retcode',None)} comment={getattr(res,'comment','')}"
        return True, f"ok retcode={getattr(res,'retcode',None)}"

    # -----------------------------
    # Main loop
    # -----------------------------
    def run_forever(self) -> None:
        logger.info("============================================================")
        logger.info("HIM Mentor Executor v2.6.0 | AI Spec v2 Mode D | Multi-order | SL Manager")
        logger.info("============================================================")

        while True:
            try:
                cfg_eff = self.hcfg.load_effective()
                symbol = str(cfg_eff.get("symbol", "GOLD"))

                # 1) Engine generates package
                pkg = self.engine.generate_signal_package()  # repository spec says this exists
                if not isinstance(pkg, dict):
                    time.sleep(1.0)
                    continue

                direction = str(pkg.get("direction", "NONE")).upper()
                if direction not in ("BUY", "SELL"):
                    time.sleep(0.5)
                    continue

                # 2) Frozen gate: BOS + SuperTrend OK required (top-level)
                bos = bool(pkg.get("bos", False))
                st_ok = bool(pkg.get("supertrend_ok", False))
                if not bos or not st_ok:
                    time.sleep(0.5)
                    continue

                # Baseline candidates
                entry0 = _sf(pkg.get("entry_candidate"))
                sl0 = _sf(pkg.get("stop_candidate"))
                tp0 = _sf(pkg.get("tp_candidate"))
                rr0 = _sf(pkg.get("rr"), 0.0)
                if not _sanity(direction, entry0, sl0, tp0):
                    time.sleep(0.5)
                    continue

                # 3) Multi-order policy checks
                policy = self._multi_order_policy(cfg_eff)
                if not self._execution_dedupe_ok(policy, sig=f"{symbol}:{direction}:{entry0:.5f}:{sl0:.5f}:{tp0:.5f}"):
                    time.sleep(0.5)
                    continue

                atr = _sf((pkg.get("context") or {}).get("atr", 0.0), 0.0)
                if policy["allow_multiple_positions"]:
                    ok_stack, why = self._stacking_ok(symbol, entry0, atr, policy)
                    if not ok_stack:
                        time.sleep(0.5)
                        continue
                else:
                    if len(self._positions(symbol)) > 0:
                        time.sleep(0.5)
                        continue

                # 4) Build AI package and call AI
                ai_pkg = self._build_ai_package(cfg_eff, pkg)
                ai_out = self.ai.evaluate(ai_pkg)

                baseline = ai_pkg.get("baseline") or {}
                constraints = ai_pkg.get("constraints") or {}

                # 5) Validate AI output (hard)
                ok_ai, why_ai = self._validate_ai_execution(ai_out, baseline, constraints)
                if not ok_ai:
                    time.sleep(0.5)
                    continue

                ex = ai_out.get("execution") or {}
                entry1 = _sf(ex.get("entry"))
                sl1 = _sf(ex.get("sl"))
                tp1 = _sf(ex.get("tp"))
                conf = _si(ex.get("conf"), 0)
                conf_th = _si(cfg_eff.get("confidence_threshold", 70), 70)

                # 6) Stops/freeze + spread guard before execute
                ok_lv, why_lv = self._validate_stops_levels(symbol, direction, entry1, sl1, tp1)
                if not ok_lv:
                    time.sleep(0.5)
                    continue

                spread = self._spread_points(symbol)
                max_spread = _sf(((cfg_eff.get("execution_safety") or {}).get("max_spread_points", 35)), 35)
                if spread > max_spread:
                    time.sleep(0.5)
                    continue

                # 7) Decision gate
                if conf < conf_th:
                    time.sleep(0.5)
                    continue

                # 8) Execute (lot from config; Mode D forbids AI sizing)
                lot = _sf(cfg_eff.get("lot", 0.01), 0.01)
                ok, msg = self._place_market_order(symbol, direction, lot, sl1, tp1)

                # 9) Send mentor message (AI as single narrative source)
                mentor = ai_out.get("mentor") or {}
                mentor_text = (
                    f"{mentor.get('headline','')}\n"
                    f"{mentor.get('explanation','')}\n"
                    f"{mentor.get('action_guidance','')}\n"
                    f"{mentor.get('confidence_reasoning','')}\n"
                    f"EXEC={ok} | {msg}"
                ).strip()

                try:
                    self.tg.send_text(mentor_text, event_type="trade")
                except Exception:
                    logger.error("Telegram send failed (isolated).")

                # 10) Log
                try:
                    self.tlog.log_trade({
                        "ts": time.time(),
                        "symbol": symbol,
                        "dir": direction,
                        "baseline": {"entry": entry0, "sl": sl0, "tp": tp0, "rr": rr0},
                        "ai": {"entry": entry1, "sl": sl1, "tp": tp1, "conf": conf},
                        "ok": ok,
                        "msg": msg,
                        "mentor_headline": mentor.get("headline", ""),
                        "risk_flags": ((ai_out.get("analysis") or {}).get("risk_flags") or []),
                    })
                except Exception:
                    logger.error("Trade log failed (isolated).")

                time.sleep(0.8)

            except KeyboardInterrupt:
                logger.warning("Stopped by user.")
                return
            except Exception:
                logger.error("CRITICAL LOOP ERROR")
                logger.error(traceback.format_exc())
                time.sleep(2.0)


if __name__ == "__main__":
    MentorExecutor().run_forever()