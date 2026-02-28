"""
Mentor Executor - Execution Controller (Phase 9 Live Stabilization)
Version: 2.7.0

Changelog:
- 2.7.0 (2026-02-28)
  - NEW: Enforce validator_v1_0 (confirm-only schema enforcement) at NEW TRADE boundary (fail-closed)
      * Direction/lot lock (lot sourced from cfg)
      * Entry shift bound (ATR-based)
      * SL tighten-only vs engine baseline
      * RR floor
  - KEEP: Existing SL manager logic, stops/freeze/spread clamp, MT5 safety guards

Frozen decisions:
- mode=sideway_scalp, min_rr=1.5, atr_sl_mult=1.6, require_confirmation=true
- AI cannot change direction, cannot change lot sizing
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
from validator_v1_0 import ValidationPolicy, validate_ai_response_v1_0

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HIM")


# -----------------------------
# small helpers
# -----------------------------
def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _si(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
    if direction == "BUY":
        return sl < entry < tp
    if direction == "SELL":
        return tp < entry < sl
    return False


class MentorExecutor:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.engine = TradingEngine(config_path)
        self.ai = AIMentor()
        self.tg = TelegramNotifier(config_path)
        self.tlog = TradeLogger(config_path)

    # -----------------------------
    # MT5 utils
    # -----------------------------
    def _get_info(self, symbol: str):
        return mt5.symbol_info(symbol)

    def _spread_points(self, symbol: str) -> float:
        tick = mt5.symbol_info_tick(symbol)
        info = self._get_info(symbol)
        if not tick or not info or not info.point:
            return 0.0
        return abs(float(tick.ask) - float(tick.bid)) / float(info.point)

    def _positions(self, symbol: str):
        try:
            return mt5.positions_get(symbol=symbol) or []
        except Exception:
            return []

    # -----------------------------
    # AI package builder
    # -----------------------------
    def _build_ai_package(self, cfg_eff: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(cfg_eff.get("symbol", "GOLD"))
        direction = str(pkg.get("direction", "NONE")).upper()

        entry0 = _sf(pkg.get("entry_candidate"))
        sl0 = _sf(pkg.get("stop_candidate"))
        tp0 = _sf(pkg.get("tp_candidate"))
        rr0 = _sf(pkg.get("rr"), 0.0)

        ctx = (pkg.get("context") or {})
        atr = _sf(ctx.get("atr", pkg.get("atr", 0.0)), 0.0)

        spread_pts = self._spread_points(symbol)
        info = self._get_info(symbol)
        exec_ctx = {
            "spread_current": float(spread_pts),
            "spread_avg_5m": float(spread_pts),
            "spread_z": 0.0,
            "slippage_estimate": 0.3,
            "freeze_level": int(getattr(info, "trade_freeze_level", 0) or 0),
            "stop_level": int(getattr(info, "trade_stops_level", 0) or 0),
        }

        structure = {
            "bos": bool(pkg.get("bos", False)),
            "choch": bool(ctx.get("choch", False)),
            "htf_bias": ctx.get("HTF_trend") or ctx.get("htf_bias") or "unknown",
            "mtf_bias": ctx.get("MTF_trend") or ctx.get("mtf_bias") or "unknown",
            "ltf_bias": ctx.get("LTF_trend") or ctx.get("ltf_bias") or "unknown",
            "proximity_score": ctx.get("proximity_score"),
        }

        technical = {
            "bb_state": ctx.get("bb_state") or ctx.get("bb"),
            "rsi": ctx.get("rsi"),
            "adx": ctx.get("adx"),
            "supertrend": ctx.get("supertrend_dir") or "unknown",
            "volatility_z": ctx.get("vol_z") or ctx.get("vol"),
        }

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

        ai_cfg = (cfg_eff.get("ai") or {})
        constraints = {
            "min_rr": _sf(cfg_eff.get("min_rr", 1.5), 1.5),
            "entry_shift_max_atr": _sf(ai_cfg.get("entry_shift_max_atr", 0.20), 0.20),
            "sl_atr_min": _sf(ai_cfg.get("sl_atr_min", 1.20), 1.20),
            "sl_atr_max": _sf(ai_cfg.get("sl_atr_max", 1.80), 1.80),
            "conf_execute_threshold": _si(cfg_eff.get("confidence_threshold", 70), 70),
            "event_conf_cap_high": _si(ai_cfg.get("event_conf_cap_high", 72), 72),
        }

        # baseline now includes lot/mode for validator locks
        lot = _sf(cfg_eff.get("lot", 0.01), 0.01)
        mode = str(cfg_eff.get("mode", "") or "")

        return {
            "symbol": symbol,
            "baseline": {
                "dir": direction,
                "entry": entry0,
                "sl": sl0,
                "tp": tp0,
                "rr": rr0,
                "atr": atr,
                "spread_points": spread_pts,
                "lot": lot,
                "mode": mode,
            },
            "execution_context": exec_ctx,
            "technical": technical,
            "structure": structure,
            "context": context,
            "portfolio": portfolio,
            "constraints": constraints,
        }

    # -----------------------------
    # Validator enforcement (NEW TRADE)
    # -----------------------------
    def _validate_ai_execution(self, ai_out: Dict[str, Any], baseline: Dict[str, Any], constraints: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Fail-closed validation for AI execution block.
        This enforces BOTH:
        - existing numeric/range checks (sl_atr_min/max etc.)
        - validator_v1_0 confirm-only rules (direction/lot lock, entry shift, SL tighten-only, RR floor)
        """
        ex = (ai_out.get("execution") or {})
        direction = str(baseline.get("dir", "NONE")).upper()

        entry0, sl0, tp0 = _sf(baseline.get("entry")), _sf(baseline.get("sl")), _sf(baseline.get("tp"))
        atr = max(_sf(baseline.get("atr"), 0.0), 1e-9)
        lot0 = _sf(baseline.get("lot", 0.0), 0.0)
        mode0 = str(baseline.get("mode", "") or "")

        entry1, sl1, tp1 = _sf(ex.get("entry")), _sf(ex.get("sl")), _sf(ex.get("tp"))
        rr1 = _sf(ex.get("rr"), 0.0)

        if not _sanity(direction, entry1, sl1, tp1):
            return False, "sanity_failed"

        # Existing bounded shift (ATR)
        max_shift = _sf(constraints.get("entry_shift_max_atr", 0.20), 0.20) * atr
        if abs(entry1 - entry0) > (max_shift + 1e-9):
            return False, "entry_shift_exceed"

        # Existing SL distance range (ATR) — keep execution safety
        sl_dist = abs(sl1 - entry1)
        sl_min = _sf(constraints.get("sl_atr_min", 1.20), 1.20) * atr
        sl_max = _sf(constraints.get("sl_atr_max", 1.80), 1.80) * atr
        if sl_dist < (sl_min - 1e-9) or sl_dist > (sl_max + 1e-9):
            return False, "sl_atr_range_fail"

        # Existing RR floor check
        min_rr = _sf(constraints.get("min_rr", 1.50), 1.50)
        if rr1 < (min_rr - 1e-9):
            return False, "rr_below_min"

        # NEW: validator_v1_0 confirm-only enforcement (fail-closed)
        engine_order = {
            "direction": direction,
            "entry": entry0,
            "sl": sl0,
            "tp": tp0,
            "lot": lot0,
            "atr": atr,
            "mode": mode0,
        }
        ai_payload = {
            "schema_version": "1.0",
            "decision": "CONFIRM",
            "confidence": float(_sf(ex.get("conf"), 0.0)) / 100.0,
            "direction": direction,
            "lot": lot0,
            "entry": entry1,
            "sl": sl1,
            "tp": tp1,
            "note": "executor_mapped",
        }
        policy = ValidationPolicy(
            rr_floor=float(min_rr),
            entry_shift_max_atr_mult=float(_sf(constraints.get("entry_shift_max_atr", 0.20), 0.20)),
            entry_shift_max_pct=0.0,
            enforce_mode_lock=False,
        )
        vr = validate_ai_response_v1_0(ai_payload, engine_order, policy=policy)

        if (not vr.ok) or (vr.decision != "CONFIRM"):
            # keep reason deterministic
            code = vr.errors[0] if vr.errors else "validator_reject"
            return False, f"validator_reject:{code}"

        return True, "ok"

    # -----------------------------
    # Stops / Freeze (new trade pre-check)
    # -----------------------------
    def _validate_stops_levels(self, symbol: str, direction: str, entry: float, sl: float, tp: float) -> Tuple[bool, str]:
        info = self._get_info(symbol)
        point = float(info.point) if info and info.point else 0.0
        if point <= 0:
            return False, "invalid_symbol_point"

        stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
        min_dist = float(stops_level_points) * point

        if direction == "BUY":
            if not (sl < entry < tp):
                return False, "sanity_failed_buy"
            if (entry - sl) < min_dist or (tp - entry) < min_dist:
                return False, "stop_level_failed"
        elif direction == "SELL":
            if not (tp < entry < sl):
                return False, "sanity_failed_sell"
            if (sl - entry) < min_dist or (entry - tp) < min_dist:
                return False, "stop_level_failed"
        else:
            return False, "invalid_direction"

        return True, "ok"

    # -----------------------------
    # Order placement
    # -----------------------------
    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Tuple[bool, str]:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False, "no_tick"

        price = float(tick.ask) if direction == "BUY" else float(tick.bid)
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 2602,
            "comment": "HIM",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        res = mt5.order_send(req)
        if not res:
            return False, "order_send_none"
        if int(res.retcode) != mt5.TRADE_RETCODE_DONE:
            return False, f"retcode={res.retcode} comment={getattr(res,'comment','')}"
        return True, "done"

    # -----------------------------
    # Main loop (simplified)
    # -----------------------------
    def run(self) -> None:
        if not mt5.initialize():
            logger.error("MT5 init failed")
            return

        logger.info("MT5 Connected")
        while True:
            try:
                cfg_eff = resolve_effective_config(self.config_path) or {}
                symbol = str(cfg_eff.get("symbol", "GOLD"))
                mode = str(cfg_eff.get("mode", "sideway_scalp"))

                # Engine tick
                pkg = self.engine.generate_signal_package(symbol=symbol, mode=mode)

                # No candidate
                if not pkg or not pkg.get("direction") or pkg.get("direction") == "NONE":
                    time.sleep(0.5)
                    continue

                # avoid stacking
                if len(self._positions(symbol)) > 0:
                    time.sleep(0.5)
                    continue

                # AI (mock/rule-based)
                ai_pkg = self._build_ai_package(cfg_eff, pkg)
                ai_out = self.ai.evaluate(ai_pkg)

                baseline = ai_pkg.get("baseline") or {}
                constraints = ai_pkg.get("constraints") or {}

                ok_ai, why_ai = self._validate_ai_execution(ai_out, baseline, constraints)
                if not ok_ai:
                    time.sleep(0.5)
                    continue

                ex = ai_out.get("execution") or {}
                direction = str(baseline.get("dir", "NONE")).upper()
                entry1 = _sf(ex.get("entry"))
                sl1 = _sf(ex.get("sl"))
                tp1 = _sf(ex.get("tp"))
                conf = _si(ex.get("conf"), 0)
                conf_th = _si(cfg_eff.get("confidence_threshold", 70), 70)

                ok_lv, why_lv = self._validate_stops_levels(symbol, direction, entry1, sl1, tp1)
                if not ok_lv:
                    time.sleep(0.5)
                    continue

                spread = self._spread_points(symbol)
                max_spread = _sf(((cfg_eff.get("execution_safety") or {}).get("max_spread_points", 35)), 35)
                if spread > max_spread:
                    time.sleep(0.5)
                    continue

                if conf < conf_th:
                    time.sleep(0.5)
                    continue

                lot = _sf(cfg_eff.get("lot", 0.01), 0.01)
                ok, msg = self._place_market_order(symbol, direction, lot, sl1, tp1)

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
                    pass

                try:
                    self.tlog.log_trade(
                        {
                            "ts": time.time(),
                            "symbol": symbol,
                            "dir": direction,
                            "baseline": {"entry": _sf(baseline.get("entry")), "sl": _sf(baseline.get("sl")), "tp": _sf(baseline.get("tp")), "rr": _sf(baseline.get("rr"))},
                            "ai": {"entry": entry1, "sl": sl1, "tp": tp1, "conf": conf},
                            "ok": ok,
                            "msg": msg,
                            "mentor_headline": mentor.get("headline", ""),
                            "risk_flags": ((ai_out.get("analysis") or {}).get("risk_flags") or []),
                        }
                    )
                except Exception:
                    pass

                time.sleep(0.5)

            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
            except Exception:
                logger.error("Loop exception:\n" + traceback.format_exc())
                time.sleep(1.0)


def main() -> int:
    ex = MentorExecutor("config.json")
    ex.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())