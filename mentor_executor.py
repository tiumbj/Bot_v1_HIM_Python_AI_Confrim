"""
File: mentor_executor.py
Path: C:\\Hybrid_Intelligence_Mentor\\mentor_executor.py
Version: 2.8.0

Changelog:
- v2.8.0:
  - Integrate Regime Switch (ADX+BBWidth/ATR) -> build effective config in-memory.
  - Engine -> engine_order (entry/sl/tp/lot/mode/atr).
  - AI confirm via /api/ai_confirm using schema v1.0 baseline fields.
  - Validator enforced (fail-closed).
  - MT5 safety guards: stale tick, spread, stop_level/freeze_level, duplicate position guard.
  - Supports DRY_RUN mode (default True unless config execution.live=true).

Notes (TH):
- โฟกัส: ทำให้ End-to-End ทำงานจริง (Engine→AI→Validator→MT5)
- ไม่พยายาม optimize strategy ในไฟล์นี้
- ถ้าไม่มี candidate หรือถูกบล็อก: log DEBUG_SKIP พร้อมเหตุผล
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

from ai_bridge_v1 import confirm_via_api
from regime_switch import decide_regime, build_effective_config

LOG = logging.getLogger("mentor_executor")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now_ts() -> int:
    return int(time.time())


def _ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol not found: {symbol}")
    if not info.visible:
        mt5.symbol_select(symbol, True)


def _get_spread_points(symbol: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    point = float(info.point)
    if point <= 0:
        return None
    return float(tick.ask - tick.bid) / point


def _is_tick_stale(symbol: str, max_age_sec: int) -> Tuple[bool, str]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return True, "no_tick"
    age = _now_ts() - int(tick.time)
    if age > max_age_sec:
        return True, f"stale_tick_age_sec={age}"
    return False, ""


def _has_open_position(symbol: str) -> bool:
    pos = mt5.positions_get(symbol=symbol)
    return pos is not None and len(pos) > 0


def _validate_stop_levels(symbol: str, sl: float, tp: float, entry: float, side: str) -> Tuple[bool, str]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return False, "no_symbol_info"

    stop_level = int(getattr(info, "trade_stops_level", 0))
    freeze_level = int(getattr(info, "trade_freeze_level", 0))
    point = float(info.point) if float(info.point) > 0 else 0.0

    if point <= 0:
        return False, "bad_point"

    min_dist = float(max(stop_level, freeze_level)) * point
    if min_dist <= 0:
        return True, ""  # no restriction

    if side == "BUY":
        if (entry - sl) < min_dist:
            return False, f"sl_too_close(min_dist={min_dist})"
        if (tp - entry) < min_dist:
            return False, f"tp_too_close(min_dist={min_dist})"
    else:
        if (sl - entry) < min_dist:
            return False, f"sl_too_close(min_dist={min_dist})"
        if (entry - tp) < min_dist:
            return False, f"tp_too_close(min_dist={min_dist})"
    return True, ""


def _send_market_order(symbol: str, side: str, lot: float, sl: float, tp: float, deviation: int) -> Tuple[bool, str]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, "no_tick"

    if side == "BUY":
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
        "deviation": int(deviation),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "HIM v2.8.0",
    }

    result = mt5.order_send(request)
    if result is None:
        return False, "order_send_none"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"order_send_failed(retcode={result.retcode})"
    return True, f"order_ok(ticket={result.order})"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = "config.json"
    cfg = _load_json(config_path)

    symbol = cfg.get("symbol", "GOLD")
    mode = cfg.get("mode", "sideway_scalp")

    exec_cfg = cfg.get("execution", {})
    live = bool(exec_cfg.get("live", False))
    lot = float(exec_cfg.get("volume", 0.01))
    max_spread_points = float(exec_cfg.get("max_spread_points", 60))  # คุณปรับได้
    tick_max_age_sec = int(exec_cfg.get("tick_max_age_sec", 15))
    deviation = int(exec_cfg.get("deviation", 20))
    no_duplicate_position = bool(exec_cfg.get("no_duplicate_position", True))

    ai_cfg = cfg.get("ai", {})
    api_url = ai_cfg.get("api_url", "http://127.0.0.1:5000/api/ai_confirm")
    timeout_sec = float(ai_cfg.get("timeout_sec", 10))

    LOG.info("HIM Mentor Executor v2.8.0 | symbol=%s | mode=%s | live=%s | lot=%s", symbol, mode, live, lot)

    if not mt5.initialize():
        LOG.error("MT5 init failed: %s", mt5.last_error())
        return

    try:
        _ensure_symbol(symbol)

        stale, stale_reason = _is_tick_stale(symbol, tick_max_age_sec)
        if stale:
            LOG.info("DEBUG_SKIP | %s", stale_reason)
            return

        spread = _get_spread_points(symbol)
        if spread is None:
            LOG.info("DEBUG_SKIP | spread_unavailable")
            return
        if spread > max_spread_points:
            LOG.info("DEBUG_SKIP | spread_block spread_points=%.2f > max=%.2f", spread, max_spread_points)
            return

        if no_duplicate_position and _has_open_position(symbol):
            LOG.info("DEBUG_SKIP | position_guard already_has_position")
            return

        # 1) Regime switch
        decision, err = decide_regime(cfg, symbol)
        if err:
            LOG.info("DEBUG_SKIP | regime_error=%s", err)
            return

        eff_cfg = build_effective_config(cfg, decision)
        LOG.info(
            "REGIME | %s | adx=%.2f (max=%.2f) | bb_width_atr=%.3f (max=%.3f) | tf=M%d",
            decision.regime,
            decision.adx,
            decision.adx_max,
            decision.bb_width_atr,
            decision.bb_width_atr_max,
            decision.timeframe_minutes,
        )

        # 2) Engine
        from engine import TradingEngine  # import after MT5 init

        # ใช้ effective config แบบ in-memory โดยเขียนไฟล์ชั่วคราวลง .state
        import os

        os.makedirs(".state", exist_ok=True)
        eff_path = ".state/effective_config.executor.json"
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(eff_cfg, f, ensure_ascii=False, indent=2)

        eng = TradingEngine(eff_path)
        pkg = eng.generate_signal_package()
        ctx = pkg.get("context", {}) if isinstance(pkg.get("context", {}), dict) else {}

        direction = pkg.get("direction", "NONE")
        blocked_by = ctx.get("blocked_by")
        if direction not in ("BUY", "SELL"):
            LOG.info("DEBUG_SKIP | no_candidate_direction_%s | blocked_by=%s", direction, blocked_by)
            return
        if blocked_by not in (None, "", "None"):
            LOG.info("DEBUG_SKIP | blocked_by=%s", blocked_by)
            return

        engine_order = {
            "direction": direction,
            "entry": pkg.get("entry_candidate"),
            "sl": pkg.get("stop_candidate"),
            "tp": pkg.get("tp_candidate"),
            "lot": lot,
            "mode": mode,
            "atr": ctx.get("atr"),
        }

        if any(engine_order[k] is None for k in ("entry", "sl", "tp")):
            LOG.info("DEBUG_SKIP | engine_missing_fields entry/sl/tp=%s/%s/%s", engine_order["entry"], engine_order["sl"], engine_order["tp"])
            return

        ok_levels, reason_levels = _validate_stop_levels(
            symbol,
            float(engine_order["sl"]),
            float(engine_order["tp"]),
            float(engine_order["entry"]),
            direction,
        )
        if not ok_levels:
            LOG.info("DEBUG_SKIP | stop_level_block %s", reason_levels)
            return

        # 3) AI confirm + Validator (fail-closed)
        ai_decision, ai_payload = confirm_via_api(
            api_url=api_url,
            timeout_sec=timeout_sec,
            engine_order=engine_order,
        )

        if not ai_decision.final_confirm:
            LOG.info("DEBUG_SKIP | ai_reject_or_validator_reject | hint=%s | ai_payload=%s", ai_decision.mentor_hint, ai_payload)
            return

        # 4) Execution
        if not live:
            LOG.info(
                "DRY_RUN_ORDER | side=%s lot=%.2f entry=%.2f sl=%.2f tp=%.2f | conf=%.1f | note=%s",
                ai_decision.side,
                lot,
                ai_decision.entry,
                ai_decision.sl,
                ai_decision.tp,
                ai_decision.confidence,
                ai_decision.mentor_hint,
            )
            return

        sent, msg = _send_market_order(symbol, ai_decision.side, lot, ai_decision.sl, ai_decision.tp, deviation)
        if not sent:
            LOG.error("ORDER_FAIL | %s", msg)
            return

        LOG.info("ORDER_OK | %s", msg)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()