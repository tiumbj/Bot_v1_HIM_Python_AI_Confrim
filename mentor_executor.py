"""
File: mentor_executor.py
Path: C:\\Hybrid_Intelligence_Mentor\\mentor_executor.py
Version: 2.10.1

CHANGELOG
- 2.10.1
  - FIX: TRANSITION = HARD NO-TRADE ZONE (return before Engine/AI/RiskGuard)
  - KEEP: Adaptive Strategy Router (SIDEWAY->sideway_scalp, TREND->breakout)
  - KEEP: RiskGuard v1.0.0 + mt5.history_deals_get() (Option A)
  - LIVE flag remains derived from effective config (router may force DRY_RUN)

SAFETY
- If adaptive_regime == TRANSITION:
    - Do NOT call engine
    - Do NOT call AI
    - Do NOT call RiskGuard
    - Log: DEBUG_SKIP | transition_phase_guard
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

from ai_bridge_v1 import confirm_via_api
from regime_switch import decide_regime
from risk_guard_v1_0 import (
    RiskGuard,
    GuardConfig,
    TradePlan,
    MarketSnapshot,
    AccountSnapshot,
    PerformanceSnapshot,
    GuardAction,
)
from strategy_router import build_effective_config_adaptive

LOG = logging.getLogger("mentor_executor")

_SENT_FINGERPRINTS: Dict[str, datetime] = {}
_LAST_EXEC_UTC: Optional[datetime] = None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol not found: {symbol}")
    if not info.visible:
        mt5.symbol_select(symbol, True)


def _market_snapshot(symbol: str, *, atr: Optional[float]) -> MarketSnapshot:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        raise RuntimeError("market snapshot unavailable")

    return MarketSnapshot(
        tick_time_utc=datetime.fromtimestamp(int(tick.time), tz=timezone.utc),
        bid=float(tick.bid),
        ask=float(tick.ask),
        point=float(info.point),
        digits=int(info.digits),
        atr=float(atr) if atr is not None else None,
        bb_width=None,
    )


def _account_snapshot(symbol: str) -> AccountSnapshot:
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("account_info unavailable")

    positions = mt5.positions_get(symbol=symbol)
    net_dir = None
    if positions and len(positions) > 0:
        p0 = positions[0]
        net_dir = "BUY" if int(p0.type) == int(mt5.POSITION_TYPE_BUY) else "SELL"

    return AccountSnapshot(
        equity=float(acc.equity),
        balance=float(acc.balance),
        positions_count=len(positions) if positions else 0,
        net_position_direction=net_dir,
    )


def _performance_snapshot_today_via_deals_get() -> PerformanceSnapshot:
    today = date.today()
    start_utc = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    end_utc = _now_utc()

    deals = mt5.history_deals_get(start_utc, end_utc)
    if deals is None:
        raise RuntimeError("history_deals_get returned None")

    realized = 0.0
    closed: list[Tuple[int, float]] = []

    for d in deals:
        profit = float(getattr(d, "profit", 0.0))
        realized += profit

        entry = getattr(d, "entry", None)
        t = int(getattr(d, "time", 0))
        try:
            if entry is not None and int(entry) == int(mt5.DEAL_ENTRY_OUT):
                closed.append((t, profit))
        except Exception:
            pass

    closed.sort(key=lambda x: x[0])
    streak = 0
    for _, p in reversed(closed):
        if p < 0:
            streak += 1
        elif p > 0:
            break
        else:
            continue

    return PerformanceSnapshot(
        today=today,
        realized_pl_today=float(realized),
        consecutive_losses=int(streak),
    )


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
        return True, ""

    side_u = side.upper()
    if side_u == "BUY":
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


def _build_risk_guard_config(cfg_eff: Dict[str, Any]) -> GuardConfig:
    execution = cfg_eff.get("execution", {}) if isinstance(cfg_eff.get("execution", {}), dict) else {}

    tick_max_age_sec = int(execution.get("tick_max_age_sec", 8))
    max_spread_points = int(execution.get("max_spread_points", 300))
    cooldown_sec = int(execution.get("cooldown_sec", execution.get("duplicate_cooldown_sec", 45)))

    daily_loss_limit_pct = float(execution.get("daily_loss_limit_pct", 0.01))
    max_consecutive_losses = int(execution.get("max_consecutive_losses", 3))

    max_slippage_points = int(execution.get("max_slippage_points", execution.get("max_deviation_points", 80)))

    max_positions = int(execution.get("max_positions", 1))
    no_hedge = bool(execution.get("no_hedge", True))
    dedup_window_sec = int(execution.get("dedup_window_sec", 90))

    return GuardConfig(
        tick_max_age_sec=max(1, tick_max_age_sec),
        max_spread_points=max(1, max_spread_points),
        max_positions=max(1, max_positions),
        no_hedge=no_hedge,
        dedup_window_sec=max(1, dedup_window_sec),
        cooldown_sec=max(0, cooldown_sec),
        daily_loss_limit_pct=max(0.0, daily_loss_limit_pct),
        max_consecutive_losses=max(0, max_consecutive_losses),
        max_slippage_points=max(0, max_slippage_points),
        min_rr_exec=None,
        fail_closed=True,
    )


def _prune_sent_fingerprints(now: datetime, *, keep_sec: int = 600) -> None:
    cutoff = now - timedelta(seconds=max(1, keep_sec))
    dead = [k for k, t in _SENT_FINGERPRINTS.items() if t < cutoff]
    for k in dead:
        _SENT_FINGERPRINTS.pop(k, None)


def _send_market_order(symbol: str, side: str, lot: float, sl: float, tp: float, deviation: int) -> Tuple[bool, str]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, "no_tick"

    side_u = side.upper()
    order_type = mt5.ORDER_TYPE_BUY if side_u == "BUY" else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if side_u == "BUY" else tick.bid)

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
        "comment": "HIM v2.10.1",
    }

    result = mt5.order_send(request)
    if result is None:
        return False, "order_send_none"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"order_send_failed(retcode={result.retcode})"
    return True, f"order_ok(ticket={result.order})"


def main() -> None:
    global _LAST_EXEC_UTC

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = _load_json("config.json")
    symbol = str(cfg.get("symbol", "GOLD"))

    if not mt5.initialize():
        LOG.error("MT5 init failed: %s", mt5.last_error())
        return

    try:
        _ensure_symbol(symbol)

        now = _now_utc()
        _prune_sent_fingerprints(now)

        # 1) Compute base regime metrics
        decision, err = decide_regime(cfg, symbol)
        if err:
            LOG.info("DEBUG_SKIP | regime_error=%s", err)
            return

        # 2) Adaptive router: pick mode & effective config
        cfg_eff, sel = build_effective_config_adaptive(cfg, decision)

        mode_eff = str(cfg_eff.get("mode", "manual"))
        live_eff = bool(cfg_eff.get("enable_execution", False))
        lot = float(cfg_eff.get("lot", (cfg_eff.get("execution", {}) or {}).get("volume", 0.01)))

        execution = cfg_eff.get("execution", {}) if isinstance(cfg_eff.get("execution", {}), dict) else {}
        deviation = int(execution.get("deviation", execution.get("max_deviation_points", 20)))

        ai_cfg = cfg_eff.get("ai", {}) if isinstance(cfg_eff.get("ai", {}), dict) else {}
        api_url = ai_cfg.get("api_url", "http://127.0.0.1:5000/api/ai_confirm")
        timeout_sec = float(ai_cfg.get("timeout_sec", 10))

        os.makedirs(".state", exist_ok=True)
        eff_path = ".state/effective_config.executor.json"
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(cfg_eff, f, ensure_ascii=False, indent=2)

        LOG.info(
            "HIM Mentor Executor v2.10.1 | symbol=%s | mode=%s | LIVE=%s | lot=%.2f",
            symbol, mode_eff, live_eff, lot
        )

        LOG.info(
            "ROUTER | adaptive_regime=%s | mode_selected=%s | transition_dry_run=%s | adx=%.2f bbw=%.3f counts(on/off)=%d/%d",
            sel.get("adaptive_regime"),
            sel.get("mode_selected"),
            bool(sel.get("transition_dry_run")),
            float(sel.get("metrics", {}).get("adx", 0.0)),
            float(sel.get("metrics", {}).get("bb_width_atr", 0.0)),
            int(sel.get("metrics", {}).get("trend_on_count", 0)),
            int(sel.get("metrics", {}).get("trend_off_count", 0)),
        )

        # 2.5) HARD NO-TRADE ZONE: TRANSITION => stop immediately
        if str(sel.get("adaptive_regime", "")).upper() == "TRANSITION":
            LOG.info("DEBUG_SKIP | transition_phase_guard")
            return

        # 3) Engine uses effective config only (SIDEWAY or TREND only at this point)
        from engine import TradingEngine

        eng = TradingEngine(eff_path)
        pkg = eng.generate_signal_package()
        ctx = pkg.get("context", {}) if isinstance(pkg.get("context", {}), dict) else {}

        direction = str(pkg.get("direction", "NONE")).upper()
        blocked_by = ctx.get("blocked_by")

        if direction not in ("BUY", "SELL"):
            LOG.info("DEBUG_SKIP | no_candidate_direction=%s | blocked_by=%s", direction, blocked_by)
            return
        if blocked_by not in (None, "", "None"):
            LOG.info("DEBUG_SKIP | blocked_by=%s", blocked_by)
            return

        entry = pkg.get("entry_candidate")
        sl = pkg.get("stop_candidate")
        tp = pkg.get("tp_candidate")
        if entry is None or sl is None or tp is None:
            LOG.info("DEBUG_SKIP | engine_missing_fields entry/sl/tp=%s/%s/%s", entry, sl, tp)
            return

        entry_f = float(entry)
        sl_f = float(sl)
        tp_f = float(tp)
        min_rr = float(ctx.get("min_rr", cfg_eff.get("min_rr", 1.5)))

        ok_levels, reason_levels = _validate_stop_levels(symbol, sl_f, tp_f, entry_f, direction)
        if not ok_levels:
            LOG.info("DEBUG_SKIP | stop_level_block %s", reason_levels)
            return

        # 4) RiskGuard PRE-EXEC
        rg_cfg = _build_risk_guard_config(cfg_eff)
        rg = RiskGuard(rg_cfg)

        market = _market_snapshot(symbol, atr=ctx.get("atr"))
        account = _account_snapshot(symbol)

        try:
            perf = _performance_snapshot_today_via_deals_get()
        except Exception as e:
            LOG.info("RISK_BLOCK | action=BLOCK_HARD | reasons=history_unavailable err=%s", str(e))
            return

        plan = TradePlan(
            symbol=symbol,
            direction=direction,
            entry=entry_f,
            sl=sl_f,
            tp=tp_f,
            min_rr=min_rr,
            created_utc=now,
        )

        rg_dec = rg.evaluate_pre_exec(
            plan,
            market,
            account,
            perf,
            now_utc=now,
            last_exec_utc=_LAST_EXEC_UTC,
            sent_fingerprints=_SENT_FINGERPRINTS,
        )

        if rg_dec.action != GuardAction.ALLOW:
            LOG.info(
                "RISK_BLOCK | action=%s | fingerprint=%s | reasons=%s | perf_today=%.2f loss_streak=%d",
                rg_dec.action.value,
                rg_dec.fingerprint,
                [(r.code, r.message, r.data) for r in rg_dec.reasons],
                perf.realized_pl_today,
                perf.consecutive_losses,
            )
            return

        if rg_dec.fingerprint:
            _SENT_FINGERPRINTS[rg_dec.fingerprint] = now

        # 5) AI confirm + execute (SIDEWAY/TREND only)
        engine_order = {
            "direction": direction,
            "entry": entry_f,
            "sl": sl_f,
            "tp": tp_f,
            "lot": float(lot),
            "mode": mode_eff,
            "atr": ctx.get("atr"),
        }

        ai_decision, ai_payload = confirm_via_api(
            api_url=api_url,
            timeout_sec=timeout_sec,
            engine_order=engine_order,
        )

        if not ai_decision.final_confirm:
            LOG.info("DEBUG_SKIP | ai_reject_or_validator_reject | hint=%s | ai_payload=%s", ai_decision.mentor_hint, ai_payload)
            return

        if not live_eff:
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

        _LAST_EXEC_UTC = _now_utc()
        LOG.info("ORDER_OK | %s", msg)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()