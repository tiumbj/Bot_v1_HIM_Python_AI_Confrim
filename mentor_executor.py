"""
Mentor Executor - Execution Controller (Production Hardening Phase)
Version: 2.4.2

Changelog:
- 2.4.2 (2026-02-26):
  - Observability: when Engine returns direction=NONE, log context reasons:
      * blocked_by, watch_state, breakout_state/side, bos, retest_ok,
        proximity_score/side/dist/thr, supertrend_dir/value/ok
  - CFG_EFFECTIVE: log a focused but complete snapshot including ai/risk/execution/telegram if present
  - Keep LOCKED rules:
      * Telegram sends ONLY after final approval (no reject/NONE spam)
      * Engine = pricing owner, Executor = risk+safety enforcement
      * AI confirm-only when ai.enabled=true

Notes:
- This file is a full replacement (No patch).
"""

from __future__ import annotations

import os
import json
import time
import math
import logging
import traceback
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

from engine import TradingEngine
from ai_mentor import AIMentor
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger
from config_resolver import resolve_effective_config


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


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(x / step) * step


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

        # Dedupe/cooldown (separate)
        self._last_tg_sig: Optional[str] = None
        self._last_tg_ts: float = 0.0

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
            err = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {err}")
        logger.info("MT5 Connected")

    @staticmethod
    def _get_tick(symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for symbol: {symbol} last_error={mt5.last_error()}")
        return tick

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
    def _positions_exist(symbol: str) -> bool:
        pos = mt5.positions_get(symbol=symbol)
        # Conservative: if MT5 cannot return positions, block to avoid duplicate risk
        if pos is None:
            return True
        return bool(len(pos) > 0)

    # -----------------------------
    # Config snapshot (better visibility)
    # -----------------------------
    @staticmethod
    def _cfg_snapshot(cfg_eff: Dict[str, Any]) -> Dict[str, Any]:
        # Focused but complete enough for auditing
        snap = {
            "mode": cfg_eff.get("mode"),
            "symbol": cfg_eff.get("symbol"),
            "enable_execution": cfg_eff.get("enable_execution"),
            "confidence_threshold": cfg_eff.get("confidence_threshold"),
            "min_score": cfg_eff.get("min_score"),
            "min_rr": cfg_eff.get("min_rr"),
            "lot": cfg_eff.get("lot"),
            "timeframes": cfg_eff.get("timeframes"),
            "atr_period": cfg_eff.get("atr_period"),
            "atr_sl_mult": cfg_eff.get("atr_sl_mult"),
        }
        # Optional sections if exist
        for k in ("ai", "risk", "execution", "execution_safety", "telegram"):
            if k in cfg_eff:
                snap[k] = cfg_eff.get(k)
        return snap

    # -----------------------------
    # Safety layer
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
                return False, f"stop_level_failed_sl dist={entry - sl:.5f} min={min_dist:.5f}"
            if (tp - entry) < min_dist:
                return False, f"stop_level_failed_tp dist={tp - entry:.5f} min={min_dist:.5f}"
        elif direction == "SELL":
            if not (tp < entry < sl):
                return False, "sanity_failed_sell"
            if (sl - entry) < min_dist:
                return False, f"stop_level_failed_sl dist={sl - entry:.5f} min={min_dist:.5f}"
            if (entry - tp) < min_dist:
                return False, f"stop_level_failed_tp dist={entry - tp:.5f} min={min_dist:.5f}"
        else:
            return False, "invalid_direction"

        # Freeze-level: conservative check (skip if too close to current market)
        if freeze_level_points > 0:
            tick = self._get_tick(symbol)
            freeze_dist = float(freeze_level_points) * point
            cur = float(tick.ask if direction == "BUY" else tick.bid)
            if abs(cur - entry) <= freeze_dist:
                return False, f"freeze_level_too_close cur-entry={abs(cur-entry):.5f} freeze={freeze_dist:.5f}"

        return True, "ok"

    # -----------------------------
    # Risk sizing (compat keys)
    # -----------------------------
    @staticmethod
    def _get_risk_pct(cfg_eff: Dict[str, Any]) -> float:
        """
        Supported:
        - risk.risk_pct: fraction (0.01 = 1%)
        - risk.risk_percent: percent (1.0 = 1%)
        Priority: risk_pct if exists else risk_percent
        Clamp: 0–5%
        """
        risk_cfg = (cfg_eff.get("risk") or {}) if isinstance(cfg_eff, dict) else {}
        if "risk_pct" in risk_cfg:
            v = _safe_float(risk_cfg.get("risk_pct"), 0.01)
            risk_pct = v
        elif "risk_percent" in risk_cfg:
            v = _safe_float(risk_cfg.get("risk_percent"), 1.0)
            risk_pct = v / 100.0
        else:
            risk_pct = 0.01
        return _clamp(float(risk_pct), 0.0, 0.05)

    def _calc_lot(
        self,
        cfg_eff: Dict[str, Any],
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        fallback_lot: float,
    ) -> Tuple[float, str]:
        try:
            acct = mt5.account_info()
            if acct is None:
                return float(fallback_lot), "account_info_none_fallback"
            equity = float(getattr(acct, "equity", 0.0) or 0.0)
            if equity <= 0:
                return float(fallback_lot), "equity_invalid_fallback"

            risk_pct = self._get_risk_pct(cfg_eff)
            risk_amount = equity * risk_pct
            if risk_amount <= 0:
                return float(fallback_lot), "risk_amount_invalid_fallback"

            info = self._get_info(symbol)
            vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
            vol_max = float(getattr(info, "volume_max", 100.0) or 100.0)
            vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)

            order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            loss_1lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
            if loss_1lot is None:
                return float(fallback_lot), "order_calc_profit_none_fallback"

            risk_per_lot = abs(float(loss_1lot))
            if risk_per_lot <= 0:
                return float(fallback_lot), "risk_per_lot_invalid_fallback"

            raw_lot = risk_amount / risk_per_lot
            lot = _floor_to_step(raw_lot, vol_step)  # floor only (never exceed risk)
            lot = _clamp(lot, vol_min, vol_max)

            return float(lot), f"risk equity={equity:.2f} risk_pct={risk_pct:.4f} risk={risk_amount:.2f} risk_per_1lot={risk_per_lot:.2f}"
        except Exception as e:
            return float(fallback_lot), f"risk_calc_exception_fallback: {e}"

    # -----------------------------
    # Telegram approved-only
    # -----------------------------
    def _telegram_dedupe_ok(self, cfg_eff: Dict[str, Any], sig: str) -> bool:
        tg_cfg = (cfg_eff.get("telegram") or {}) if isinstance(cfg_eff, dict) else {}
        cooldown_sec = _safe_int(tg_cfg.get("cooldown_sec", 900), 900)

        now = time.time()
        if self._last_tg_sig == sig and (now - self._last_tg_ts) < cooldown_sec:
            return False
        self._last_tg_sig = sig
        self._last_tg_ts = now
        return True

    def _safe_send_telegram_trade(self, cfg_eff: Dict[str, Any], text: str, sig: str) -> None:
        try:
            if not self._telegram_dedupe_ok(cfg_eff, sig):
                logger.info("Telegram: dedupe/cooldown suppress.")
                return
            ok = self.tg.send_text(text, event_type="trade")
            if not ok:
                logger.warning("Telegram send failed or disabled.")
        except Exception:
            logger.error("CRITICAL: telegram send failed (isolated).")
            logger.error(traceback.format_exc())

    # -----------------------------
    # Execution dedupe
    # -----------------------------
    def _execution_dedupe_ok(self, cfg_eff: Dict[str, Any], sig: str) -> bool:
        exec_cfg = (cfg_eff.get("execution") or {}) if isinstance(cfg_eff, dict) else {}
        cooldown_sec = _safe_int(exec_cfg.get("cooldown_sec", 120), 120)

        now = time.time()
        if self._last_exec_sig == sig and (now - self._last_exec_ts) < cooldown_sec:
            return False
        self._last_exec_sig = sig
        self._last_exec_ts = now
        return True

    # -----------------------------
    # NONE reason tracing (NEW)
    # -----------------------------
    @staticmethod
    def _trace_none_context(pkg: Dict[str, Any]) -> Dict[str, Any]:
        ctx = pkg.get("context", {}) or {}
        return {
            "blocked_by": ctx.get("blocked_by"),
            "watch_state": ctx.get("watch_state"),
            "breakout_state": ctx.get("breakout_state"),
            "breakout_side": ctx.get("breakout_side"),
            "bos": ctx.get("bos"),
            "retest_ok": ctx.get("retest_ok"),
            "proximity_score": ctx.get("proximity_score"),
            "proximity_side": ctx.get("proximity_side"),
            "proximity_best_dist": ctx.get("proximity_best_dist"),
            "threshold_points": ctx.get("breakout_proximity_threshold_points"),
            "supertrend_dir": ctx.get("supertrend_dir"),
            "supertrend_value": ctx.get("supertrend_value"),
            "supertrend_ok": ctx.get("supertrend_ok"),
            "HTF_trend": ctx.get("HTF_trend"),
            "MTF_trend": ctx.get("MTF_trend"),
            "LTF_trend": ctx.get("LTF_trend"),
            "bias_source": ctx.get("bias_source"),
        }

    # -----------------------------
    # Decision: AI / Technical
    # -----------------------------
    @staticmethod
    def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
        if direction == "BUY":
            return sl < entry < tp
        if direction == "SELL":
            return tp < entry < sl
        return False

    def _technical_gate_decision(self, cfg_eff: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
        direction = str(pkg.get("direction", "NONE"))
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)

        entry = _safe_float(pkg.get("entry_candidate"), 0.0)
        sl = _safe_float(pkg.get("stop_candidate"), 0.0)
        tp = _safe_float(pkg.get("tp_candidate"), 0.0)

        ctx = pkg.get("context", {}) or {}
        bos = bool(ctx.get("bos", False))
        st_ok = bool(ctx.get("supertrend_ok", False))

        min_score = _safe_float(cfg_eff.get("min_score", 7.0), 7.0)
        min_rr = _safe_float(cfg_eff.get("min_rr", 2.0), 2.0)
        conf_th = int(cfg_eff.get("confidence_threshold", 75))
        conf_py = int(pkg.get("confidence_py", 0))

        approved = (
            direction in ("BUY", "SELL")
            and bos
            and st_ok
            and score >= min_score
            and rr >= min_rr
            and conf_py >= conf_th
            and self._sanity(direction, entry, sl, tp)
        )

        reasoning = [
            "AI disabled -> Technical Gate",
            f"BOS={bos}, SuperTrendOK={st_ok}",
            f"Score={score:.1f} (min={min_score:.1f}), RR={rr:.2f} (min={min_rr:.2f})",
            f"confidence_py={conf_py} (th={conf_th})",
        ]

        warnings = []
        if not bos:
            warnings.append("BOS required but false")
        if not st_ok:
            warnings.append("SuperTrend conflict")
        if score < min_score:
            warnings.append("Score below min")
        if rr < min_rr:
            warnings.append("RR below min")
        if conf_py < conf_th:
            warnings.append("confidence_py below threshold")
        if not self._sanity(direction, entry, sl, tp):
            warnings.append("price sanity failed")

        return {
            "approved": bool(approved),
            "confidence": int(conf_py),
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "reasoning": "\n".join(reasoning),
            "warnings": "\n".join(warnings),
        }

    # -----------------------------
    # Order execution
    # -----------------------------
    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Dict[str, Any]:
        tick = self._get_tick(symbol)
        price = float(tick.ask if direction == "BUY" else tick.bid)
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 30,
            "magic": 20260226,
            "comment": "HIM v2.4.2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "error": "order_send returned None", "last_error": mt5.last_error()}

        ok = (result.retcode == mt5.TRADE_RETCODE_DONE)
        return {
            "ok": bool(ok),
            "retcode": int(result.retcode),
            "order": int(getattr(result, "order", 0)),
            "deal": int(getattr(result, "deal", 0)),
            "price": float(price),
            "comment": str(getattr(result, "comment", "")),
            "request": {
                "symbol": symbol,
                "direction": direction,
                "volume": float(lot),
                "sl": float(sl),
                "tp": float(tp),
            },
        }

    # -----------------------------
    # Telegram message (approved only)
    # -----------------------------
    @staticmethod
    def _build_mentor_message(symbol: str, direction: str, pkg: Dict[str, Any], decision: Dict[str, Any], enable_execution: bool, lot: float) -> str:
        ctx = pkg.get("context", {}) or {}
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)

        confidence = int(decision.get("confidence", 0))
        entry = _safe_float(decision.get("entry", 0.0), 0.0)
        sl = _safe_float(decision.get("sl", 0.0), 0.0)
        tp = _safe_float(decision.get("tp", 0.0), 0.0)

        st_dir = str(ctx.get("supertrend_dir", "NA"))
        st_val = _safe_float(ctx.get("supertrend_value", 0.0), 0.0)
        st_ok = bool(ctx.get("supertrend_ok", False))

        status = "APPROVED"
        if not enable_execution:
            status += " (DRY-RUN)"

        msg = []
        msg.append(f"HIM Signal — {status}")
        msg.append(f"Symbol: {symbol}")
        msg.append(f"Direction: {direction}")
        msg.append(f"Lot: {lot:.2f}")
        msg.append(f"Confidence: {confidence}%")
        msg.append(f"Entry: {entry:.2f}")
        msg.append(f"SL: {sl:.2f}")
        msg.append(f"TP: {tp:.2f}")
        msg.append("")
        msg.append("Mentor Explanation")
        msg.append(f"1) HTF={ctx.get('HTF_trend')} MTF={ctx.get('MTF_trend')} LTF={ctx.get('LTF_trend')} (bias={ctx.get('bias_source')})")
        msg.append(f"2) BOS={ctx.get('bos')} Retest={ctx.get('retest_ok')} Breakout={ctx.get('breakout_state')}({ctx.get('breakout_side')})")
        msg.append(f"3) SuperTrend dir={st_dir} value={st_val:.2f} ok={st_ok}")
        msg.append(f"4) Score={score:.1f}/10 RR={rr:.2f} blocked_by={ctx.get('blocked_by')}")
        return "\n".join(msg)

    # -----------------------------
    # Main loop
    # -----------------------------
    def run_once(self) -> None:
        cfg_eff = self.hcfg.load_effective()
        logger.info(f"CFG_EFFECTIVE: {self._cfg_snapshot(cfg_eff)}")

        symbol = str(cfg_eff.get("symbol", "GOLD"))
        enable_execution = bool(cfg_eff.get("enable_execution", False))
        conf_th = int(cfg_eff.get("confidence_threshold", 75))

        ai_cfg = (cfg_eff.get("ai") or {}) if isinstance(cfg_eff, dict) else {}
        ai_enabled = bool(ai_cfg.get("enabled", False))

        exec_cfg = (cfg_eff.get("execution") or {}) if isinstance(cfg_eff, dict) else {}
        max_spread_points = _safe_float(exec_cfg.get("max_spread_points", 60), 60)
        block_if_position_exists = bool(exec_cfg.get("block_if_position_exists", True))

        # 1) Engine package
        try:
            pkg = self.engine.generate_signal_package()
        except Exception as e:
            logger.error(f"CRITICAL: engine failed: {e}")
            logger.error(traceback.format_exc())
            self.tlog.append({"type": "engine_error", "symbol": symbol, "error": str(e), "traceback": traceback.format_exc()})
            return

        direction = str(pkg.get("direction", "NONE"))
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)
        confidence_py = _safe_int(pkg.get("confidence_py", 0), 0)

        logger.info(f"SIGNAL: {direction} score={score:.1f} rr={rr:.2f} confidence_py={confidence_py}")
        self.tlog.append({"type": "signal", "symbol": symbol, "package": pkg})

        # 2) If NONE, log reasons (NEW)
        if direction not in ("BUY", "SELL"):
            none_ctx = self._trace_none_context(pkg)
            logger.info(f"NONE_REASON: {none_ctx}")
            self.tlog.append({"type": "none_reason", "symbol": symbol, "detail": none_ctx})
            logger.info("No clear bias. Skip.")
            return

        # 3) Position check
        if block_if_position_exists and self._positions_exist(symbol):
            logger.info("Gate: position exists. Skip (no stacking).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "position_exists"})
            return

        # 4) Spread
        sp = self._spread_points(symbol)
        if sp > max_spread_points:
            logger.info(f"Gate: spread too high ({sp:.1f} > {max_spread_points:.1f} points).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "spread_too_high", "spread_points": sp})
            return

        # 5) Decision
        if ai_enabled:
            ai_resp = self.ai.approve_trade(pkg)
            if not AIMentor.validate_response(ai_resp):
                logger.error("CRITICAL: AI response invalid shape. Skip.")
                self.tlog.append({"type": "ai_invalid", "symbol": symbol, "package": pkg, "ai_raw": ai_resp})
                return
            decision = ai_resp
            source = "AI"
        else:
            decision = self._technical_gate_decision(cfg_eff, pkg)
            source = "TECHNICAL_GATE"

        approved = bool(decision["approved"])
        confidence = int(decision["confidence"])
        entry = float(decision["entry"])
        sl = float(decision["sl"])
        tp = float(decision["tp"])

        logger.info(f"DECISION({source}): approved={approved} confidence={confidence}%")
        if decision.get("reasoning"):
            logger.info("REASONING:\n" + str(decision["reasoning"]))
        if decision.get("warnings"):
            logger.info("WARNINGS:\n" + str(decision["warnings"]))

        self.tlog.append({"type": "decision", "symbol": symbol, "source": source, "decision": decision})

        # 6) Final gates
        if not approved:
            logger.info("Gate: rejected.")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "not_approved", "source": source})
            return
        if confidence < conf_th:
            logger.info(f"Gate: confidence below threshold ({confidence} < {conf_th}).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "confidence_below_threshold"})
            return

        ok_stops, why = self._validate_stops_levels(symbol, direction, entry, sl, tp)
        if not ok_stops:
            logger.info(f"Gate: stops validation failed: {why}")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "stops_validation_failed", "detail": why})
            return

        # 7) Lot
        fallback_lot = float(cfg_eff.get("lot", 0.01) or 0.01)
        lot, lot_reason = self._calc_lot(cfg_eff, symbol, direction, entry, sl, fallback_lot)
        logger.info(f"LOT: {lot:.2f} ({lot_reason})")
        self.tlog.append({"type": "lot_calc", "symbol": symbol, "lot": lot, "detail": lot_reason})

        # 8) Telegram (approved only)
        tg_sig = f"{symbol}|{direction}|{entry:.2f}|{sl:.2f}|{tp:.2f}|{lot:.2f}"
        msg = self._build_mentor_message(symbol, direction, pkg, decision, enable_execution=enable_execution, lot=lot)
        self._safe_send_telegram_trade(cfg_eff, msg, sig=tg_sig)

        # 9) Execute
        if not enable_execution:
            logger.info("Gate: enable_execution=false (dry-run).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "execution_disabled"})
            return

        exec_sig = f"{symbol}|{direction}|{entry:.2f}|{sl:.2f}|{tp:.2f}"
        if not self._execution_dedupe_ok(cfg_eff, exec_sig):
            logger.info("Gate: execution dedupe/cooldown suppress.")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "exec_dedupe"})
            return

        order_res = self._place_market_order(symbol, direction, lot, sl, tp)
        if order_res.get("ok"):
            logger.info(f"ORDER OK: ticket={order_res.get('order')} price={order_res.get('price')}")
        else:
            logger.error(f"ORDER FAILED: {order_res}")

        self.tlog.append({"type": "order", "symbol": symbol, "direction": direction, "lot": lot, "order_result": order_res})

    def run_loop(self, interval_sec: int = 15) -> None:
        logger.info("HIM MentorExecutor loop started.")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"CRITICAL LOOP ERROR: {e}")
                logger.error(traceback.format_exc())
                self.tlog.append({"type": "loop_error", "error": str(e), "traceback": traceback.format_exc()})
            time.sleep(interval_sec)


if __name__ == "__main__":
    ex = MentorExecutor()
    ex.run_loop(interval_sec=15)