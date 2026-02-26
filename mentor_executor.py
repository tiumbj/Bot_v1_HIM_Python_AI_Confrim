"""
Mentor Executor - Execution Controller (FINAL SPEC + Telegram + Trade Logging)
Version: 2.2.2

Changelog:
- 2.2.0: Implement effective config resolver (root + profiles[mode]) and log config snapshot
- 2.2.1: Fix mentor message formatting crash when ctx fields are None (safe formatting helpers)
- 2.2.2: Add full traceback logging + isolate mentor message build/send to prevent loop crash

Rules enforced:
- Execute ONLY if: approved == true AND confidence >= threshold AND enable_execution == true
- AI response MUST be strictly validated before execution
- No silent freeze: all critical failures logged (with traceback)
"""

from __future__ import annotations

import os
import json
import time
import logging
import traceback
from typing import Any, Dict

import MetaTrader5 as mt5

from engine import TradingEngine
from ai_mentor import AIMentor
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger
from config_resolver import resolve_effective_config, summarize_effective_config


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


def _fmt_float(v: Any, decimals: int = 2, na: str = "NA") -> str:
    """
    Safe numeric formatter:
    - None / non-numeric => "NA"
    - numeric => formatted string
    """
    try:
        if v is None:
            return na
        x = float(v)
        fmt = "{:." + str(int(decimals)) + "f}"
        return fmt.format(x)
    except Exception:
        return na


class HotConfig:
    def __init__(self, path: str):
        self.path = path
        self._mtime = 0.0
        self._cache: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
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


class MentorExecutor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_dir, "config.json")
        self.hcfg = HotConfig(self.config_path)

        self.engine = TradingEngine(config_path=self.config_path)
        self.ai = AIMentor()
        self.tg = TelegramNotifier(config_path=self.config_path)
        self.tlog = TradeLogger(os.path.join(base_dir, "trade_history.json"))

        self._ensure_mt5()

    def _ensure_mt5(self) -> None:
        if mt5.terminal_info() is not None:
            return
        if not mt5.initialize():
            err = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {err}")
        logger.info("MT5 Connected")

    def _get_tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for symbol: {symbol}")
        return tick

    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Dict[str, Any]:
        info = mt5.symbol_info(symbol)
        if info is None:
            return {"ok": False, "error": f"Symbol not found: {symbol}"}

        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                return {"ok": False, "error": f"Cannot select symbol: {symbol}"}

        tick = self._get_tick(symbol)
        price = tick.ask if direction == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 30,
            "magic": 20260225,
            "comment": "HIM v2.2.2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "error": "order_send returned None"}

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

    @staticmethod
    def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
        if direction == "BUY":
            return sl < entry < tp
        if direction == "SELL":
            return tp < entry < sl
        return False

    @staticmethod
    def _build_mentor_message(symbol: str, direction: str, pkg: Dict[str, Any], ai_resp: Dict[str, Any]) -> str:
        """
        MUST NOT CRASH (even when fields missing/None)
        """
        ctx = pkg.get("context", {}) or {}

        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)

        confidence = int(ai_resp.get("confidence", 0))
        entry = _safe_float(ai_resp.get("entry", 0.0), 0.0)
        sl = _safe_float(ai_resp.get("sl", 0.0), 0.0)
        tp = _safe_float(ai_resp.get("tp", 0.0), 0.0)

        rsi_s = _fmt_float(ctx.get("rsi"), decimals=1, na="NA")
        vol_s = _fmt_float(ctx.get("vol_ratio"), decimals=2, na="NA")

        msg = []
        msg.append("HIM Signal")
        msg.append(f"Symbol: {symbol}")
        msg.append(f"Direction: {direction}")
        msg.append(f"Confidence: {confidence}%")
        msg.append(f"Entry: {entry:.2f}")
        msg.append(f"SL: {sl:.2f}")
        msg.append(f"TP: {tp:.2f}")
        msg.append("")
        msg.append("Mentor Explanation")
        msg.append(f"1) HTF trend: {ctx.get('HTF_trend')} | MTF: {ctx.get('MTF_trend')} | LTF: {ctx.get('LTF_trend')}")
        msg.append(f"2) CHoCH: {ctx.get('choch')} | BOS(required): {ctx.get('bos')}")
        msg.append(f"3) Zone: {('มี' if ctx.get('zone') else 'ไม่มี')} | SuperTrend: {ctx.get('supertrend')} (ok={ctx.get('supertrend_ok')})")
        msg.append(f"4) Momentum: RSI={rsi_s} | VolRatio={vol_s}")
        msg.append(f"5) Score={score:.1f}/10 | RR={rr:.2f}")

        ai_reason = (ai_resp.get("reasoning") or "").strip()
        ai_warn = (ai_resp.get("warnings") or "").strip()

        if ai_reason:
            msg.append("")
            msg.append("AI Reasoning")
            msg.append(ai_reason)

        if ai_warn:
            msg.append("")
            msg.append("Warnings")
            msg.append(ai_warn)

        return "\n".join(msg)

    def _safe_telegram(self, symbol: str, direction: str, pkg: Dict[str, Any], ai_resp: Dict[str, Any]) -> None:
        """
        Isolate mentor message + telegram send so loop cannot crash.
        Logs full traceback if something fails.
        """
        try:
            mentor_msg = self._build_mentor_message(symbol, direction, pkg, ai_resp)
        except Exception as e:
            logger.error(f"CRITICAL: build_mentor_message failed: {e}")
            logger.error("TRACEBACK:\n" + traceback.format_exc())
            self.tlog.append({"type": "mentor_message_error", "error": str(e), "package": pkg, "ai": ai_resp})
            return

        try:
            tg_ok = self.tg.send_text(mentor_msg)
            if not tg_ok:
                logger.warning("Telegram send failed or disabled (check config telegram.enabled and env vars).")
                self.tlog.append({"type": "telegram_failed", "symbol": symbol, "direction": direction})
        except Exception as e:
            logger.error(f"CRITICAL: telegram send_text failed: {e}")
            logger.error("TRACEBACK:\n" + traceback.format_exc())
            self.tlog.append({"type": "telegram_send_error", "error": str(e), "symbol": symbol, "direction": direction})

    def run_once(self) -> None:
        raw_cfg = self.hcfg.load()
        cfg = resolve_effective_config(raw_cfg)
        snap = summarize_effective_config(cfg)
        logger.info(f"CFG_EFFECTIVE: {snap}")

        symbol = str(cfg.get("symbol", "GOLD"))
        enable_execution = bool(cfg.get("enable_execution", False))
        conf_th = int(cfg.get("confidence_threshold", 75))
        lot = float(cfg.get("lot", 0.01))

        # 1) Engine signal package
        try:
            pkg = self.engine.generate_signal_package()
        except Exception as e:
            logger.error(f"CRITICAL: engine failed: {e}")
            logger.error("TRACEBACK:\n" + traceback.format_exc())
            self.tlog.append({"type": "engine_error", "symbol": symbol, "error": str(e)})
            return

        direction = str(pkg.get("direction", "NONE"))
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)
        logger.info(f"SIGNAL: {direction} score={score:.1f} rr={rr:.2f}")

        self.tlog.append({"type": "signal", "symbol": symbol, "package": pkg})

        if direction not in ("BUY", "SELL"):
            logger.info("No clear bias. Skip.")
            return

        # 2) AI approval (strict JSON only)
        ai_resp = self.ai.approve_trade(pkg)
        if not AIMentor.validate_response(ai_resp):
            logger.error("CRITICAL: AI response invalid shape. Skip execution.")
            self.tlog.append({"type": "ai_invalid", "symbol": symbol, "package": pkg, "ai_raw": ai_resp})
            return

        approved = bool(ai_resp["approved"])
        confidence = int(ai_resp["confidence"])
        entry = _safe_float(ai_resp["entry"], 0.0)
        sl = _safe_float(ai_resp["sl"], 0.0)
        tp = _safe_float(ai_resp["tp"], 0.0)

        logger.info(f"AI: approved={approved} confidence={confidence}%")
        if ai_resp["reasoning"]:
            logger.info("REASONING:\n" + ai_resp["reasoning"])
        if ai_resp["warnings"]:
            logger.info("WARNINGS:\n" + ai_resp["warnings"])

        self.tlog.append({"type": "ai_decision", "symbol": symbol, "direction": direction, "ai": ai_resp})

        # 3) Mentor message + Telegram (safe; never crash loop)
        self._safe_telegram(symbol, direction, pkg, ai_resp)

        # 4) Final safety gate
        if not approved:
            logger.info("Gate: rejected by AI.")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "ai_rejected", "ai": ai_resp})
            return

        if confidence < conf_th:
            logger.info(f"Gate: confidence below threshold ({confidence} < {conf_th}).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "confidence_below_threshold", "ai": ai_resp})
            return

        if not enable_execution:
            logger.info("Gate: enable_execution=false (dry-run).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "execution_disabled", "ai": ai_resp})
            return

        if not self._sanity(direction, entry, sl, tp):
            logger.error("CRITICAL: price sanity check failed. Skip execution.")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "sanity_failed", "ai": ai_resp})
            return

        # 5) Execute
        order_res = self._place_market_order(symbol=symbol, direction=direction, lot=lot, sl=sl, tp=tp)
        if order_res.get("ok"):
            logger.info(f"ORDER OK: ticket={order_res.get('order')} price={order_res.get('price')}")
        else:
            logger.error(f"ORDER FAILED: {order_res}")

        self.tlog.append(
            {
                "type": "order",
                "symbol": symbol,
                "direction": direction,
                "enable_execution": enable_execution,
                "confidence_threshold": conf_th,
                "effective_config": snap,
                "ai": ai_resp,
                "order_result": order_res,
            }
        )

    def run_loop(self, interval_sec: int = 15) -> None:
        logger.info("HIM MentorExecutor loop started.")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"CRITICAL LOOP ERROR: {e}")
                logger.error("TRACEBACK:\n" + traceback.format_exc())
                self.tlog.append({"type": "loop_error", "error": str(e)})
            time.sleep(interval_sec)


if __name__ == "__main__":
    ex = MentorExecutor()
    ex.run_loop(interval_sec=15)