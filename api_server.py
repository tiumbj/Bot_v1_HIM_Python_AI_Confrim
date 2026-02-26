"""
api_server.py — HIM Intelligent Dashboard API
Version: 1.2.0

Changelog:
- 1.2.0 (2026-02-26):
  - FIX: /api/mode endpoint (now supports /api/mode/<mode_id> correctly)
  - UX: rename modes to human-friendly IDs: precision / balanced / frequent
  - COMPAT: still accepts legacy A/B/C mode IDs (A->precision, B->balanced, C->frequent)
  - STRATEGY: add breakout config keys designed for Intelligent Dashboard buttons:
      breakout.confirm_buffer_atr
      breakout.require_retest
      breakout.retest_band_atr
      breakout.proximity_threshold_atr
      breakout.proximity_min_score
  - SAFETY: atomic config write, normalized defaults, stable /api/status snapshot
Notes:
- Never calls mt5.shutdown()
- Telegram test default remains plain text (parse_mode=None)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request, send_from_directory

import MetaTrader5 as mt5

from engine import TradingEngine
from telegram_notifier import TelegramNotifier
from performance_tracker import PerformanceTracker


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("him_system.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("HIM")


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DASHBOARD_HTML = os.path.join(BASE_DIR, "dashboard.html")

app = Flask(__name__)
mt5_lock = threading.Lock()

engine = TradingEngine(config_path=CONFIG_PATH)
tg = TelegramNotifier(config_path=CONFIG_PATH)


# -----------------------------
# JSON utils
# -----------------------------
def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"CRITICAL: cannot read json: {path} err={e}")
        return {}


def _safe_write_json_atomic(path: str, data: Dict[str, Any]) -> bool:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.error(f"CRITICAL: cannot write json: {path} err={e}")
        return False


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst) if isinstance(dst, dict) else {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _is_blank_dict(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 0


# -----------------------------
# Config normalization
# -----------------------------
def _normalize_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Replace None/invalid values with sane defaults (commissioning safe)
    Return: (normalized_cfg, changed_flag)
    """
    changed = False
    out = dict(cfg) if isinstance(cfg, dict) else {}

    def set_if_missing_or_none(key: str, default: Any) -> None:
        nonlocal changed, out
        if key not in out or out.get(key) is None:
            out[key] = default
            changed = True

    def ensure_dict(key: str, default: Dict[str, Any]) -> None:
        nonlocal changed, out
        v = out.get(key)
        if v is None or not isinstance(v, dict) or _is_blank_dict(v):
            out[key] = dict(default)
            changed = True

    def ensure_float(d: Dict[str, Any], key: str, default: float) -> bool:
        nonlocal changed
        v = d.get(key)
        if v is None:
            d[key] = float(default)
            changed = True
            return True
        try:
            d[key] = float(v)
            return False
        except Exception:
            d[key] = float(default)
            changed = True
            return True

    def ensure_int(d: Dict[str, Any], key: str, default: int) -> bool:
        nonlocal changed
        v = d.get(key)
        if v is None:
            d[key] = int(default)
            changed = True
            return True
        try:
            d[key] = int(float(v))
            return False
        except Exception:
            d[key] = int(default)
            changed = True
            return True

    # Top-level defaults
    set_if_missing_or_none("mode", "balanced")
    set_if_missing_or_none("symbol", "GOLD")
    set_if_missing_or_none("enable_execution", False)
    ensure_int(out, "confidence_threshold", 75)
    ensure_float(out, "min_score", 7.0)
    ensure_float(out, "min_rr", 2.0)
    ensure_float(out, "lot", 0.01)

    # Nested defaults
    ensure_dict(out, "timeframes", {"htf": "H4", "mtf": "H1", "ltf": "M15"})
    ensure_dict(out, "risk", {"atr_period": 14, "atr_sl_mult": 1.8})
    ensure_dict(out, "supertrend", {"period": 10, "multiplier": 3.0})

    # Breakout/Option-C knobs for dashboard buttons
    ensure_dict(
        out,
        "breakout",
        {
            "confirm_buffer_atr": 0.0,      # 0.0 = strict close > pivot
            "require_retest": True,         # Option C default
            "retest_band_atr": 0.25,        # band around ref for retest
            "proximity_threshold_atr": 0.8, # proximity threshold in ATR units
            "proximity_min_score": 50,      # 0..100
        },
    )
    if isinstance(out.get("breakout"), dict):
        b = out["breakout"]
        ensure_float(b, "confirm_buffer_atr", 0.0)
        # bool: accept only real bool, else coerce
        if b.get("require_retest") is None:
            b["require_retest"] = True
            changed = True
        else:
            b["require_retest"] = bool(b.get("require_retest"))
        ensure_float(b, "retest_band_atr", 0.25)
        ensure_float(b, "proximity_threshold_atr", 0.8)
        ensure_int(b, "proximity_min_score", 50)

    # AI defaults (status only)
    ensure_dict(
        out,
        "ai",
        {
            "enabled": False,
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "timeout_seconds": 10,
            "retry_attempts": 3,
            "fallback_to_technical": True,
            "model": "",
        },
    )
    if isinstance(out.get("ai"), dict):
        ai = out["ai"]
        ai["enabled"] = bool(ai.get("enabled", False))
        if not (ai.get("provider") or "").strip():
            ai["provider"] = "openai"
            changed = True
        if not (ai.get("api_key_env") or "").strip():
            ai["api_key_env"] = "OPENAI_API_KEY"
            changed = True
        if ai.get("model") is None:
            ai["model"] = ""
            changed = True

    # Telegram defaults
    ensure_dict(
        out,
        "telegram",
        {
            "enabled": True,
            "token_env": "TELEGRAM_BOT_TOKEN",
            "chat_id_env": "TELEGRAM_CHAT_ID",
            "notify_on": ["signal", "trade", "error"],
            "cooldown_sec": 900,
        },
    )

    return out, changed


def _load_config() -> Dict[str, Any]:
    raw = _safe_read_json(CONFIG_PATH)
    cfg, changed = _normalize_config(raw)
    if changed:
        ok = _safe_write_json_atomic(CONFIG_PATH, cfg)
        logger.info(f"Config normalized (changed=True) write_back={ok}")
    return cfg


def _save_config_patch(patch: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(patch, dict):
        return False, "patch_not_object"
    cur = _load_config()
    merged = _deep_merge(cur, patch)
    merged, _ = _normalize_config(merged)
    ok = _safe_write_json_atomic(CONFIG_PATH, merged)
    return ok, ("ok" if ok else "write_failed")


# -----------------------------
# MT5 helpers
# -----------------------------
def _ensure_mt5() -> Tuple[bool, str]:
    try:
        if mt5.terminal_info() is not None:
            return True, "already_initialized"
        ok = mt5.initialize()
        if not ok:
            return False, f"initialize_failed: {mt5.last_error()}"
        return True, "initialized"
    except Exception as e:
        return False, f"exception: {e}"


def _get_tick(symbol: str) -> Dict[str, Any]:
    ok, msg = _ensure_mt5()
    if not ok:
        return {"ok": False, "error": msg, "last_error": str(mt5.last_error())}
    t = mt5.symbol_info_tick(symbol)
    if t is None:
        return {"ok": False, "error": f"tick_none for {symbol}", "last_error": str(mt5.last_error())}
    return {
        "ok": True,
        "symbol": symbol,
        "time_msc": int(getattr(t, "time_msc", 0)),
        "bid": float(getattr(t, "bid", 0.0)),
        "ask": float(getattr(t, "ask", 0.0)),
        "last": float(getattr(t, "last", 0.0)),
    }


def _mt5_snapshot(symbol: str) -> Dict[str, Any]:
    ok, msg = _ensure_mt5()
    snap: Dict[str, Any] = {
        "mt5_ok": ok,
        "mt5_state": msg,
        "symbol": symbol,
        "account": None,
        "positions": [],
        "positions_count": 0,
        "last_error": str(mt5.last_error()),
    }
    if not ok:
        return snap

    acc = mt5.account_info()
    if acc:
        snap["account"] = {
            "login": getattr(acc, "login", None),
            "balance": float(getattr(acc, "balance", 0.0)),
            "equity": float(getattr(acc, "equity", 0.0)),
            "margin": float(getattr(acc, "margin", 0.0)),
            "margin_free": float(getattr(acc, "margin_free", 0.0)),
            "profit": float(getattr(acc, "profit", 0.0)),
            "currency": getattr(acc, "currency", None),
        }

    positions = mt5.positions_get(symbol=symbol)
    if positions:
        snap["positions_count"] = len(positions)
        for p in positions[:30]:
            snap["positions"].append(
                {
                    "ticket": int(getattr(p, "ticket", 0)),
                    "type": "BUY" if int(getattr(p, "type", 0)) == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": float(getattr(p, "volume", 0.0)),
                    "price_open": float(getattr(p, "price_open", 0.0)),
                    "price_current": float(getattr(p, "price_current", 0.0)),
                    "sl": float(getattr(p, "sl", 0.0)),
                    "tp": float(getattr(p, "tp", 0.0)),
                    "profit": float(getattr(p, "profit", 0.0)),
                }
            )
    return snap


def _build_dry_run_order(symbol: str, direction: str) -> Dict[str, Any]:
    tick = _get_tick(symbol)
    if not tick.get("ok"):
        return {"ok": False, "error": tick.get("error"), "tick": tick}
    entry = tick["ask"] if direction == "BUY" else tick["bid"]
    return {
        "ok": True,
        "symbol": symbol,
        "direction": direction,
        "entry_ref": entry,
        "note": "DRY-RUN only (no order_send). SL/TP must come from engine/executor policy.",
        "tick": tick,
    }


# -----------------------------
# Modes (new names + legacy compat)
# -----------------------------
MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # เข้มงวด/เทรดน้อย/คุณภาพสูง
    "precision": {
        "mode": "precision",
        "confidence_threshold": 85,
        "min_score": 8.5,
        "min_rr": 2.5,
        "breakout": {
            "confirm_buffer_atr": 0.10,
            "require_retest": True,
            "retest_band_atr": 0.20,
            "proximity_threshold_atr": 0.70,
            "proximity_min_score": 60,
        },
    },
    # สมดุล (default)
    "balanced": {
        "mode": "balanced",
        "confidence_threshold": 75,
        "min_score": 7.0,
        "min_rr": 2.0,
        "breakout": {
            "confirm_buffer_atr": 0.05,
            "require_retest": True,
            "retest_band_atr": 0.25,
            "proximity_threshold_atr": 0.80,
            "proximity_min_score": 50,
        },
    },
    # ถี่ขึ้น/ยอมให้สัญญาณมากขึ้น (ยังคง Option C)
    "frequent": {
        "mode": "frequent",
        "confidence_threshold": 65,
        "min_score": 6.0,
        "min_rr": 1.7,
        "breakout": {
            "confirm_buffer_atr": 0.00,
            "require_retest": True,
            "retest_band_atr": 0.30,
            "proximity_threshold_atr": 1.00,
            "proximity_min_score": 40,
        },
    },
}

LEGACY_MODE_MAP = {"A": "precision", "B": "balanced", "C": "frequent"}


def _resolve_mode_id(mode_id: str) -> str:
    m = (mode_id or "").strip()
    if not m:
        return "balanced"
    u = m.upper()
    if u in LEGACY_MODE_MAP:
        return LEGACY_MODE_MAP[u]
    return m.lower()


# -----------------------------
# Status helpers
# -----------------------------
def _ai_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ai = cfg.get("ai") if isinstance(cfg, dict) else {}
    if not isinstance(ai, dict):
        ai = {}
    provider = str(ai.get("provider") or "openai").strip().lower()
    enabled = bool(ai.get("enabled", False))
    api_key_env = str(ai.get("api_key_env") or ("OPENAI_API_KEY" if provider == "openai" else "AI_API_KEY")).strip()
    model = str(ai.get("model") or "").strip()
    key_present = bool(os.getenv(api_key_env)) if api_key_env else False
    ok = bool(enabled and key_present and bool(model))
    return {
        "ok": ok,
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "key_present": key_present,
        "note": "Status only (no API call).",
    }


def _telegram_status() -> Dict[str, Any]:
    try:
        enabled, token, chat_id, notify_on = tg._resolve_credentials()  # type: ignore[attr-defined]
        return {
            "ok": bool(enabled and token and chat_id),
            "enabled": bool(enabled),
            "has_token": bool(token),
            "has_chat_id": bool(chat_id),
            "notify_on": notify_on,
            "note": "Status only (no send).",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    if os.path.exists(DASHBOARD_HTML):
        return send_from_directory(BASE_DIR, "dashboard.html")
    return jsonify({"error": "dashboard.html not found"}), 404


@app.get("/dashboard.html")
def dashboard_html():
    if os.path.exists(DASHBOARD_HTML):
        return send_from_directory(BASE_DIR, "dashboard.html")
    return jsonify({"error": "dashboard.html not found"}), 404


@app.get("/api/config")
def api_get_config():
    return jsonify(_load_config())


@app.post("/api/config")
def api_post_config():
    patch = request.get_json(silent=True) or {}
    ok, msg = _save_config_patch(patch)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    return jsonify({"ok": True})


@app.post("/api/mode/<mode_id>")
def api_apply_mode(mode_id: str):
    resolved = _resolve_mode_id(mode_id)
    preset = MODE_PRESETS.get(resolved)
    if not preset:
        return jsonify({"ok": False, "error": f"unknown_mode: {mode_id}"}), 404
    ok, msg = _save_config_patch(preset)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    return jsonify({"ok": True, "mode": resolved})


@app.post("/api/mode")
def api_apply_mode_json():
    payload = request.get_json(silent=True) or {}
    mode_id = str(payload.get("mode") or "")
    resolved = _resolve_mode_id(mode_id)
    preset = MODE_PRESETS.get(resolved)
    if not preset:
        return jsonify({"ok": False, "error": f"unknown_mode: {mode_id}"}), 404
    ok, msg = _save_config_patch(preset)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    return jsonify({"ok": True, "mode": resolved})


@app.get("/api/ai_status")
def api_ai_status():
    cfg = _load_config()
    return jsonify(_ai_status(cfg))


@app.get("/api/telegram_status")
def api_telegram_status():
    return jsonify(_telegram_status())


@app.get("/api/status")
def api_status():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))
    with mt5_lock:
        mt5snap = _mt5_snapshot(symbol)
        price = _get_tick(symbol)

    return jsonify(
        {
            "ts": int(time.time()),
            "config": {
                "mode": cfg.get("mode"),
                "symbol": symbol,
                "enable_execution": bool(cfg.get("enable_execution", False)),
                "confidence_threshold": int(cfg.get("confidence_threshold", 75)),
                "min_score": float(cfg.get("min_score", 7.0)),
                "min_rr": float(cfg.get("min_rr", 2.0)),
                "lot": float(cfg.get("lot", 0.01)),
                "timeframes": cfg.get("timeframes", {}),
                "supertrend": cfg.get("supertrend", {}),
                "risk": cfg.get("risk", {}),
                "breakout": cfg.get("breakout", {}),
            },
            "mt5": mt5snap,
            "price": price,
            "ai": _ai_status(cfg),
            "telegram": _telegram_status(),
            "log_file": {"exists": os.path.exists(os.path.join(BASE_DIR, "him_system.log")), "path": os.path.join(BASE_DIR, "him_system.log")},
        }
    )


@app.get("/api/signal_preview")
def api_signal_preview():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))
    with mt5_lock:
        try:
            pkg = engine.generate_signal_package()
        except Exception as e:
            logger.error(f"CRITICAL: engine.generate_signal_package failed: {e}")
            return jsonify({"ok": False, "error": f"engine_failed: {e}"}), 500

    direction = str((pkg or {}).get("direction", "NONE")).upper()
    dry = _build_dry_run_order(symbol=symbol, direction=direction) if direction in ("BUY", "SELL") else {"ok": True, "note": "direction=NONE"}
    return jsonify({"ok": True, "symbol": symbol, "package": pkg, "dry_run_order": dry})


@app.get("/api/performance")
def api_performance():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))
    with mt5_lock:
        try:
            pt = PerformanceTracker(stats_file="performance_stats.json", symbol=symbol, lookback_days=365)
            pt.sync_trade_closes()
            s = pt.summary()

            trades_raw = pt.stats.get("trades", []) if isinstance(pt.stats, dict) else []
            trades = [t for t in trades_raw if isinstance(t, dict) and t.get("deal_ticket") is not None and t.get("time_iso")]
            trades = sorted(trades, key=lambda x: int(x.get("time", 0)), reverse=True)[:20]
        except Exception as e:
            logger.error(f"CRITICAL: performance failed: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify(
        {
            "ok": True,
            "symbol": symbol,
            "summary": {
                "total_trades": s.total_trades,
                "wins": s.wins,
                "losses": s.losses,
                "win_rate": s.win_rate,
                "net_profit": s.net_profit,
                "gross_profit": s.gross_profit,
                "gross_loss": s.gross_loss,
                "profit_factor": s.profit_factor,
                "best_trade": s.best_trade,
                "worst_trade": s.worst_trade,
            },
            "recent": trades,
        }
    )


@app.post("/api/telegram_test")
def api_telegram_test():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "HIM Telegram Test: OK\n(plain text)")
    event_type = str(payload.get("event_type") or "signal")
    ok, status, body = tg.send_text_debug(text=text, event_type=event_type, parse_mode=None)
    return jsonify(
        {
            "ok": bool(ok),
            "event_type": event_type,
            "telegram_http_status": int(status),
            "telegram_response_snippet": (body or "")[:400],
            "telegram": _telegram_status(),
        }
    )


def main() -> None:
    logger.info("HIM API Server starting...")
    logger.info("Open: http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()