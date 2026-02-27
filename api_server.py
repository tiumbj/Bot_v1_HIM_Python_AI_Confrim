"""
api_server.py — HIM Intelligent Dashboard API
Version: 1.6.0

Changelog:
- 1.6.0 (2026-02-27):
  - FEATURE: integrate PerformanceTracker (logs/signals.csv, logs/trades.csv, logs/performance_summary.json)
  - UPDATE: /api/signal_preview logs signal snapshot for statistical validation
  - UPDATE: /api/performance now returns computed summary (ready when trades resolved)
  - KEEP: _get_engine() creates fresh engine instance per request (no cache)
  - KEEP: Never calls mt5.shutdown()

Notes:
- Statistical validation uses MT5 history to resolve TP/SL hit-first after signal timestamp.
- This API does NOT place trades.

"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Tuple

import MetaTrader5 as mt5
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from engine import TradingEngine
from performance_tracker import PerformanceTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DASHBOARD_HTML = os.path.join(BASE_DIR, "dashboard.html")

app = Flask(__name__)
CORS(app)

mt5_lock = threading.Lock()
engine_lock = threading.Lock()

_perf = PerformanceTracker()


def _get_engine() -> TradingEngine:
    # Always create fresh engine instance
    # Prevent stale class/version caching
    return TradingEngine(CONFIG_PATH)


def _load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _ensure_dict(obj: Dict[str, Any], key: str, default: Dict[str, Any]) -> None:
    if key not in obj or not isinstance(obj.get(key), dict):
        obj[key] = dict(default)


def _save_config_patch(patch: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(patch, dict):
        return False, "patch_not_dict"

    cfg = _load_config()
    if not isinstance(cfg, dict):
        cfg = {}

    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v

    _ensure_dict(cfg, "_meta", {"config_version": "unknown"})
    _ensure_dict(cfg, "timeframes", {"htf": "H4", "mtf": "H1", "ltf": "M15"})
    _ensure_dict(cfg, "supertrend", {"period": 10, "mult": 3.0})
    _ensure_dict(cfg, "risk", {"atr_period": 14, "atr_sl_mult": 1.6})
    _ensure_dict(cfg, "breakout", {})
    _ensure_dict(cfg, "continuation", {"enabled": False})
    _ensure_dict(cfg, "sideway_scalp", {"enabled": False})
    _ensure_dict(cfg, "telegram", {"enabled": False})
    _ensure_dict(cfg, "ai", {"enabled": False})

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True, "ok"
    except Exception as e:
        return False, f"write_failed: {e}"


def _mt5_snapshot(symbol: str) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"ok": False}
    if not mt5.initialize():
        snap["error"] = "mt5_init_failed"
        return snap

    info = mt5.symbol_info(symbol)
    if info is None:
        snap["error"] = f"symbol_info_none: {symbol}"
        return snap

    tick = mt5.symbol_info_tick(symbol)
    positions = mt5.positions_get(symbol=symbol)

    snap["ok"] = True
    snap["symbol"] = symbol
    snap["digits"] = int(getattr(info, "digits", 0))
    snap["spread"] = int(getattr(tick, "spread", 0)) if tick else None
    snap["ask"] = float(getattr(tick, "ask", 0.0)) if tick else None
    snap["bid"] = float(getattr(tick, "bid", 0.0)) if tick else None

    snap["positions"] = []
    if positions:
        for p in positions:
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


def _get_tick(symbol: str) -> Dict[str, Any]:
    if not mt5.initialize():
        return {"ok": False, "error": "mt5_init_failed"}
    t = mt5.symbol_info_tick(symbol)
    if not t:
        return {"ok": False, "error": "tick_none"}
    return {"ok": True, "ask": float(t.ask), "bid": float(t.bid), "time_msc": int(getattr(t, "time_msc", 0))}


MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "sideway_scalp": {
        "mode": "sideway_scalp",
        "confidence_threshold": 60,
        "min_score": 5.5,
        "min_rr": 1.5,
        "risk": {"atr_period": 14, "atr_sl_mult": 1.6},
        "timeframes": {"htf": "H1", "mtf": "M15", "ltf": "M5"},
        "sideway_scalp": {
            "enabled": True,
            "adx_period": 14,
            "adx_max": 22.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_width_atr_max": 6.0,
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "require_confirmation": True,
            "touch_buffer_atr": 0.10,
        },
        "continuation": {"enabled": False},
    }
}
LEGACY_MODE_MAP = {"S": "sideway_scalp"}


def _resolve_mode_id(mode_id: str) -> str:
    m = (mode_id or "").strip()
    if not m:
        return "sideway_scalp"
    u = m.upper()
    if u in LEGACY_MODE_MAP:
        return LEGACY_MODE_MAP[u]
    return m.lower()


@app.get("/")
def root():
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


@app.get("/api/status")
def api_status():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))
    with mt5_lock:
        mt5snap = _mt5_snapshot(symbol)
        price = _get_tick(symbol)
    return jsonify({"ts": int(time.time()), "config": cfg, "mt5": mt5snap, "price": price})


@app.get("/api/signal_preview")
def api_signal_preview():
    """
    Generate a signal snapshot and log it to logs/signals.csv for statistical validation.
    """
    try:
        cfg = _load_config()
        tf_exec = str(((cfg.get("timeframes") or {}).get("ltf")) or "M5")

        eng = _get_engine()
        pkg = eng.generate_signal_package()

        # Log signal snapshot (both passed & blocked are logged; evaluation resolves only passed)
        try:
            _perf.append_signal_from_engine_package(pkg, tf_exec=tf_exec)
        except Exception:
            # Do not break signal endpoint if logger fails
            pass

        return jsonify({"ok": True, "signal": pkg})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/performance")
def api_performance():
    """
    Resolve past signals -> trades (TP/SL hit-first) and return summary.
    ready=True when at least 1 trade resolved.
    """
    cfg = _load_config()
    horizon = int(request.args.get("horizon", "360") or "360")

    with mt5_lock:
        try:
            out = _perf.update_and_summarize(max_horizon_minutes=horizon)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "ts": int(time.time()), "config_mode": cfg.get("mode")}), 500

    summary = (out.get("summary") or {}) if isinstance(out, dict) else {}
    return jsonify(
        {
            "ok": True,
            "ready": bool(summary.get("ready", False)),
            "resolved": int(out.get("resolved", 0) or 0),
            "summary": summary,
            "config_mode": cfg.get("mode"),
            "ts": int(time.time()),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)