"""
api_server.py
Version: 3.2.2

CHANGELOG
- 3.2.2 (2026-03-01)
  - ADD: /api/health (simple readiness probe)
  - FIX: POST /api/config requires confirm_live="CONFIRM LIVE" when enabling enable_execution=true
  - ADD: /api/status adds execution_mode ("LIVE"/"DRY_RUN") and telegram.enabled status
  - KEEP: GET "/" serve dashboard.html, merge update config, MT5 snapshot, /api/performance, /api/ai_confirm schema v1.0

File: api_server.py
Path: C:\Hybrid_Intelligence_Mentor\api_server.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request, send_from_directory

# project modules (local imports)
from ai_mentor import AIMentor
from config_resolver import resolve_effective_config

# optional MT5
try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None  # type: ignore


app = Flask(__name__)
CONFIG_PATH_DEFAULT = "config.json"
CONFIRM_LIVE_TEXT = "CONFIRM LIVE"


# -----------------------------
# Static / Dashboard (FIX 404)
# -----------------------------
@app.get("/")
def root_dashboard():
    return send_from_directory(".", "dashboard.html")


@app.get("/favicon.ico")
def favicon():
    if os.path.exists("favicon.ico"):
        return send_from_directory(".", "favicon.ico")
    return ("", 204)


# -----------------------------
# Helpers: JSON file
# -----------------------------
def _load_json_file(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_json_file(path: str, data: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on", "enable", "enabled")


# -----------------------------
# MT5 status helper
# -----------------------------
def _mt5_status_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mt5_ok": False,
        "account_login": None,
        "account_server": None,
        "account_currency": None,
        "account_leverage": None,
        "positions_count": None,
        "last_error": None,
    }

    if mt5 is None:
        out["last_error"] = "MetaTrader5 module not available"
        return out

    try:
        if not mt5.initialize():
            out["last_error"] = str(mt5.last_error())
            return out

        acc = mt5.account_info()
        if acc is None:
            out["last_error"] = "account_info is None"
            return out

        out["mt5_ok"] = True
        out["account_login"] = getattr(acc, "login", None)
        out["account_server"] = getattr(acc, "server", None)
        out["account_currency"] = getattr(acc, "currency", None)
        out["account_leverage"] = getattr(acc, "leverage", None)

        try:
            out["positions_count"] = int(mt5.positions_total())
        except Exception:
            out["positions_count"] = None

        return out
    except Exception as e:
        out["last_error"] = f"mt5_status_error: {e}"
        return out
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


# -----------------------------
# Helpers used by /api/ai_confirm
# -----------------------------
def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _norm_dir(x: Any) -> str:
    s = str(x or "").strip().upper()
    return s if s in ("BUY", "SELL") else "NONE"


def _rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return abs(tp - entry) / risk


# -----------------------------
# API: Health
# -----------------------------
@app.get("/api/health")
def api_health():
    cfg_eff = resolve_effective_config(CONFIG_PATH_DEFAULT) or {}
    mt5_snap = _mt5_status_snapshot()
    return jsonify(
        {
            "ok": True,
            "ts": time.time(),
            "config_loaded": bool(cfg_eff),
            "mt5_ok": bool(mt5_snap.get("mt5_ok")),
        }
    ), 200


# -----------------------------
# API: Config
# -----------------------------
@app.get("/api/config")
def api_get_config():
    cfg = _load_json_file(CONFIG_PATH_DEFAULT)
    return jsonify(cfg), 200


def _require_confirm_live_if_needed(patch: Dict[str, Any], current_effective: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Return (ok, error_message). Enforce confirm_live for enable_execution=true transitions.
    """
    wants_enable = patch.get("enable_execution", None)
    if wants_enable is None:
        return True, ""

    wants_enable_b = _boolish(wants_enable)
    if not wants_enable_b:
        # disabling is always allowed
        return True, ""

    # wants enable=True
    already_enabled = bool(current_effective.get("enable_execution", False))
    if already_enabled:
        return True, ""

    confirm = patch.get("confirm_live", None)
    if confirm != CONFIRM_LIVE_TEXT:
        return False, "confirm_live_required: send confirm_live='CONFIRM LIVE' to enable live execution"

    return True, ""


@app.post("/api/config")
def api_set_config():
    """
    Merge update (สำคัญ):
    - Dashboard ส่ง patch หรือส่งทั้งก้อนก็ได้
    - กัน config key หายจากการเขียนทับทั้งไฟล์
    - Safety: ถ้าจะเปิด enable_execution=true ต้องมี confirm_live="CONFIRM LIVE"
    """
    patch = request.get_json(silent=True) or {}
    if not isinstance(patch, dict):
        return jsonify({"ok": False, "error": "config_payload_must_be_object"}), 400

    # enforce confirm live (server-side)
    cfg_eff = resolve_effective_config(CONFIG_PATH_DEFAULT) or {}
    ok_confirm, err = _require_confirm_live_if_needed(patch, cfg_eff)
    if not ok_confirm:
        return jsonify({"ok": False, "error": err}), 400

    # do not persist confirm_live into config.json
    if "confirm_live" in patch:
        patch = dict(patch)
        patch.pop("confirm_live", None)

    current = _load_json_file(CONFIG_PATH_DEFAULT)
    merged = _deep_merge(current, patch)
    ok = _save_json_file(CONFIG_PATH_DEFAULT, merged)
    return jsonify({"ok": ok}), (200 if ok else 500)


@app.post("/api/mode/<mode_id>")
def api_set_mode(mode_id: str):
    cfg = _load_json_file(CONFIG_PATH_DEFAULT)
    cfg["mode"] = str(mode_id)
    ok = _save_json_file(CONFIG_PATH_DEFAULT, cfg)
    return jsonify({"ok": ok, "mode": cfg.get("mode")}), (200 if ok else 500)


# -----------------------------
# API: Status / Signal Preview / Performance
# -----------------------------
@app.get("/api/status")
def api_status():
    cfg_eff = resolve_effective_config(CONFIG_PATH_DEFAULT) or {}
    mt5_snap = _mt5_status_snapshot()

    enable_exec = bool(cfg_eff.get("enable_execution", False))
    exec_mode = "LIVE" if enable_exec else "DRY_RUN"

    telegram_cfg = (cfg_eff.get("telegram", {}) or {}) if isinstance(cfg_eff.get("telegram", {}), dict) else {}
    telegram_enabled = bool(telegram_cfg.get("enabled", False))

    return jsonify(
        {
            "ok": True,
            "ts": time.time(),
            "config": {
                "mode": cfg_eff.get("mode"),
                "symbol": cfg_eff.get("symbol", "GOLD"),
                "enable_execution": enable_exec,
                "execution_mode": exec_mode,
                "confidence_threshold": cfg_eff.get("confidence_threshold"),
                "min_score": cfg_eff.get("min_score"),
                "min_rr": cfg_eff.get("min_rr"),
                "lot": cfg_eff.get("lot") or (cfg_eff.get("execution", {}) or {}).get("volume"),
                "telegram": {
                    "enabled": telegram_enabled,
                },
            },
            "mt5": mt5_snap,
        }
    ), 200


@app.get("/api/signal_preview")
def api_signal_preview():
    cfg_eff = resolve_effective_config(CONFIG_PATH_DEFAULT) or {}
    return jsonify(
        {
            "ok": True,
            "ts": time.time(),
            "symbol": cfg_eff.get("symbol", "GOLD"),
            "mode": cfg_eff.get("mode"),
            "note": "signal preview placeholder (engine snapshot implemented elsewhere)",
        }
    ), 200


@app.get("/api/performance")
def api_performance():
    resp: Dict[str, Any] = {
        "ok": True,
        "ts": time.time(),
        "trades_today": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "profit": 0.0,
        "loss_amount": 0.0,
        "net": 0.0,
        "source": "default_zero",
    }

    try:
        import performance_tracker  # type: ignore

        if hasattr(performance_tracker, "get_daily_report"):
            rep = performance_tracker.get_daily_report()  # type: ignore
            if isinstance(rep, dict):
                resp.update(rep)
                resp["source"] = "performance_tracker.get_daily_report"
                return jsonify(resp), 200

        if hasattr(performance_tracker, "PerformanceTracker"):
            pt = performance_tracker.PerformanceTracker()  # type: ignore
            if hasattr(pt, "daily_summary"):
                rep = pt.daily_summary()  # type: ignore
                if isinstance(rep, dict):
                    resp.update(rep)
                    resp["source"] = "PerformanceTracker.daily_summary"
                    return jsonify(resp), 200
    except Exception:
        pass

    return jsonify(resp), 200


# -----------------------------
# API: AI Confirm (Schema v1.0)
# -----------------------------
@app.post("/api/ai_confirm")
def api_ai_confirm():
    payload = request.get_json(silent=True) or {}

    direction = _norm_dir(payload.get("direction"))
    entry = _sf(payload.get("entry"))
    sl = _sf(payload.get("sl"))
    tp = _sf(payload.get("tp"))
    lot = _sf(payload.get("lot"), 0.0)
    atr = _sf(payload.get("atr"), 0.0)
    mode = str(payload.get("mode") or payload.get("regime") or "")

    if direction not in ("BUY", "SELL") or entry == 0.0 or sl == 0.0 or tp == 0.0:
        return jsonify(
            {
                "schema_version": "1.0",
                "decision": "REJECT",
                "confidence": 0.0,
                "note": "missing_baseline_fields(direction/entry/sl/tp)",
            }
        ), 200

    ai_pkg = {
        "symbol": payload.get("symbol", "GOLD"),
        "baseline": {
            "dir": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": _rr(entry, sl, tp),
            "atr": atr,
            "lot": lot,
            "mode": mode,
        },
        "constraints": {
            "min_rr": 1.5,
            "entry_shift_max_atr": _sf(payload.get("entry_shift_max_atr"), 0.20),
        },
        "context": {
            "watch_state": payload.get("watch_state"),
            "breakout_state": payload.get("breakout_state"),
            "htf_trend": payload.get("htf_trend"),
            "mtf_trend": payload.get("mtf_trend"),
            "ltf_trend": payload.get("ltf_trend"),
            "proximity_score": payload.get("proximity_score"),
        },
    }

    mentor = AIMentor()
    out = mentor.evaluate(ai_pkg) or {}
    ex = out.get("execution") or {}

    conf_pct = _sf(ex.get("conf"), 0.0)
    conf01 = max(0.0, min(1.0, conf_pct / 100.0))

    decision = "CONFIRM"
    if bool(ex.get("reject", False)) or conf01 <= 0.0:
        decision = "REJECT"

    resp = {
        "schema_version": "1.0",
        "decision": decision,
        "confidence": conf01,
        "entry": ex.get("entry", entry),
        "sl": ex.get("sl", sl),
        "tp": ex.get("tp", tp),
        "note": (out.get("mentor") or {}).get("headline") or "AIMentor",
    }
    return jsonify(resp), 200


def main() -> int:
    host = os.environ.get("HIM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("HIM_API_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())