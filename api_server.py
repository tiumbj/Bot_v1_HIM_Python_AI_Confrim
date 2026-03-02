"""
Hybrid Intelligence Mentor (HIM)
File: api_server.py
Version: v3.2.5 (External Dashboard Link Redirect)
Date: 2026-03-02 (Asia/Bangkok)

CHANGELOG
- v3.2.5:
  - Remove heavy local dashboard serving.
  - Add lightweight redirect endpoints:
      * GET /          -> redirect to external dashboard URL (if configured)
      * GET /dashboard -> redirect to external dashboard URL (if configured)
  - If external URL missing -> show plain text with a copyable link placeholder.
  - Keep API endpoints intact (status/config/health etc.) for production.

RATIONALE (Production)
- ลดภาระระบบ: ไม่ต้องมี dashboard.html, static files, UI mapping, polling UI logic
- ปลอดภัย: API ยังคงเป็น source of truth, UI แยก host (website) ได้เลย
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify, redirect, Response

try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None  # API จะ fail-closed ใน health/status บางส่วน
    _mt5_import_error = str(e)


APP_VERSION = "v3.2.5"
DEFAULT_CONFIG_PATH = os.environ.get("HIM_CONFIG", "config.json")

app = Flask(__name__)


# -------------------------
# Utilities
# -------------------------

def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def cfg_get(cfg: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def mt5_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Best-effort snapshot. Fail-closed: if MT5 not available, return minimal diagnostic.
    """
    snap: Dict[str, Any] = {
        "ok": False,
        "symbol": symbol,
        "ts": utc_iso_now(),
    }

    if mt5 is None:
        snap["error"] = f"MetaTrader5_import_failed: {_mt5_import_error}"
        return snap

    if not mt5.initialize():
        snap["error"] = "mt5.initialize_failed"
        return snap

    try:
        term = mt5.terminal_info()
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)

        snap.update({
            "ok": True,
            "terminal": {
                "trade_allowed": bool(getattr(term, "trade_allowed", False)) if term else False,
                "connected": bool(getattr(term, "connected", False)) if term else False,
                "path": getattr(term, "path", None) if term else None,
            },
            "account": {
                "trade_allowed": bool(getattr(acc, "trade_allowed", False)) if acc else False,
                "login": getattr(acc, "login", None) if acc else None,
                "balance": float(getattr(acc, "balance", 0.0)) if acc else None,
                "equity": float(getattr(acc, "equity", 0.0)) if acc else None,
            },
            "symbol_info": {
                "visible": bool(getattr(sym, "visible", False)) if sym else False,
                "trade_mode": int(getattr(sym, "trade_mode", -1)) if sym else None,
            },
            "tick": {
                "time": int(getattr(tick, "time", 0)) if tick else None,
                "bid": float(getattr(tick, "bid", 0.0)) if tick else None,
                "ask": float(getattr(tick, "ask", 0.0)) if tick else None,
                "last": float(getattr(tick, "last", 0.0)) if tick else None,
            }
        })
        return snap
    except Exception as e:
        snap["error"] = f"{type(e).__name__}: {e}"
        snap["trace"] = traceback.format_exc(limit=2)
        return snap
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def external_dashboard_url(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Read external dashboard URL from config:
      dashboard.external_url
    """
    url = cfg_get(cfg, ["dashboard", "external_url"], None)
    if not url:
        return None
    url = str(url).strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    return url


# -------------------------
# Routes: External Dashboard Redirect (Lightest)
# -------------------------

@app.get("/")
def root() -> Response:
    cfg = safe_load_json(DEFAULT_CONFIG_PATH)
    url = external_dashboard_url(cfg)
    if url:
        return redirect(url, code=302)

    # เบาสุด: plain text
    return Response(
        "External Dashboard URL not configured.\n"
        "Set config.json: dashboard.external_url = \"https://your-dashboard-website\"\n",
        mimetype="text/plain",
        status=200
    )


@app.get("/dashboard")
def dashboard_redirect() -> Response:
    cfg = safe_load_json(DEFAULT_CONFIG_PATH)
    url = external_dashboard_url(cfg)
    if url:
        return redirect(url, code=302)

    return Response(
        "External Dashboard URL not configured.\n"
        "Set config.json: dashboard.external_url = \"https://your-dashboard-website\"\n",
        mimetype="text/plain",
        status=200
    )


# -------------------------
# API: Health / Status / Config
# -------------------------

@app.get("/api/health")
def api_health() -> Response:
    return jsonify({
        "ok": True,
        "service": "HIM api_server",
        "version": APP_VERSION,
        "ts": utc_iso_now(),
        "config_path": DEFAULT_CONFIG_PATH,
    })


@app.get("/api/status")
def api_status() -> Response:
    cfg = safe_load_json(DEFAULT_CONFIG_PATH)
    symbol = str(cfg.get("symbol", "GOLD"))

    enable_execution = bool(cfg.get("enable_execution", False))
    execution_mode = "LIVE" if enable_execution else "DRY_RUN"

    status: Dict[str, Any] = {
        "ok": True,
        "version": APP_VERSION,
        "ts": utc_iso_now(),
        "symbol": symbol,
        "enable_execution": enable_execution,
        "execution_mode": execution_mode,
        "telegram": {
            "enabled": bool(cfg_get(cfg, ["telegram", "enabled"], False))
        },
        # IMPORTANT: keep mt5 block for operator visibility
        "mt5": mt5_snapshot(symbol),
        "dashboard": {
            "external_url": external_dashboard_url(cfg)
        }
    }
    return jsonify(status)


@app.get("/api/config")
def api_get_config() -> Response:
    cfg = safe_load_json(DEFAULT_CONFIG_PATH)
    return jsonify({
        "ok": True,
        "ts": utc_iso_now(),
        "config": cfg
    })


@app.post("/api/config")
def api_set_config() -> Response:
    """
    Update config.json with provided JSON object. Fail-closed:
    - body must be JSON object
    - only writes if valid
    """
    try:
        body = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "config_must_be_object"}), 400

    # minimal write
    safe_write_json(DEFAULT_CONFIG_PATH, body)
    return jsonify({"ok": True, "ts": utc_iso_now()})


def main() -> int:
    host = os.environ.get("HIM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("HIM_API_PORT", "5000"))
    debug = bool(os.environ.get("HIM_API_DEBUG", "").strip())

    print(json.dumps({
        "ts": utc_iso_now(),
        "msg": "api_server_start",
        "version": APP_VERSION,
        "host": host,
        "port": port,
        "config_path": DEFAULT_CONFIG_PATH,
        "note": "Dashboard serving removed. Use external dashboard URL in config.json.",
    }, ensure_ascii=False))

    app.run(host=host, port=port, debug=debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())