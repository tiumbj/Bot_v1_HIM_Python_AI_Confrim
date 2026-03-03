"""
Hybrid Intelligence Mentor (HIM)
File: mentor_executor.py
Version: v2.13.4 (PRODUCTION MODEL - Schema Router for Engine v2.12.x + Legacy Support)
Date: 2026-03-03 (Asia/Bangkok)

CHANGELOG
- v2.13.4:
  - PRODUCTION: Keep v2.13.3 structure (autodiscovery, compact JSON, fail-closed, engine_dbg)
  - NEW: Schema Router
      A) Engine v2.12.x (top-level): decision/blocked_by/gates/metrics
      B) Legacy schema (context.*): keep extract_from_signal_package()
  - FIX: Prevent commissioning blindness (decision/blocked_by/obs.metrics/obs.gates no longer null when engine provides them)
  - SAFETY: confirm-only (does NOT place orders)

EVIDENCE (runtime symptom to prevent)
- Previously executor returned ok=true but decision=null, metrics/gates null in logs (commissioning tools saw null fields).
- Root cause: extractor assumed legacy 'context' schema; engine v2.12.x returns top-level schema.

SAFETY
- Confirm-only: does NOT place orders.
- Fail-closed: exception -> ok=false JSON with short trace + engine_dbg.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_blocked_by(x: Any) -> List[str]:
    """
    Normalize to list[str] and de-duplicate (preserve order).
    Supports:
    - list/tuple/set
    - comma-separated string
    - None
    """
    items: List[str] = []
    if x is None:
        items = []
    elif isinstance(x, (list, tuple, set)):
        for it in x:
            s = str(it).strip()
            if s:
                items.append(s)
    else:
        s = str(x).strip()
        if s:
            parts = [p.strip() for p in s.split(",")]
            items = [p for p in parts if p]

    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out


# -----------------------------------------------------------------------------
# Engine auto-discovery caller (best-effort, schema-agnostic)
# -----------------------------------------------------------------------------

def _to_dict_any(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        return out
    if hasattr(out, "to_dict") and callable(getattr(out, "to_dict")):
        d = out.to_dict()
        return d if isinstance(d, dict) else {"repr": repr(d)}
    if hasattr(out, "__dict__"):
        return dict(out.__dict__)
    return {"repr": repr(out)}


def _call_method_safely(obj: Any, name: str, symbol: str, cfg: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return False, None, "not_callable"

    try:
        try:
            sig = inspect.signature(fn)
            params = [p for p in sig.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            argc = len(params)  # bound method already binds self
        except Exception:
            argc = -1

        tries: List[Tuple[Tuple[Any, ...], str]] = []
        if argc == 0:
            tries = [((), "0args")]
        elif argc == 1:
            tries = [((symbol,), "1arg_symbol")]
        elif argc >= 2:
            tries = [((symbol, cfg), "2args_symbol_cfg"), ((symbol,), "1arg_symbol"), ((), "0args")]
        else:
            tries = [((symbol, cfg), "2args_symbol_cfg"), ((symbol,), "1arg_symbol"), ((), "0args")]

        last_err = ""
        for args, tag in tries:
            try:
                out = fn(*args)
                d = _to_dict_any(out)
                return True, d, tag
            except TypeError as te:
                last_err = f"TypeError:{te}"
                continue
            except Exception as e:
                return False, None, f"{tag}:{type(e).__name__}:{e}"

        return False, None, last_err or "no_compatible_signature"
    except Exception as e:
        return False, None, f"invoke_failed:{type(e).__name__}:{e}"


def call_engine_autodiscovery(config_path: str, cfg: Dict[str, Any], symbol: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    dbg: Dict[str, Any] = {
        "ok": False,
        "engine_ctor": None,
        "picked_method": None,
        "picked_call": None,
        "candidates": [],
        "errors": []
    }

    try:
        from engine import TradingEngine  # type: ignore
    except Exception as ex:
        dbg["errors"].append(f"import_engine_failed:{type(ex).__name__}:{ex}")
        return None, dbg

    engine_instances: List[Tuple[str, Any]] = []
    try:
        engine_instances.append(("TradingEngine(config_path)", TradingEngine(config_path)))
    except Exception as ex:
        dbg["errors"].append(f"ctor_path_failed:{type(ex).__name__}:{ex}")

    # dict-ctor may succeed for some engines; keep for compatibility
    try:
        if isinstance(cfg, dict) and cfg:
            engine_instances.append(("TradingEngine(config_dict)", TradingEngine(cfg)))
    except Exception as ex:
        dbg["errors"].append(f"ctor_dict_failed:{type(ex).__name__}:{ex}")

    if not engine_instances:
        return None, dbg

    preferred = ["generate_signal_package"]
    patterns = ["signal", "package", "eval", "run", "analy", "build"]

    def score_name(n: str) -> int:
        nlow = n.lower()
        score = 0
        for i, pn in enumerate(preferred):
            if nlow == pn.lower():
                score += 200 - i
        for p in patterns:
            if p in nlow:
                score += 10
        if nlow.startswith("_"):
            score -= 50
        return score

    for ctor_name, eng in engine_instances:
        names = [n for n in dir(eng) if not n.startswith("__")]
        scored = sorted([(score_name(n), n) for n in names], reverse=True)
        top = [n for s, n in scored if s > 0][:25]

        dbg["engine_ctor"] = ctor_name
        dbg["candidates"] = top

        for name in top:
            ok, d, call_tag = _call_method_safely(eng, name, symbol=symbol, cfg=cfg)
            if ok and isinstance(d, dict):
                dbg["ok"] = True
                dbg["picked_method"] = name
                dbg["picked_call"] = call_tag
                return d, dbg
            else:
                if call_tag and len(dbg["errors"]) < 12:
                    dbg["errors"].append(f"{name}:{call_tag}")

    return None, dbg


# -----------------------------------------------------------------------------
# Schema-aligned extraction (Legacy: context.*)
# -----------------------------------------------------------------------------

def extract_from_signal_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy schema alignment (kept from v2.13.3):
    Expected:
      pkg: direction, entry_candidate, stop_candidate, tp_candidate, rr, score, confidence_py,
      ctx: context{blocked_by,gates,metrics,...}
    """
    ctx = pkg.get("context", {})
    if not isinstance(ctx, dict):
        ctx = {}

    blocked_by_list = normalize_blocked_by(ctx.get("blocked_by"))

    direction = pkg.get("direction", None)  # e.g. BUY/SELL/NONE
    decision = direction

    entry = pkg.get("entry_candidate", None)
    sl = pkg.get("stop_candidate", None)
    tp = pkg.get("tp_candidate", None)
    rr = safe_float(pkg.get("rr", None), 0.0) or 0.0

    score = safe_float(pkg.get("score", None), 0.0) or 0.0
    confidence_py = safe_int(pkg.get("confidence_py", None), 0) or 0

    htf_trend = ctx.get("HTF_trend", None)
    mtf_trend = ctx.get("MTF_trend", None)
    ltf_trend = ctx.get("LTF_trend", None)

    supertrend_dir = ctx.get("supertrend_dir", None)
    supertrend_value = safe_float(ctx.get("supertrend_value", None), None)
    supertrend_ok = pkg.get("supertrend_ok", None)

    atr = safe_float(ctx.get("atr", None), None)
    atr_period = safe_int(ctx.get("atr_period", None), None)
    atr_sl_mult = safe_float(ctx.get("atr_sl_mult", None), None)
    min_rr = safe_float(ctx.get("min_rr", None), None)

    gates = ctx.get("gates", None)
    if not isinstance(gates, dict):
        gates = {}

    ctx_metrics = ctx.get("metrics", None)
    if not isinstance(ctx_metrics, dict):
        ctx_metrics = {}

    obs = {
        "engine_version": ctx.get("engine_version", None),
        "mode": ctx.get("mode", None),
        "bias_source": ctx.get("bias_source", None),
        "direction_bias": ctx.get("direction_bias", None),
        "watch_state": ctx.get("watch_state", None),

        "structure": {
            "htf_trend": htf_trend,
            "mtf_trend": mtf_trend,
            "ltf_trend": ltf_trend,
        },
        "metrics": {
            "atr": atr,
            "atr_period": atr_period,
            "supertrend_dir": supertrend_dir,
            "supertrend_value": supertrend_value,

            "event_timeframe": ctx_metrics.get("event_timeframe", None),
            "bb_width_points": ctx_metrics.get("bb_width_points", None),
            "bb_width_atr": ctx_metrics.get("bb_width_atr", None),
            "bb_width_atr_min": ctx_metrics.get("bb_width_atr_min", None),
            "vol_expansion_score": ctx_metrics.get("vol_expansion_score", None),
            "vol_expansion_threshold": ctx_metrics.get("vol_expansion_threshold", None),
            "vol_expansion_margin": ctx_metrics.get("vol_expansion_margin", None),

            "adx": ctx_metrics.get("adx_value", None),
            "plus_di": ctx_metrics.get("plus_di", None),
            "minus_di": ctx_metrics.get("minus_di", None),

            "supertrend_conflict": ctx_metrics.get("supertrend_conflict", None),
            "supertrend_distance_points": ctx_metrics.get("supertrend_distance_points", None),
            "supertrend_distance_atr": ctx_metrics.get("supertrend_distance_atr", None),
        },
        "thresholds": {
            "atr_sl_mult": atr_sl_mult,
            "min_rr": min_rr,
        },
        "gates": {
            "blocked_by": blocked_by_list,
            "supertrend_ok": supertrend_ok,
            "supertrend_conflict": ctx_metrics.get("supertrend_conflict", None),
            "bias_ok": gates.get("bias_ok", None),
            "vol_expansion_ok": gates.get("vol_expansion_ok", None),
            "bos_ok": gates.get("bos_ok", None),
            "bos_break_ok": gates.get("bos_break_ok", None),
            "retest_checked": gates.get("retest_checked", None),
            "retest_ok": gates.get("retest_ok", None),
            "retest_bypass_reason": gates.get("retest_bypass_reason", None),
            "proximity_checked": gates.get("proximity_checked", None),
            "proximity_ok": gates.get("proximity_ok", None),
            "proximity_bypass_reason": gates.get("proximity_bypass_reason", None),
            "rr_ok": gates.get("rr_ok", None),
        }
    }

    return {
        "decision": decision,
        "blocked_by": blocked_by_list,
        "confidence": score,
        "confidence_py": confidence_py,
        "plan": {"entry": entry, "sl": sl, "tp": tp, "rr": rr},
        "obs": obs,
    }


# -----------------------------------------------------------------------------
# Schema Router (NEW: Engine v2.12.x top-level schema)
# -----------------------------------------------------------------------------

def is_engine_v212_schema(pkg: Dict[str, Any]) -> bool:
    # Engine v2.12.x returns top-level: decision/blocked_by/gates/metrics
    if not isinstance(pkg, dict):
        return False
    if "decision" not in pkg:
        return False
    if "blocked_by" not in pkg:
        return False
    if "gates" not in pkg:
        return False
    return True


def extract_from_engine_v212(pkg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Engine v2.12.x schema:
      decision: "BLOCKED" | "PASS"
      blocked_by: list[str]
      gates: dict
      metrics: dict
    We map to mentor schema while keeping stable keys used by tools.
    """
    decision = pkg.get("decision", None)
    blocked_by_list = normalize_blocked_by(pkg.get("blocked_by", []))

    metrics = pkg.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    gates_src = pkg.get("gates", {})
    if not isinstance(gates_src, dict):
        gates_src = {}

    # Keep obs layout similar to legacy (structure/thresholds can be None here, but key must exist)
    obs = {
        "engine_version": pkg.get("engine_version", None),
        "mode": None,
        "bias_source": None,
        "direction_bias": pkg.get("bias", None),
        "watch_state": None,

        "structure": {
            "htf_trend": None,
            "mtf_trend": None,
            "ltf_trend": None,
        },
        "metrics": {
            # minimal commissioning-critical metrics
            "atr": safe_float(cfg_get(pkg, ["price", "atr"], None), None),
            "atr_period": None,

            "supertrend_dir": metrics.get("supertrend_dir_event", None),
            "supertrend_value": None,

            "event_timeframe": pkg.get("event_timeframe", None),
            "bb_width_points": None,
            "bb_width_atr": metrics.get("bb_width_atr", None),
            "bb_width_atr_min": metrics.get("bb_width_atr_min_used", None),

            "vol_expansion_score": None,
            "vol_expansion_threshold": metrics.get("bb_width_atr_min_used", None),
            "vol_expansion_margin": None,

            "adx": None,
            "plus_di": None,
            "minus_di": None,

            "supertrend_conflict": gates_src.get("supertrend_conflict", None),
            "supertrend_distance_points": None,
            "supertrend_distance_atr": metrics.get("supertrend_distance_atr", None),
        },
        "thresholds": {
            "atr_sl_mult": None,
            "min_rr": None,
        },
        "gates": {
            "blocked_by": blocked_by_list,
            "supertrend_ok": gates_src.get("supertrend_ok", None),
            "supertrend_conflict": gates_src.get("supertrend_conflict", None),

            # keep legacy key names (tools expect these keys to exist)
            "bias_ok": gates_src.get("bias_ok", None),
            "vol_expansion_ok": gates_src.get("vol_expansion_ok", None),

            "bos_ok": gates_src.get("bos_ok", None),
            "bos_break_ok": gates_src.get("bos_break_ok", None),

            "retest_checked": gates_src.get("retest_checked", None),
            "retest_ok": gates_src.get("retest_ok", None),
            "retest_bypass_reason": gates_src.get("retest_bypass_reason", None),

            "proximity_checked": gates_src.get("proximity_checked", None),
            "proximity_ok": gates_src.get("proximity_ok", None),
            "proximity_bypass_reason": gates_src.get("proximity_bypass_reason", None),

            "rr_ok": gates_src.get("rr_ok", None),
        }
    }

    # plan: engine v2.12.x may not provide entry/sl/tp; keep schema stable
    rr_val = safe_float(metrics.get("rr", None), None)
    plan = {"entry": None, "sl": None, "tp": None, "rr": (rr_val if rr_val is not None else 0.0)}

    # confidence: engine v2.12.x doesn't provide score by default; keep 0.0
    confidence = safe_float(metrics.get("score", 0.0), 0.0) or 0.0

    return {
        "decision": decision,
        "blocked_by": blocked_by_list,
        "confidence": confidence,
        "confidence_py": 0,
        "plan": plan,
        "obs": obs,
    }


def extract_router(pkg: Dict[str, Any]) -> Dict[str, Any]:
    if is_engine_v212_schema(pkg):
        return extract_from_engine_v212(pkg)
    return extract_from_signal_package(pkg)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()

    config_path = os.environ.get("HIM_CONFIG", "config.json")
    cfg = load_json(config_path, {})
    if not isinstance(cfg, dict):
        cfg = {}

    symbol_cfg = str(cfg.get("symbol", "GOLD"))
    live = bool(cfg_get(cfg, ["execution", "enable_execution"], cfg.get("enable_execution", False)))
    lot = safe_float(cfg_get(cfg, ["risk", "lot"], cfg.get("lot", None)), None)

    pkg, engine_dbg = call_engine_autodiscovery(config_path, cfg, symbol_cfg)

    if not isinstance(pkg, dict):
        payload = {
            "ts": utc_iso_now(),
            "type": "MENTOR_EXECUTOR",
            "version": "v2.13.4",
            "ok": False,
            "symbol": symbol_cfg,
            "live": live,
            "lot": lot,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "error": "ENGINE_CALL_FAILED",
            "engine_dbg": engine_dbg,
        }
        print(json_dumps_compact(payload))
        return 2

    # Schema Router (PRODUCTION)
    ex = extract_router(pkg)

    payload = {
        "ts": utc_iso_now(),
        "type": "MENTOR_EXECUTOR",
        "version": "v2.13.4",
        "ok": True,
        "symbol": str(pkg.get("symbol", symbol_cfg)),
        "live": live,
        "lot": lot,
        "elapsed_ms": int((time.time() - t0) * 1000),

        "decision": ex.get("decision", None),
        "blocked_by": ex.get("blocked_by", []),
        "confidence": ex.get("confidence", 0.0),
        "confidence_py": ex.get("confidence_py", 0),
        "plan": ex.get("plan", {"entry": None, "sl": None, "tp": None, "rr": 0.0}),
        "obs": ex.get("obs", {}),

        # Debug (short + useful)
        "engine_dbg": engine_dbg,
    }

    print(json_dumps_compact(payload))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        payload = {
            "ts": utc_iso_now(),
            "type": "MENTOR_EXECUTOR",
            "version": "v2.13.4",
            "ok": False,
            "error": f"UNCAUGHT:{type(e).__name__}:{e}",
            "trace": traceback.format_exc(limit=3),
        }
        print(json_dumps_compact(payload))
        raise SystemExit(2)