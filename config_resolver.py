"""
HIM Config Resolver
Version: 1.0.0

Purpose:
- Create "effective config" by deep-merging:
  base(root) + profiles[mode] overrides
- Enforce consistent runtime behavior across engine/executor/telegram/etc.

Rules:
- If mode missing => use base config only
- If profiles[mode] missing => use base config only
- Deep-merge dicts, overwrite scalars/lists
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge: base <- override
    - dict merges recursively
    - list/scalar overwrite
    """
    out: Dict[str, Any] = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def resolve_effective_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return effective config = (root without profiles) + profiles[mode].
    Keeps original cfg untouched.
    """
    if not isinstance(cfg, dict):
        return {}

    mode = str(cfg.get("mode", "")).strip().upper()
    profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles", {}), dict) else {}

    # Base = root config without "profiles" (so profile doesn't get nested into runtime)
    base = deepcopy(cfg)
    base.pop("profiles", None)

    if not mode:
        return base

    prof = profiles.get(mode)
    if not isinstance(prof, dict):
        return base

    # Merge profile over base
    effective = _deep_merge(base, prof)

    # Ensure effective.mode is the chosen mode (normalize)
    effective["mode"] = mode
    return effective


def summarize_effective_config(effective: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small snapshot for logs/debug (safe to print).
    """
    tf = effective.get("timeframes", {}) or {}
    risk = effective.get("risk", {}) or {}
    return {
        "mode": effective.get("mode"),
        "symbol": effective.get("symbol"),
        "enable_execution": effective.get("enable_execution"),
        "confidence_threshold": effective.get("confidence_threshold"),
        "min_score": effective.get("min_score"),
        "min_rr": effective.get("min_rr"),
        "lot": effective.get("lot"),
        "timeframes": {"htf": tf.get("htf"), "mtf": tf.get("mtf"), "ltf": tf.get("ltf")},
        "atr_period": risk.get("atr_period"),
        "atr_sl_mult": risk.get("atr_sl_mult"),
    }