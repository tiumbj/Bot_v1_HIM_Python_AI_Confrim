"""
ai_bridge_v1.py
Version: 1.1.0

Changelog:
- 1.1.0 (2026-02-28)
  - NEW: Support AI Schema v1.0 response: {schema_version:"1.0", decision:"CONFIRM|REJECT", confidence:0..1, entry, sl, tp}
  - NEW: Enforce deterministic validator_v1_0 at boundary (fail-closed)
  - KEEP: Backward compatible with legacy JSON: {final_confirm, side, entry, sl, tp, confidence(0..100)}
  - KEEP: Backward compatible with text parsing fallback

Frozen decisions enforced here:
- AI confirm-only: cannot change direction/lot, bounded entry shift, SL tighten-only, RR floor >= 1.5
- Invalid/unparseable => final_confirm=False (fail closed)

Notes:
- For validator to work, caller must pass signal with baseline fields:
  direction, entry, sl, tp, lot (optional but recommended), atr (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from validator_v1_0 import ValidationPolicy, validate_ai_response_v1_0

logger = logging.getLogger("HIM")


@dataclass(frozen=True)
class AIConfirmDecision:
    final_confirm: bool
    side: str  # "BUY" or "SELL"
    entry: float
    sl: float
    tp: float
    confidence: float  # 0..100
    mentor_hint: str = ""


class AIConfirmClient:
    """
    Contract (supported):
    1) Schema v1.0 (preferred):
       {"schema_version":"1.0","decision":"CONFIRM|REJECT","confidence":0..1,"entry":..,"sl":..,"tp":..,"note":"..."}
    2) Legacy JSON:
       {"final_confirm":true,"side":"BUY","entry":..,"sl":..,"tp":..,"confidence":0..100,"mentor_hint":"..."}
    3) Legacy text:
       "BUY Entry 1234 SL 1220 TP 1260 Confidence 72% FINAL_CONFIRM=true"
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        self.api_url = str((self.cfg.get("ai") or {}).get("api_url", "")).strip()
        self.timeout_sec = int((self.cfg.get("ai") or {}).get("timeout_sec", 15))

        # Policy (defaults align frozen decisions)
        ai_cfg = (self.cfg.get("ai") or {})
        self.policy = ValidationPolicy(
            rr_floor=float(ai_cfg.get("min_rr", 1.5) or 1.5),
            entry_shift_max_atr_mult=float(ai_cfg.get("entry_shift_max_atr", 0.20) or 0.20),
            entry_shift_max_pct=float(ai_cfg.get("entry_shift_max_pct", 0.0) or 0.0),
            enforce_mode_lock=bool(ai_cfg.get("enforce_mode_lock", False)),
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.error(f"CRITICAL: AIConfirmClient config load failed: {e}")
            return {}

    def build_compact_payload(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep payload minimal to reduce token cost.
        Include baseline order fields if available to allow strict confirm-only.
        """
        ctx = (signal.get("context") or {}) if isinstance(signal, dict) else {}
        baseline = (signal.get("baseline") or {}) if isinstance(signal, dict) else {}

        payload = {
            "symbol": signal.get("symbol", "GOLD"),
            # baseline / engine suggestion
            "direction": baseline.get("direction") or signal.get("direction"),
            "entry": baseline.get("entry"),
            "sl": baseline.get("sl"),
            "tp": baseline.get("tp"),
            "lot": baseline.get("lot"),
            "rr": baseline.get("rr"),
            "atr": baseline.get("atr") or ctx.get("atr"),

            # context (compact)
            "side_hint": (ctx.get("proximity_side") or ctx.get("bias_side") or signal.get("direction") or "NONE"),
            "watch_state": ctx.get("watch_state", "NONE"),
            "breakout_state": ctx.get("breakout_state", "NONE"),
            "price_bid": ctx.get("price_bid"),
            "price_ask": ctx.get("price_ask"),
            "bos_high": ctx.get("bos_ref_high") or ctx.get("bos_high"),
            "bos_low": ctx.get("bos_ref_low") or ctx.get("bos_low"),
            "distance_buy": ctx.get("distance_buy"),
            "distance_sell": ctx.get("distance_sell"),
            "proximity_score": ctx.get("proximity_score"),
            "htf_trend": ctx.get("htf_trend"),
            "mtf_trend": ctx.get("mtf_trend"),
            "ltf_trend": ctx.get("ltf_trend"),
            "spread_points": ctx.get("spread_points"),
            "schema_required": "1.0",
        }
        return {k: v for k, v in payload.items() if v is not None}

    def request_confirm(self, signal: Dict[str, Any]) -> AIConfirmDecision:
        if not self.api_url:
            logger.error("AI api_url missing in config.json under ai.api_url")
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, "ai.api_url missing")

        payload = self.build_compact_payload(signal)

        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout_sec)
            text = (resp.text or "").strip()

            # Try JSON first
            try:
                data = resp.json()
                dec = self._parse_json_any(data, signal)
                if dec:
                    return dec
            except Exception:
                pass

            # Fallback to legacy text parse
            dec2 = self._parse_text_response(text)
            if dec2:
                return dec2

            logger.error(f"AI response unparseable: status={resp.status_code} body={text[:500]}")
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, "ai response unparseable")

        except Exception as e:
            logger.error(f"AI request failed: {e}")
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, f"ai exception: {e}")

    # -------------------------
    # Parsing + Validation
    # -------------------------
    def _parse_json_any(self, data: Any, signal: Dict[str, Any]) -> Optional[AIConfirmDecision]:
        if not isinstance(data, dict):
            return None

        # Preferred: schema v1.0
        if str(data.get("schema_version", "")).strip() == "1.0":
            return self._parse_schema_v1_0(data, signal)

        # Legacy JSON
        return self._parse_legacy_json(data)

    def _parse_schema_v1_0(self, data: Dict[str, Any], signal: Dict[str, Any]) -> Optional[AIConfirmDecision]:
        # Need baseline/engine fields to validate strictly
        baseline = (signal.get("baseline") or {}) if isinstance(signal, dict) else {}
        engine_order = {
            "direction": baseline.get("direction") or signal.get("direction"),
            "entry": baseline.get("entry"),
            "sl": baseline.get("sl"),
            "tp": baseline.get("tp"),
            "lot": baseline.get("lot", 0.0),
            "atr": baseline.get("atr") or ((signal.get("context") or {}).get("atr") if isinstance(signal, dict) else None),
            "mode": baseline.get("mode") or (signal.get("mode")),
        }

        # If essential fields missing -> fail closed
        if engine_order.get("direction") not in ("BUY", "SELL"):
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, "engine direction missing for validation")
        if engine_order.get("entry") is None or engine_order.get("sl") is None or engine_order.get("tp") is None:
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, "engine entry/sl/tp missing for validation")

        vr = validate_ai_response_v1_0(data, engine_order, policy=self.policy)

        if not vr.ok or vr.decision != "CONFIRM":
            hint = ""
            try:
                hint = f"errors={list(vr.errors)} reasons={list(vr.reasons)}"
            except Exception:
                hint = "validator reject"
            return AIConfirmDecision(False, str(engine_order.get("direction", "NONE")), float(engine_order["entry"]), float(engine_order["sl"]), float(engine_order["tp"]), 0.0, hint)

        # confirmed: use effective fields (ai may omit => engine values)
        norm = vr.normalized
        eff_entry = norm["ai"]["entry"] if norm["ai"]["entry"] is not None else norm["engine"]["entry"]
        eff_sl = norm["ai"]["sl"] if norm["ai"]["sl"] is not None else norm["engine"]["sl"]
        eff_tp = norm["ai"]["tp"] if norm["ai"]["tp"] is not None else norm["engine"]["tp"]

        conf01 = float(norm.get("confidence") or 0.0)
        conf_pct = max(0.0, min(100.0, conf01 * 100.0))
        mentor_hint = str((norm.get("ai") or {}).get("note") or "")

        return AIConfirmDecision(True, str(engine_order["direction"]), float(eff_entry), float(eff_sl), float(eff_tp), float(conf_pct), mentor_hint)

    def _parse_legacy_json(self, data: Dict[str, Any]) -> Optional[AIConfirmDecision]:
        try:
            fc = bool(data.get("final_confirm", False))
            side = str(data.get("side", "NONE")).upper()
            entry = float(data.get("entry"))
            sl = float(data.get("sl"))
            tp = float(data.get("tp"))
            conf = float(data.get("confidence"))
            mentor_hint = str(data.get("mentor_hint", "") or "")
        except Exception:
            return None

        if side not in ("BUY", "SELL"):
            return None

        conf = max(0.0, min(100.0, conf))
        return AIConfirmDecision(bool(fc), side, float(entry), float(sl), float(tp), float(conf), mentor_hint)

    def _parse_text_response(self, text: str) -> Optional[AIConfirmDecision]:
        if not text:
            return None

        side = "NONE"
        m_side = re.search(r"\b(BUY|SELL)\b", text.upper())
        if m_side:
            side = m_side.group(1)

        fc = False
        m_fc = re.search(r"FINAL[_ ]?CONFIRM\s*=\s*(TRUE|FALSE)", text.upper())
        if m_fc:
            fc = (m_fc.group(1) == "TRUE")
        else:
            fc = (side in ("BUY", "SELL"))

        def _find_float(label: str) -> Optional[float]:
            m = re.search(label + r"\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            return float(m.group(1)) if m else None

        entry = _find_float("Entry")
        sl = _find_float("SL")
        tp = _find_float("TP")
        conf = _find_float("Confidence")

        if entry is None or sl is None or tp is None or conf is None:
            return None
        if side not in ("BUY", "SELL"):
            return None

        conf = max(0.0, min(100.0, float(conf)))
        return AIConfirmDecision(bool(fc), side, float(entry), float(sl), float(tp), float(conf), "")


def _sample_signal() -> Dict[str, Any]:
    return {
        "symbol": "GOLD",
        "baseline": {"direction": "BUY", "entry": 5190.2, "sl": 5183.0, "tp": 5202.0, "lot": 0.10, "rr": 1.6, "atr": 4.55, "mode": "sideway_scalp"},
        "context": {
            "watch_state": "WATCH_BUY_BREAKOUT",
            "breakout_state": "BREAKOUT_BUY_CONFIRMED",
            "price_bid": 5190.10,
            "price_ask": 5190.20,
            "atr": 4.55,
            "bos_ref_high": 5189.67,
            "bos_ref_low": 5160.12,
            "distance_buy": -0.53,
            "distance_sell": 30.0,
            "proximity_score": 76.7,
            "htf_trend": "ranging",
            "mtf_trend": "ranging",
            "ltf_trend": "bullish",
            "spread_points": 12,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    client = AIConfirmClient(args.config)
    signal = _sample_signal() if args.sample else {}
    decision = client.request_confirm(signal)

    print("AIConfirmDecision:", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())