"""
AI Mentor - Spec v2 (Mode D: Bounded Adjustment) - Version: 2.1.0
Changelog:
- 2.1.0 (2026-02-27):
  - NEW: Spec v2 response blocks: {execution, analysis, mentor}
  - NEW: Bounded adjustment rules (direction locked, no lot sizing, constraints enforced)
  - NEW: Confidence breakdown + risk_flags + invalidations + mentor narrative (human-readable)
  - SAFETY: Fail-closed schema, always return JSON dict
Backtest evidence (placeholder):
- N/A (Spec change only; requires forward/live A/B logging to quantify)

Design intent:
- Think big (full context reasoning)
- Act small (bounded authority)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        if v is None:
            return float(d)
        return float(v)
    except Exception:
        return float(d)


def _si(v: Any, d: int = 0) -> int:
    try:
        if v is None:
            return int(d)
        return int(v)
    except Exception:
        return int(d)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


@dataclass(frozen=True)
class AIMentorConfig:
    # Defaults (override via input["constraints"] if present)
    min_rr: float = 1.50
    entry_shift_max_atr: float = 0.20
    sl_atr_min: float = 1.20
    sl_atr_max: float = 1.80
    conf_execute_threshold: int = 70  # Python will gate; AI still outputs conf
    event_conf_cap_high: int = 72

    # Confidence scoring weights (sum 100)
    w_technical: int = 25
    w_structure: int = 20
    w_context: int = 20
    w_execution: int = 15
    w_portfolio: int = 20


class AIMentor:
    """
    MOCK AI (rule-based) แต่ทำหน้าที่เหมือน LLM:
    - รับ full package
    - สังเคราะห์หลายมิติ
    - คืน JSON blocks: execution/analysis/mentor
    """

    def __init__(self, cfg: Optional[AIMentorConfig] = None):
        self.cfg = cfg or AIMentorConfig()

    # -------------------------
    # Public API
    # -------------------------
    def evaluate(self, package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Required output:
        {
          "execution": {"entry","sl","tp","rr","conf","decision"},
          "analysis": {...},
          "mentor": {...}
        }
        """
        # Fail-closed defaults
        safe_out = self._fail_closed_output(package, reason="init")

        try:
            # Extract baseline
            baseline = (package.get("baseline") or {})
            direction = str(baseline.get("dir", "NONE")).upper()
            entry0 = _sf(baseline.get("entry"))
            sl0 = _sf(baseline.get("sl"))
            tp0 = _sf(baseline.get("tp"))
            rr0 = _sf(baseline.get("rr"))

            # Constraints (override defaults)
            c = self._merge_constraints(package.get("constraints") or {})
            atr = max(_sf(baseline.get("atr"), 0.0), 1e-9)

            # Basic sanity
            if direction not in ("BUY", "SELL"):
                return self._fail_closed_output(package, reason="invalid_direction")
            if not (self._sanity(direction, entry0, sl0, tp0)):
                return self._fail_closed_output(package, reason="baseline_sanity_failed")
            if rr0 < c.min_rr:
                # baseline rr ต่ำกว่า frozen -> reject
                return self._fail_closed_output(package, reason="baseline_rr_below_min")

            # Score each dimension
            tech_score, tech_flags = self._score_technical(package)
            struct_score, struct_flags = self._score_structure(package)
            ctx_score, ctx_flags = self._score_context(package)
            exec_score, exec_flags = self._score_execution(package)
            port_score, port_flags = self._score_portfolio(package)

            risk_flags = tech_flags + struct_flags + ctx_flags + exec_flags + port_flags
            risk_flags = self._dedupe_flags(risk_flags)

            # Build confidence breakdown (0..weights)
            breakdown = {
                "technical": _clamp(tech_score, 0, self.cfg.w_technical),
                "structure": _clamp(struct_score, 0, self.cfg.w_structure),
                "context": _clamp(ctx_score, 0, self.cfg.w_context),
                "execution": _clamp(exec_score, 0, self.cfg.w_execution),
                "portfolio": _clamp(port_score, 0, self.cfg.w_portfolio),
            }
            conf_raw = int(sum(breakdown.values()))

            # Apply event cap (FOMC_HIGH etc.)
            conf = self._apply_event_cap(package, conf_raw, c)

            # Decision rule (AI-side; Python still gates by conf threshold)
            decision = "APPROVE" if conf >= self.cfg.conf_execute_threshold else "REJECT"

            # Bounded adjustment
            entry1, sl1, tp1, rr1, adjust_notes = self._bounded_adjust(
                direction=direction,
                entry=entry0,
                sl=sl0,
                tp=tp0,
                rr=rr0,
                atr=atr,
                constraints=c,
                package=package,
                risk_flags=risk_flags,
            )

            # Final constraint check (AI self-check; Python will re-check)
            checks = self._constraint_check(
                direction=direction,
                entry0=entry0,
                sl0=sl0,
                tp0=tp0,
                entry1=entry1,
                sl1=sl1,
                tp1=tp1,
                rr1=rr1,
                atr=atr,
                constraints=c,
            )

            # If AI-adjusted fails constraints -> revert baseline + downgrade conf
            if not checks["all_ok"]:
                entry1, sl1, tp1, rr1 = entry0, sl0, tp0, rr0
                decision = "REJECT"
                conf = min(conf, 45)
                risk_flags = self._dedupe_flags(risk_flags + ["CONSTRAINT_FAIL_REVERT"])

            invalidations = self._build_invalidations(direction, entry1, sl1, tp1, package)

            mentor = self._build_mentor_message(
                direction=direction,
                entry=entry1,
                sl=sl1,
                tp=tp1,
                conf=conf,
                risk_flags=risk_flags,
                adjust_notes=adjust_notes,
                package=package,
            )

            out = {
                "execution": {
                    "entry": float(entry1),
                    "sl": float(sl1),
                    "tp": float(tp1),
                    "rr": float(rr1),
                    "conf": int(conf),
                    "decision": str(decision),
                },
                "analysis": {
                    "confidence_breakdown": breakdown,
                    "risk_flags": risk_flags,
                    "invalidations": invalidations,
                    "adjust_notes": adjust_notes,
                    "constraints_check": checks,
                },
                "mentor": mentor,
            }

            return out

        except Exception:
            # Fail closed
            return safe_out

    # -------------------------
    # Internals
    # -------------------------
    def _merge_constraints(self, c: Dict[str, Any]) -> AIMentorConfig:
        # Only allow overriding selected numeric bounds; keep direction locked by design
        return AIMentorConfig(
            min_rr=_sf(c.get("min_rr", self.cfg.min_rr), self.cfg.min_rr),
            entry_shift_max_atr=_sf(c.get("entry_shift_max_atr", self.cfg.entry_shift_max_atr), self.cfg.entry_shift_max_atr),
            sl_atr_min=_sf(c.get("sl_atr_min", self.cfg.sl_atr_min), self.cfg.sl_atr_min),
            sl_atr_max=_sf(c.get("sl_atr_max", self.cfg.sl_atr_max), self.cfg.sl_atr_max),
            conf_execute_threshold=_si(c.get("conf_execute_threshold", self.cfg.conf_execute_threshold), self.cfg.conf_execute_threshold),
            event_conf_cap_high=_si(c.get("event_conf_cap_high", self.cfg.event_conf_cap_high), self.cfg.event_conf_cap_high),
            w_technical=self.cfg.w_technical,
            w_structure=self.cfg.w_structure,
            w_context=self.cfg.w_context,
            w_execution=self.cfg.w_execution,
            w_portfolio=self.cfg.w_portfolio,
        )

    def _fail_closed_output(self, package: Dict[str, Any], reason: str) -> Dict[str, Any]:
        # Use baseline if available, but conf=0 to reject
        baseline = (package.get("baseline") or {})
        entry = _sf(baseline.get("entry"))
        sl = _sf(baseline.get("sl"))
        tp = _sf(baseline.get("tp"))
        rr = _sf(baseline.get("rr"))
        return {
            "execution": {
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "rr": float(rr),
                "conf": 0,
                "decision": "REJECT",
            },
            "analysis": {
                "confidence_breakdown": {
                    "technical": 0, "structure": 0, "context": 0, "execution": 0, "portfolio": 0
                },
                "risk_flags": ["FAIL_CLOSED", f"REASON_{reason}"],
                "invalidations": [],
                "adjust_notes": [f"fail_closed: {reason}"],
                "constraints_check": {"all_ok": False},
            },
            "mentor": {
                "headline": "AI fail-closed (no trade)",
                "explanation": f"System safety triggered: {reason}",
                "action_guidance": "Skip this signal.",
                "confidence_reasoning": "Confidence=0 due to fail-closed safety.",
            },
        }

    def _sanity(self, direction: str, entry: float, sl: float, tp: float) -> bool:
        if not (_is_finite(entry) and _is_finite(sl) and _is_finite(tp)):
            return False
        if direction == "BUY":
            return sl < entry < tp
        if direction == "SELL":
            return tp < entry < sl
        return False

    def _dedupe_flags(self, flags: List[str]) -> List[str]:
        out = []
        seen = set()
        for f in flags:
            f = str(f).strip()
            if not f:
                continue
            if f not in seen:
                out.append(f)
                seen.add(f)
        return out[:30]

    # ---- scoring ----
    def _score_technical(self, package: Dict[str, Any]) -> Tuple[int, List[str]]:
        t = (package.get("technical") or {})
        flags = []
        score = self.cfg.w_technical

        rsi = _sf(t.get("rsi"), 50.0)
        bb = str(t.get("bb_state", "")).lower()
        adx = _sf(t.get("adx"), 20.0)

        # Heuristics
        if bb in ("lower_touch", "upper_touch"):
            score -= 0  # supportive of mean reversion
        else:
            score -= 4
            flags.append("BB_NOT_ALIGNED")

        if rsi < 30 or rsi > 70:
            score -= 0
        else:
            score -= 3
            flags.append("RSI_NOT_EXTREME")

        if adx < 17:
            score -= 4
            flags.append("LOW_ADX_RANGE")
        elif adx > 30:
            score -= 0
        else:
            score -= 2

        return int(_clamp(score, 0, self.cfg.w_technical)), flags

    def _score_structure(self, package: Dict[str, Any]) -> Tuple[int, List[str]]:
        s = (package.get("structure") or {})
        flags = []
        score = self.cfg.w_structure

        bos = bool(s.get("bos", False))
        choch = bool(s.get("choch", False))
        regime = str((package.get("context") or {}).get("regime", "")).lower()

        if not bos:
            score -= 8
            flags.append("NO_BOS")
        if choch:
            score -= 3
            flags.append("CHOCH_PRESENT")
        if regime == "sideways":
            score -= 2

        return int(_clamp(score, 0, self.cfg.w_structure)), flags

    def _score_context(self, package: Dict[str, Any]) -> Tuple[int, List[str]]:
        c = (package.get("context") or {})
        flags = []
        score = self.cfg.w_context

        event_risk = str(c.get("event_risk", "")).upper()
        session = str(c.get("session", "")).lower()
        corr = (c.get("correlations") or {})

        if event_risk.endswith("HIGH"):
            score -= 8
            flags.append("EVENT_RISK_HIGH")
        elif event_risk:
            score -= 4
            flags.append("EVENT_RISK_MED")

        if session in ("london", "newyork", "ny", "ny_overlap"):
            score -= 0
        else:
            score -= 2

        if isinstance(corr, dict) and len(corr) > 0:
            # If any correlation magnitude high -> risk
            mx = max(abs(_sf(v, 0.0)) for v in corr.values())
            if mx >= 0.6:
                score -= 5
                flags.append("CORRELATION_RISK_HIGH")
            elif mx >= 0.3:
                score -= 3
                flags.append("CORRELATION_RISK_MED")

        return int(_clamp(score, 0, self.cfg.w_context)), flags

    def _score_execution(self, package: Dict[str, Any]) -> Tuple[int, List[str]]:
        e = (package.get("execution_context") or {})
        flags = []
        score = self.cfg.w_execution

        spread_z = _sf(e.get("spread_z"), 0.0)
        slip = _sf(e.get("slippage_estimate"), 0.0)

        if spread_z >= 2.5:
            score -= 8
            flags.append("SPREAD_SPIKE")
        elif spread_z >= 1.5:
            score -= 4
            flags.append("SPREAD_ELEVATED")

        if slip >= 1.0:
            score -= 4
            flags.append("SLIPPAGE_RISK")
        elif slip >= 0.5:
            score -= 2

        return int(_clamp(score, 0, self.cfg.w_execution)), flags

    def _score_portfolio(self, package: Dict[str, Any]) -> Tuple[int, List[str]]:
        p = (package.get("portfolio") or {})
        flags = []
        score = self.cfg.w_portfolio

        dd = _sf(p.get("equity_drawdown_pct"), 0.0)
        corr_risk = _sf(p.get("correlation_risk"), 0.0)
        open_pos = _si(p.get("open_positions_symbol"), 0)

        if dd >= 3.0:
            score -= 10
            flags.append("DD_ELEVATED")
        elif dd >= 1.5:
            score -= 5

        if corr_risk >= 0.6:
            score -= 6
            flags.append("PORT_CORR_RISK_HIGH")
        elif corr_risk >= 0.3:
            score -= 3
            flags.append("PORT_CORR_RISK_MED")

        if open_pos >= 2:
            score -= 4
            flags.append("MULTI_POS_EXPOSURE")

        return int(_clamp(score, 0, self.cfg.w_portfolio)), flags

    def _apply_event_cap(self, package: Dict[str, Any], conf: int, c: AIMentorConfig) -> int:
        ctx = (package.get("context") or {})
        event_risk = str(ctx.get("event_risk", "")).upper()
        if event_risk.endswith("HIGH"):
            return int(min(conf, c.event_conf_cap_high))
        return int(_clamp(conf, 0, 100))

    # ---- bounded adjust ----
    def _bounded_adjust(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        rr: float,
        atr: float,
        constraints: AIMentorConfig,
        package: Dict[str, Any],
        risk_flags: List[str],
    ) -> Tuple[float, float, float, float, List[str]]:
        notes: List[str] = []

        # Default: keep baseline
        entry1, sl1, tp1 = entry, sl, tp

        # Small entry shift rule: if liquidity wall nearby -> shift slightly away
        ctx = (package.get("context") or {})
        liq = str(ctx.get("liquidity", "")).lower()
        max_shift = constraints.entry_shift_max_atr * atr

        if "wall" in liq and max_shift > 0:
            # shift by 10% of max_shift in "safer" direction
            shift = 0.10 * max_shift
            if direction == "BUY":
                entry1 = entry + shift
                tp1 = tp + shift
                sl1 = sl + shift  # preserve geometry shift; still validated later
            else:
                entry1 = entry - shift
                tp1 = tp - shift
                sl1 = sl - shift
            notes.append(f"entry_shift_liq: {shift:.5f}")

        # Tighten SL slightly if event risk high (reduce exposure)
        if "EVENT_RISK_HIGH" in risk_flags:
            tighten = 0.05 * atr  # 5% ATR
            if direction == "BUY":
                sl1 = sl1 + tighten
            else:
                sl1 = sl1 - tighten
            notes.append(f"sl_tighten_event: {tighten:.5f}")

        # Recompute RR after changes
        rr1 = self._calc_rr(direction, entry1, sl1, tp1)
        if rr1 < constraints.min_rr:
            # attempt adjust TP to restore min_rr (within bounded philosophy)
            sl_dist = abs(sl1 - entry1)
            if sl_dist > 0:
                tp_dist = constraints.min_rr * sl_dist
                if direction == "BUY":
                    tp1 = entry1 + tp_dist
                else:
                    tp1 = entry1 - tp_dist
                rr1 = self._calc_rr(direction, entry1, sl1, tp1)
                notes.append("tp_adjust_restore_min_rr")

        return float(entry1), float(sl1), float(tp1), float(rr1), notes

    def _calc_rr(self, direction: str, entry: float, sl: float, tp: float) -> float:
        risk = abs(entry - sl)
        if risk <= 0:
            return 0.0
        reward = abs(tp - entry)
        return float(reward / risk)

    def _constraint_check(
        self,
        direction: str,
        entry0: float,
        sl0: float,
        tp0: float,
        entry1: float,
        sl1: float,
        tp1: float,
        rr1: float,
        atr: float,
        constraints: AIMentorConfig,
    ) -> Dict[str, Any]:
        out = {}

        # entry shift
        max_shift = constraints.entry_shift_max_atr * atr
        out["entry_shift_ok"] = abs(entry1 - entry0) <= (max_shift + 1e-9)

        # SL ATR range
        sl_dist = abs(sl1 - entry1)
        out["sl_atr_min_ok"] = sl_dist >= (constraints.sl_atr_min * atr - 1e-9)
        out["sl_atr_max_ok"] = sl_dist <= (constraints.sl_atr_max * atr + 1e-9)

        # RR min
        out["rr_min_ok"] = rr1 >= (constraints.min_rr - 1e-9)

        # sanity
        out["sanity_ok"] = self._sanity(direction, entry1, sl1, tp1)

        out["all_ok"] = bool(
            out["entry_shift_ok"]
            and out["sl_atr_min_ok"]
            and out["sl_atr_max_ok"]
            and out["rr_min_ok"]
            and out["sanity_ok"]
        )
        return out

    def _build_invalidations(self, direction: str, entry: float, sl: float, tp: float, package: Dict[str, Any]) -> List[str]:
        e = (package.get("execution_context") or {})
        inv = []
        inv.append(f"break_{'below' if direction=='BUY' else 'above'}_sl_{sl:.5f}")
        inv.append(f"break_{'above' if direction=='BUY' else 'below'}_tp_{tp:.5f}")
        if _sf(e.get("spread_z"), 0.0) >= 1.5:
            inv.append("spread_z_rising")
        return inv[:10]

    def _build_mentor_message(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        conf: int,
        risk_flags: List[str],
        adjust_notes: List[str],
        package: Dict[str, Any],
    ) -> Dict[str, str]:
        ctx = (package.get("context") or {})
        t = (package.get("technical") or {})
        s = (package.get("structure") or {})

        event_risk = str(ctx.get("event_risk", "")).upper() or "NONE"
        regime = str(ctx.get("regime", "")).lower() or "unknown"
        session = str(ctx.get("session", "")).lower() or "unknown"
        rsi = _sf(t.get("rsi"), 0.0)
        bb = str(t.get("bb_state", "")).lower()
        bos = bool(s.get("bos", False))

        headline = f"{direction} setup | conf={conf}% | regime={regime} | event={event_risk}"
        explanation = (
            f"BB={bb}, RSI={rsi:.1f}, BOS={bos}. "
            f"Session={session}. RiskFlags={','.join(risk_flags[:6]) or 'none'}."
        )
        action = f"Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f}. " \
                 f"{'Adjustments=' + ','.join(adjust_notes) if adjust_notes else 'No adjustment.'}"
        conf_reason = "Confidence reflects technical+structure alignment, reduced by event/execution/portfolio risks."

        return {
            "headline": headline,
            "explanation": explanation,
            "action_guidance": action,
            "confidence_reasoning": conf_reason,
        }


def _selftest() -> int:
    ai = AIMentor()
    pkg = {
        "baseline": {"dir": "SELL", "entry": 1.10000, "sl": 1.10500, "tp": 1.09500, "rr": 1.67, "atr": 0.0020},
        "technical": {"bb_state": "lower_touch", "rsi": 28.5, "adx": 18.2},
        "structure": {"bos": True, "choch": False},
        "context": {"regime": "sideways", "session": "london", "event_risk": "FOMC_HIGH", "liquidity": "wall_1.0980"},
        "execution_context": {"spread_z": 1.2, "slippage_estimate": 0.3},
        "portfolio": {"open_positions_symbol": 0, "equity_drawdown_pct": 0.8, "correlation_risk": 0.3},
        "constraints": {"min_rr": 1.5, "entry_shift_max_atr": 0.2, "sl_atr_min": 1.2, "sl_atr_max": 1.8, "event_conf_cap_high": 72},
    }
    out = ai.evaluate(pkg)
    print(json.dumps(out, indent=2))
    ok = "execution" in out and "analysis" in out and "mentor" in out
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    raise SystemExit(_selftest() if args.selftest else 0)