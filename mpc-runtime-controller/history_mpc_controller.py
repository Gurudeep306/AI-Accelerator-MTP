#!/usr/bin/env python3
"""History-based workload forecasting adapter for the shared practical MPC core.

This is Controller A. It does NOT fabricate accelerator performance values.
A calibrated performance predictor and validated action set must be supplied to
practical_mpc_controller.MPCController.

Updates for Phase 2:
1. Standardized workload units: supports explicit GMAC observations and targets.
2. Preserves the exact bounded EWMA + clipped trend baseline for fair empirical evaluation.
3. Generates standardized WorkloadPrediction with discrete resolution/GMAC tier mapping.
4. Compliant cold-start fallback handling under constrained quality floors.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Union

from practical_mpc_controller import Action, Constraints, Decision, MPCController, Telemetry
from workload_prediction_interface import (
    NOMINAL_R3_GMAC,
    PredictionScope,
    TIER_GMAC_MAP,
    WorkloadPrediction,
)

# Decision boundaries between resolution tiers (midpoints in GMAC)
TIER_0_1_BOUNDARY_GMAC: float = (TIER_GMAC_MAP[0]["gmac"] + TIER_GMAC_MAP[1]["gmac"]) / 2.0  # 1.065673
TIER_1_2_BOUNDARY_GMAC: float = (TIER_GMAC_MAP[1]["gmac"] + TIER_GMAC_MAP[2]["gmac"]) / 2.0  # 2.049372


@dataclass(frozen=True)
class WorkloadObservation:
    frame_id: int
    workload_scale: float = 1.0
    mac_count: Optional[float] = None
    gmac: Optional[float] = None
    queue_depth: Optional[float] = None
    recent_latency_ms: Optional[float] = None
    tier_idx: Optional[int] = None

    def __post_init__(self):
        # Reconcile gmac and workload_scale
        if self.gmac is not None and self.gmac > 0:
            object.__setattr__(self, "workload_scale", self.gmac / NOMINAL_R3_GMAC)
        elif self.mac_count is not None and self.mac_count > 0:
            object.__setattr__(self, "gmac", self.mac_count / 1e9)
            object.__setattr__(self, "workload_scale", (self.mac_count / 1e9) / NOMINAL_R3_GMAC)
        else:
            object.__setattr__(self, "gmac", self.workload_scale * NOMINAL_R3_GMAC)


@dataclass(frozen=True)
class WorkloadForecast:
    workload_scale: float
    uncertainty: float
    method: str
    expected_gmac: float = NOMINAL_R3_GMAC
    tier_idx: int = 2
    tier_name: str = "R3_640x384"


class HistoryWorkloadPredictor:
    """Bounded EWMA + trend forecaster; intentionally simple, auditable, and reproducible.

    Estimates the next workload using only past observations. This is the temporal
    baseline. Preserves exact mathematical formulation as approved by audit.
    """
    def __init__(
        self,
        history_size: int = 12,
        alpha: float = 0.45,
        max_relative_step: float = 0.50,
        min_scale: float = 0.10,
        max_scale: float = 4.0,
    ):
        if history_size < 2:
            raise ValueError("history_size must be >= 2")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if max_relative_step <= 0 or min_scale <= 0 or max_scale <= min_scale:
            raise ValueError("invalid scale bounds")
        self.history: Deque[WorkloadObservation] = deque(maxlen=history_size)
        self.alpha = alpha
        self.max_relative_step = max_relative_step
        self.min_scale = min_scale
        self.max_scale = max_scale

    def update(self, obs: WorkloadObservation) -> None:
        if obs.workload_scale <= 0:
            raise ValueError("workload_scale must be positive")
        self.history.append(obs)

    def predict_next(self) -> WorkloadForecast:
        if not self.history:
            raise RuntimeError("No workload history available; use a conservative fallback.")
        values = [x.workload_scale for x in self.history]
        ewma = values[0]
        for x in values[1:]:
            ewma = self.alpha * x + (1.0 - self.alpha) * ewma

        if len(values) >= 2:
            # Average recent difference, clipped to avoid explosive extrapolation
            recent = values[-min(4, len(values)):]
            diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
            trend = sum(diffs) / len(diffs) if diffs else 0.0
            cap = max(ewma * self.max_relative_step, 1e-6)
            forecast = ewma + max(-cap, min(cap, trend))
        else:
            forecast = ewma

        forecast = max(self.min_scale, min(self.max_scale, forecast))
        mean = sum(values) / len(values)
        mad = sum(abs(x - mean) for x in values) / len(values)
        uncertainty = min(1.0, mad / max(mean, 1e-6))

        expected_gmac = forecast * NOMINAL_R3_GMAC
        if expected_gmac < TIER_0_1_BOUNDARY_GMAC:
            tier_idx = 0
            tier_name = "R1_320x192"
        elif expected_gmac < TIER_1_2_BOUNDARY_GMAC:
            tier_idx = 1
            tier_name = "R2_480x288"
        else:
            tier_idx = 2
            tier_name = "R3_640x384"

        return WorkloadForecast(
            workload_scale=forecast,
            uncertainty=uncertainty,
            method="ewma_plus_clipped_trend",
            expected_gmac=expected_gmac,
            tier_idx=tier_idx,
            tier_name=tier_name,
        )

    def predict_next_prediction(self) -> WorkloadPrediction:
        """Produce standardized WorkloadPrediction object."""
        fc = self.predict_next()
        return WorkloadPrediction(
            expected_gmac=fc.expected_gmac,
            gmac_std=fc.uncertainty * fc.expected_gmac,
            scope=PredictionScope.FULL_FRAME,
            overhead_ms=0.002,  # negligible EWMA arithmetic overhead
            confidence=max(0.1, 1.0 - fc.uncertainty),
            layer_boundary=0,
            tier_idx=fc.tier_idx,
            tier_name=fc.tier_name,
            label_provenance="TEMPORAL_EWMA_FORECAST",
        )


class HistoryBasedMPC:
    """Frame-level receding-horizon MPC fed by historical workload observations."""
    def __init__(
        self,
        mpc: MPCController,
        forecaster: Optional[HistoryWorkloadPredictor] = None,
        cold_start_nominal_scale: Optional[float] = None,
    ):
        self.mpc = mpc
        self.forecaster = forecaster or HistoryWorkloadPredictor()
        self.cold_start_nominal_scale = cold_start_nominal_scale

    def observe_completed_frame(self, obs: WorkloadObservation) -> None:
        self.forecaster.update(obs)

    def decide_before_next_frame(
        self,
        telemetry: Telemetry,
        constraints_override: Optional[Constraints] = None,
    ) -> Decision:
        if constraints_override is not None:
            self.mpc.validate_constraints(constraints_override)
        c = constraints_override if constraints_override is not None else self.mpc.c

        if len(self.forecaster.history) == 0:
            if self.cold_start_nominal_scale is not None:
                nominal = self.cold_start_nominal_scale
                future = [nominal] * max(0, c.horizon - 1)
                return self.mpc.choose(
                    nominal, telemetry, workload_forecast=future,
                    constraints_override=constraints_override
                )
            
            # Select compliant fallback on cold start
            fallback_act = self.mpc.fallback
            try:
                pred_fb = self.mpc.predictor.predict(fallback_act, 1.0, telemetry)
                _, _, _, qual_fb = self.mpc._robust(pred_fb, c)
                if qual_fb is None or qual_fb < c.quality_floor:
                    # Fallback violates quality floor, find compliant action
                    for act in self.mpc.validated_actions:
                        p = self.mpc.predictor.predict(act, 1.0, telemetry)
                        _, _, _, q = self.mpc._robust(p, c)
                        if q is not None and q >= c.quality_floor:
                            fallback_act = act
                            break
            except Exception:
                pass

            self.mpc.previous_action = fallback_act
            return Decision(
                action=fallback_act,
                mode="COLD_START_FALLBACK",
                reason=(
                    "No workload history available before frame 0 completes; "
                    f"applied compliant fallback action '{fallback_act.name}'."
                ),
            )

        pred_obj = self.forecaster.predict_next_prediction()
        # Flat persistence across horizon H-1
        future = [pred_obj] * max(0, c.horizon - 1)

        return self.mpc.choose(
            pred_obj,
            telemetry,
            workload_forecast=future,
            constraints_override=constraints_override,
        )


def main():
    print("Controller A: Bounded EWMA + Trend -> Workload Prediction -> MPC.")


if __name__ == "__main__":
    main()
