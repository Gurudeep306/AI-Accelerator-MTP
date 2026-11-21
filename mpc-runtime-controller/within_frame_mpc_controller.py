#!/usr/bin/env python3
"""Within-frame, first-layer-informed adaptive controller (Controller C).

This is a scheduling/control adapter, not a hardware driver. It collects measured
features after Layer 0/1, estimates remaining workload, and asks the shared MPC core
to choose a configuration for Layer 1+ onward. The caller MUST apply the returned
action only at a hardware-supported safe layer boundary.

Updates for Phase 2:
1. Replaced MobileNetV2 hardcoded constants with exact YOLOv8n graph information
   (Total Conv2d MACs: 2,623,196,160; Layer 0: 26,542,080, fraction: 1.0118%).
2. Actively uses measured_latency_ms to compute observed throughput and project remaining time.
3. Implements Strategy 3 (Asymmetric Horizon): Step 0 optimizes remaining frame workload
   against remaining deadline budget (accounting for elapsed Layer 0 time), while future
   steps use full-frame forecasts without corrupting multi-step projections.
4. Strictly gates KEEP_CURRENT with feasibility validation: never keeps an infeasible action
   (e.g., quality floor violations) just to avoid switching overhead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

from practical_mpc_controller import Action, Constraints, Decision, MPCController, Telemetry
from workload_prediction_interface import (
    NOMINAL_R3_GMAC,
    PredictionScope,
    WorkloadPrediction,
)

# YOLOv8n Default Static Conv2d MAC Budgets
YOLO_R3_CONV2D_MACS: float = 2_623_196_160.0
YOLO_R3_LAYER0_MACS: float = 26_542_080.0
YOLO_R3_LAYER0_FRACTION: float = 26_542_080.0 / 2_623_196_160.0  # 0.01011822


@dataclass(frozen=True)
class FirstLayerObservation:
    frame_id: int
    layer_id: int
    mac_count: Optional[float]
    measured_latency_ms: float
    memory_stall_fraction: Optional[float] = None
    busy_fraction: Optional[float] = None
    queue_depth: Optional[float] = None
    activation_sparsity: Optional[float] = None
    total_model_macs: Optional[float] = None


@dataclass(frozen=True)
class RemainingWorkloadEstimate:
    workload_scale: float
    uncertainty: float
    features_used: Tuple[str, ...]
    remaining_gmac: float = NOMINAL_R3_GMAC * (1.0 - YOLO_R3_LAYER0_FRACTION)
    total_gmac: float = NOMINAL_R3_GMAC
    layer0_mac_fraction: float = YOLO_R3_LAYER0_FRACTION
    measured_l0_latency_ms: float = 0.0
    projected_remaining_latency_ms: Optional[float] = None
    observed_throughput_gmac_per_s: Optional[float] = None


class RemainingWorkloadEstimator:
    """Interpretable estimator adapter grounded on YOLOv8n execution graph.

    Estimates remaining work from exact Layer 0 MAC counts and execution timing.
    Defaults to verified YOLOv8n graph values if explicit numbers are omitted.
    """
    def __init__(
        self,
        total_model_macs: Optional[float] = None,
        calibrated_callback=None,
        default_layer0_macs: Optional[float] = None,
    ):
        if total_model_macs is not None and total_model_macs <= 0:
            raise ValueError("total_model_macs must be positive")
        self.total_model_macs = total_model_macs if total_model_macs is not None else YOLO_R3_CONV2D_MACS
        self.default_layer0_macs = (
            default_layer0_macs if default_layer0_macs is not None else YOLO_R3_LAYER0_MACS
        )
        self.calibrated_callback = calibrated_callback

    def estimate(self, obs: FirstLayerObservation) -> RemainingWorkloadEstimate:
        features = []
        if self.calibrated_callback is not None:
            result = self.calibrated_callback(obs)
            scale = float(result["workload_scale"])
            uncertainty = float(result["uncertainty"])
            if scale <= 0 or not 0 <= uncertainty <= 1:
                raise ValueError("calibrated callback returned invalid estimate")
            return RemainingWorkloadEstimate(
                workload_scale=scale,
                uncertainty=uncertainty,
                features_used=("calibrated_callback",),
            )

        total_macs = obs.total_model_macs if obs.total_model_macs is not None else self.total_model_macs
        l0_macs = obs.mac_count if obs.mac_count is not None else self.default_layer0_macs

        if l0_macs <= 0 or l0_macs > total_macs:
            raise ValueError(f"first-layer MAC count ({l0_macs}) must be within (0, total_model_macs ({total_macs})]")

        remaining_macs = max(0.0, total_macs - l0_macs)
        scale = remaining_macs / total_macs
        scale = max(scale, 1e-3)
        features.append("first_layer_mac_fraction")

        l0_frac = l0_macs / total_macs
        remaining_gmac = remaining_macs / 1e9
        total_gmac = total_macs / 1e9

        # Actively incorporate measured Layer 0 latency to estimate runtime throughput
        proj_lat = None
        obs_thru = None
        lat_ms = float(obs.measured_latency_ms)
        if lat_ms > 0:
            obs_thru = (l0_macs / 1e9) / (lat_ms / 1000.0)  # GMAC / s
            proj_lat = (remaining_gmac / obs_thru) * 1000.0  # ms
            features.append("measured_layer0_latency_ms")
            features.append("observed_throughput_gmac_s")

        return RemainingWorkloadEstimate(
            workload_scale=scale,
            uncertainty=0.10,
            features_used=tuple(features),
            remaining_gmac=remaining_gmac,
            total_gmac=total_gmac,
            layer0_mac_fraction=l0_frac,
            measured_l0_latency_ms=lat_ms,
            projected_remaining_latency_ms=proj_lat,
            observed_throughput_gmac_per_s=obs_thru,
        )


@dataclass(frozen=True)
class LayerBoundaryDecision:
    frame_id: int
    apply_from_layer: int
    decision: Decision
    estimate: RemainingWorkloadEstimate
    note: str


class WithinFrameAdaptiveMPC:
    """Within-frame adaptation: observe Layer 0, decide for remaining layers."""
    def __init__(
        self,
        mpc: MPCController,
        estimator: RemainingWorkloadEstimator,
        minimum_predicted_saving_mj: float = 0.0,
        measured_switch_cost_mj: Optional[float] = None,
    ):
        self.mpc = mpc
        self.estimator = estimator
        self.minimum_predicted_saving_mj = max(0.0, minimum_predicted_saving_mj)
        self.measured_switch_cost_mj = (
            measured_switch_cost_mj if measured_switch_cost_mj is not None else mpc.c.switch_penalty_mj
        )

    def decide_after_first_layer(
        self,
        obs: FirstLayerObservation,
        telemetry: Telemetry,
        current_action: Union[Action, str],
        current_remaining_energy_mj: Optional[float] = None,
        full_frame_forecast: Optional[Sequence[object]] = None,
    ) -> LayerBoundaryDecision:
        # Validate and normalize current_action to an Action instance
        if isinstance(current_action, str):
            matches = [a for a in self.mpc.validated_actions if a.name == current_action]
            if not matches:
                raise ValueError(f"Unknown action name '{current_action}'; not in validated actions")
            norm_current_action = matches[0]
        elif isinstance(current_action, Action):
            if current_action not in self.mpc.validated_actions:
                raise ValueError(f"Action '{current_action.name}' is not in validated action set")
            norm_current_action = current_action
        else:
            raise TypeError(f"current_action must be an Action or valid action name str, got {type(current_action).__name__}")

        estimate = self.estimator.estimate(obs)

        # Standardized Workload Prediction for Step 0 (Remaining frame layers)
        step0_workload = WorkloadPrediction(
            expected_gmac=estimate.remaining_gmac,
            gmac_std=estimate.remaining_gmac * estimate.uncertainty,
            scope=PredictionScope.REMAINING_FRAME,
            overhead_ms=obs.measured_latency_ms,
            confidence=1.0 - estimate.uncertainty,
            layer_boundary=1,
            label_provenance="ANALYTICAL_GRAPH_REMAINING",
        )

        # STRATEGY 3: ASYMMETRIC HORIZON
        # Step 0 evaluates the current frame's remaining MAC workload against remaining deadline.
        # Future steps 1..H-1 evaluate FULL-FRAME workloads. Do NOT repeat the fractional scale!
        h_future = max(0, self.mpc.c.horizon - 1)
        if full_frame_forecast is not None and len(full_frame_forecast) >= h_future:
            forecast = list(full_frame_forecast[:h_future])
        else:
            # Default to full-frame nominal scale (1.0 = 2.6232 GMAC)
            forecast = [1.0] * h_future

        # Configure telemetry so the MPC core knows Layer 0 has executed
        telemetry.layer1_latency_ms = obs.measured_latency_ms
        telemetry.layer1_mac_fraction = estimate.layer0_mac_fraction

        # Solve MPC optimization over the asymmetric horizon
        decision = self.mpc.choose(step0_workload, telemetry, workload_forecast=forecast)

        note = "candidate selected; hardware must confirm safe layer-boundary reconfiguration"
        if decision.action != norm_current_action:
            if self.measured_switch_cost_mj is None:
                note += "; switch cost is uncalibrated, so benefit is not yet established"
            elif current_remaining_energy_mj is not None:
                estimated_saving = current_remaining_energy_mj - (
                    decision.predicted_energy_mj
                    if decision.predicted_energy_mj is not None else current_remaining_energy_mj
                )
                if estimated_saving < self.measured_switch_cost_mj + self.minimum_predicted_saving_mj:
                    # CRITICAL FIX: Verify feasibility of norm_current_action BEFORE overriding!
                    # Never keep an infeasible configuration just to save switching cost!
                    c = self.mpc.c
                    is_feasible = True
                    infeasibility_reasons = []
                    if hasattr(self.mpc, "predictor") and hasattr(self.mpc, "_robust"):
                        try:
                            pred_curr = self.mpc.predictor.predict(norm_current_action, step0_workload, telemetry)
                            lat_curr, pwr_curr, _, qual_curr = self.mpc._robust(pred_curr, c)
                            if qual_curr is None or qual_curr < c.quality_floor:
                                is_feasible = False
                                infeasibility_reasons.append(
                                    f"quality {qual_curr if qual_curr is not None else 'None'} < floor {c.quality_floor:.3f}"
                                )
                            if pwr_curr > c.power_cap_w:
                                is_feasible = False
                                infeasibility_reasons.append(f"power {pwr_curr:.2f}W > cap {c.power_cap_w:.2f}W")
                            e2e_curr = telemetry.accel_queue_wait_ms + obs.measured_latency_ms + lat_curr
                            if e2e_curr > c.deadline_ms:
                                is_feasible = False
                                infeasibility_reasons.append(f"e2e latency {e2e_curr:.1f}ms > deadline {c.deadline_ms:.1f}ms")
                        except Exception as exc:
                            is_feasible = False
                            infeasibility_reasons.append(f"prediction failed: {exc}")

                    if is_feasible:
                        # Only keep current action when it is strictly FEASIBLE
                        note = "kept current action because predicted benefit does not cover switch cost"
                        decision = Decision(
                            action=norm_current_action,
                            mode="KEEP_CURRENT",
                            reason="switch overhead exceeds estimated benefit; current action is fully feasible",
                            predicted_latency_ms=decision.predicted_latency_ms,
                            predicted_power_w=decision.predicted_power_w,
                            predicted_energy_mj=decision.predicted_energy_mj,
                            predicted_temperature_c=decision.predicted_temperature_c,
                            feasible_actions=decision.feasible_actions,
                            planned_sequence=decision.planned_sequence,
                            predicted_e2e_latency_ms=getattr(decision, "predicted_e2e_latency_ms", None),
                        )
                    else:
                        # Current action violates constraints: MUST execute MPC switch to feasible action!
                        note = (
                            f"switch enforced: current action '{norm_current_action.name}' is infeasible "
                            f"({'; '.join(infeasibility_reasons)}), despite switch cost"
                        )

        return LayerBoundaryDecision(
            frame_id=obs.frame_id,
            apply_from_layer=1,
            decision=decision,
            estimate=estimate,
            note=note,
        )


def main():
    print("Controller C: YOLOv8n layer-0 observation -> remaining-work estimate -> Asymmetric Horizon MPC.")


if __name__ == "__main__":
    main()
