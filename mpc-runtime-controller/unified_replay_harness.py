#!/usr/bin/env python3
"""Unified timestamped replay evaluation harness for AI Accelerator controllers.

Fairly compares:
- Controller A: History-Based MPC (receding-horizon, past observations)
- Controller B: Frame-Preview MPC (pre-inference features, workload regression)
- Controller C: Within-Frame Adaptive MPC (first-layer observation, layer-boundary decision)
- Fixed Baseline (Nominal High Quality: W1.00-E3-INT8)
- Fixed Baseline (Low Power / Minimum Latency: W0.50-E1-INT4)
- Rule-Based Baseline (Dynamic reactive heuristic)

All controllers share identical:
- Constraints (deadline, power, temperature, quality)
- Action space (18 operating points)
- Calibrated performance model
- Frame arrival trace and initial conditions
- Overhead accounting (preview latency/energy, decision computation, switching penalty)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from practical_mpc_controller import (
    Action, Constraints, DEFAULT_ACTIONS, MPCController, Prediction, Telemetry, Decision,
    DEFAULT_SWITCH_ENERGY_MJ, DEFAULT_SWITCH_LATENCY_MS, DEFAULT_PREVIEW_ENERGY_MJ, DEFAULT_PREVIEW_LATENCY_MS
)
from history_mpc_controller import HistoryBasedMPC, HistoryWorkloadPredictor, WorkloadObservation
from frame_preview_mpc import (
    FEATURE_NAMES, FramePreviewMPCPipeline, LinearWorkloadModel, ProfilePerformancePredictor
)
from within_frame_mpc_controller import (
    WithinFrameAdaptiveMPC, RemainingWorkloadEstimator, FirstLayerObservation, LayerBoundaryDecision
)


@dataclass(frozen=True)
class FrameArrivalEvent:
    frame_id: int
    arrival_time_ms: float
    deadline_ms: float
    actual_workload_scale: float
    first_layer_mac_fraction: float
    image_features: Dict[str, float]


@dataclass
class NormalizedDecisionRecord:
    frame_id: int
    controller_name: str
    action: Action
    action_name: str
    mode: str
    reason: str
    predicted_latency_ms: Optional[float]
    predicted_power_w: Optional[float]
    predicted_energy_mj: Optional[float]
    predicted_temp_c: Optional[float]
    decision_latency_ms: float
    preview_latency_ms: float = 0.0
    preview_energy_mj: float = 0.0
    reconfig_latency_ms: float = 0.0
    reconfig_energy_mj: float = 0.0
    feasible_action_count: int = 0
    planned_sequence: Tuple[str, ...] = field(default_factory=tuple)
    predicted_e2e_latency_ms: Optional[float] = None
    predicted_total_energy_mj: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameExecutionResult:
    frame_id: int
    controller_name: str
    arrival_time_ms: float
    decision_start_ms: float
    decision_end_ms: float
    dispatch_time_ms: float
    completion_time_ms: float
    end_to_end_latency_ms: float
    deadline_miss: bool
    inference_latency_ms: float
    inference_energy_mj: float
    total_energy_mj: float
    quality: Optional[float]
    temperature_c: float
    queue_depth_at_arrival: int
    decision: NormalizedDecisionRecord
    host_queue_depth_at_arrival: int = 0
    accel_queue_depth_at_arrival: int = 0
    rejected: bool = False
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class FixedBaselineController:
    """Fixed-configuration baseline controller."""
    def __init__(self, action: Action, name: str = "Fixed"):
        self.action = action
        self.name = name

    def decide(self, frame_id: int, telem: Telemetry, deadline_ms: float) -> NormalizedDecisionRecord:
        t0 = time.perf_counter()
        dec_time_ms = (time.perf_counter() - t0) * 1000.0
        return NormalizedDecisionRecord(
            frame_id=frame_id,
            controller_name=self.name,
            action=self.action,
            action_name=self.action.name,
            mode="FIXED",
            reason="Static baseline configuration",
            predicted_latency_ms=None,
            predicted_power_w=None,
            predicted_energy_mj=None,
            predicted_temp_c=None,
            decision_latency_ms=dec_time_ms,
            feasible_action_count=1,
            planned_sequence=(self.action.name,)
        )


class RuleBasedBaselineController:
    """Heuristic rule-based baseline controller:
    Reacts quickly to queue depth and latency pressure by stepping up or down.
    """
    def __init__(self, actions: Sequence[Action], name: str = "Rule-Based"):
        self.actions = sorted(actions)
        self.name = name
        # Default middle action
        self.current_idx = len(self.actions) // 2

    def decide(self, frame_id: int, telem: Telemetry, queue_depth: int,
               deadline_ms: float) -> NormalizedDecisionRecord:
        t0 = time.perf_counter()
        reason = "Nominal operating point maintained"
        prev_lat = telem.observed_latency_ms

        if queue_depth > 1 or (prev_lat is not None and prev_lat > 0.75 * deadline_ms):
            # Throttle down for lower latency
            if self.current_idx > 0:
                self.current_idx = max(0, self.current_idx - 2)
                reason = "Throttled down due to queue congestion or tight latency"
        elif queue_depth == 0 and (prev_lat is not None and prev_lat < 0.35 * deadline_ms):
            # Scale up for higher quality if thermal permits
            if telem.temperature_c < 60.0 and self.current_idx < len(self.actions) - 1:
                self.current_idx = min(len(self.actions) - 1, self.current_idx + 1)
                reason = "Scaled up to improve quality under slack"

        chosen = self.actions[self.current_idx]
        dec_time_ms = (time.perf_counter() - t0) * 1000.0

        return NormalizedDecisionRecord(
            frame_id=frame_id,
            controller_name=self.name,
            action=chosen,
            action_name=chosen.name,
            mode="RULE_BASED",
            reason=reason,
            predicted_latency_ms=None,
            predicted_power_w=None,
            predicted_energy_mj=None,
            predicted_temp_c=None,
            decision_latency_ms=dec_time_ms,
            feasible_action_count=len(self.actions),
            planned_sequence=(chosen.name,)
        )


# ---------------------------------------------------------------------------
# Replay Simulation Environment
# ---------------------------------------------------------------------------

class ReplaySimulator:
    """Simulates realistic timestamped pipeline, thermal dynamics, and hardware overhead."""
    def __init__(self, predictor: ProfilePerformancePredictor, constraints: Constraints,
                 switch_latency_ms: float = DEFAULT_SWITCH_LATENCY_MS,
                 switch_energy_mj: float = DEFAULT_SWITCH_ENERGY_MJ,
                 preview_latency_ms: float = DEFAULT_PREVIEW_LATENCY_MS,
                 preview_energy_mj: float = DEFAULT_PREVIEW_ENERGY_MJ):
        self.predictor = predictor
        self.c = constraints
        self.switch_latency_ms = switch_latency_ms
        self.switch_energy_mj = switch_energy_mj
        self.preview_latency_ms = preview_latency_ms
        self.preview_energy_mj = preview_energy_mj

    def run_trace(self, controller_key: str, controller_obj: Any,
                  trace: List[FrameArrivalEvent]) -> List[FrameExecutionResult]:
        results: List[FrameExecutionResult] = []
        accel_idle_at_ms = 0.0
        current_temp_c = self.c.ambient_c
        current_action: Optional[Action] = None
        last_time_ms = 0.0

        # Pending buffer & causal host decision estimate
        prev_host_dec_end = 0.0
        recent_host_dec_ms: List[float] = [12.0]

        for event in trace:
            arrival_ms = event.arrival_time_ms

            # Thermal evolution during idle gap before this frame
            gap_target_ms = max(arrival_ms, accel_idle_at_ms)
            if gap_target_ms > last_time_ms:
                dt_s = (gap_target_ms - last_time_ms) / 1000.0
                tau = max(self.c.thermal_tau_s, 1e-6)
                alpha = math.exp(-dt_s / tau)
                # Idle power ~ 0.05 W
                current_temp_c = alpha * current_temp_c + (1.0 - alpha) * (self.c.ambient_c + (self.c.thermal_resistance_c_per_w or 2.0) * 0.05)
                last_time_ms = gap_target_ms

            # Host queue depth: frames that arrived earlier whose host decision has not finished
            host_queue_depth = sum(1 for r in results if r.decision_end_ms > arrival_ms)
            # Accelerator queue depth: frames that arrived earlier whose decision completed but execution hasn't finished
            accel_queue_depth = sum(1 for r in results if r.decision_end_ms <= arrival_ms < r.completion_time_ms)
            # Total queue depth: total frames active in system at arrival
            queue_depth = host_queue_depth + accel_queue_depth

            # Causal queue wait & host decision estimates at frame arrival
            host_queue_wait_ms = max(0.0, prev_host_dec_end - arrival_ms)
            accel_queue_wait_ms = max(0.0, accel_idle_at_ms - arrival_ms)
            est_host_dec_ms = float(np.median(recent_host_dec_ms[-5:]))

            # Base telemetry available to controller
            last_res = results[-1] if results else None
            telem = Telemetry(
                temperature_c=current_temp_c,
                observed_latency_ms=last_res.inference_latency_ms if last_res else None,
                observed_power_w=None,
                observed_energy_mj=last_res.total_energy_mj if last_res else None,
                timestamp_s=arrival_ms / 1000.0,
                host_queue_wait_ms=host_queue_wait_ms,
                accel_queue_wait_ms=accel_queue_wait_ms,
                estimated_host_decision_ms=est_host_dec_ms,
                preview_latency_ms=0.0,
                switch_latency_ms=self.switch_latency_ms
            )

            if controller_key == "controller_c":
                # =====================================================================
                # CONTROLLER C: WITHIN-FRAME TIMELINE (STRICT SEQUENTIAL CAUSALITY)
                # 1. Wait for accelerator to become idle: t_l1_start = max(arrival_ms, accel_idle_at_ms)
                # 2. Execute Layer 1 on accelerator with initial_action: [t_l1_start, t_l1_end]
                # 3. Layer 1 telemetry available; host runs MPC decision: [t_dec_start, t_dec_end]
                #    (Accelerator stalls waiting for Layer 2 configuration)
                # 4. Reconfiguration if chosen_action != initial_action: [t_reconfig_start, t_reconfig_end]
                # 5. Execute Remaining Layers on accelerator with chosen_action: [t_rem_start, t_completion]
                # =====================================================================
                ctrl_c: WithinFrameAdaptiveMPC = controller_obj
                initial_action = current_action or ctrl_c.mpc.fallback

                # Step 1: Wait for accelerator to become idle
                t_l1_start = max(arrival_ms, accel_idle_at_ms)

                # Step 2: Execute Layer 1
                pred_init = self.predictor.predict(initial_action, event.actual_workload_scale, telem)
                l1_frac = event.first_layer_mac_fraction
                l1_lat_ms = pred_init.latency_ms * l1_frac
                l1_energy_mj = pred_init.energy_mj * l1_frac
                t_l1_end = t_l1_start + l1_lat_ms

                # Step 3: Layer 1 observation & MPC decision
                obs = FirstLayerObservation(
                    frame_id=event.frame_id, layer_id=1,
                    mac_count=l1_frac * 1000.0, measured_latency_ms=l1_lat_ms
                )
                t_dec_start = max(t_l1_end, prev_host_dec_end)
                telem_c = Telemetry(
                    temperature_c=current_temp_c,
                    observed_latency_ms=last_res.inference_latency_ms if last_res else None,
                    observed_power_w=None,
                    observed_energy_mj=last_res.total_energy_mj if last_res else None,
                    timestamp_s=arrival_ms / 1000.0,
                    host_queue_wait_ms=host_queue_wait_ms,
                    accel_queue_wait_ms=accel_queue_wait_ms,
                    estimated_host_decision_ms=est_host_dec_ms,
                    preview_latency_ms=0.0,
                    switch_latency_ms=self.switch_latency_ms,
                    layer1_latency_ms=l1_lat_ms,
                    layer1_mac_fraction=l1_frac
                )
                t0 = time.perf_counter()
                rem_energy_est = pred_init.energy_mj * (1.0 - l1_frac)
                lbd: LayerBoundaryDecision = ctrl_c.decide_after_first_layer(
                    obs, telem_c, current_action=initial_action,
                    current_remaining_energy_mj=rem_energy_est
                )
                dec_ms = (time.perf_counter() - t0) * 1000.0
                recent_host_dec_ms.append(dec_ms)
                t_dec_end = t_dec_start + dec_ms
                prev_host_dec_end = t_dec_end
                chosen_action = lbd.decision.action

                # Step 4: Reconfiguration
                reconfig_lat = 0.0
                reconfig_energy = 0.0
                if chosen_action != initial_action:
                    reconfig_lat = self.switch_latency_ms
                    reconfig_energy = self.switch_energy_mj
                t_reconfig_start = t_dec_end
                t_reconfig_end = t_reconfig_start + reconfig_lat

                # Step 5: Remaining layers execution
                t_rem_start = t_reconfig_end
                actual_pred = self.predictor.predict(chosen_action, event.actual_workload_scale, telem_c)
                rem_lat_ms = actual_pred.latency_ms * (1.0 - l1_frac)
                rem_energy_mj = actual_pred.energy_mj * (1.0 - l1_frac)
                t_completion = t_rem_start + rem_lat_ms

                # Totals for Controller C
                inf_latency = l1_lat_ms + rem_lat_ms
                inf_energy = l1_energy_mj + rem_energy_mj
                tot_energy = inf_energy + reconfig_energy
                t_dispatch = t_l1_start
                accel_idle_at_ms = t_completion

                # Piecewise thermal integration across the 3 distinct segments:
                # Segment 1: Layer 1 execution with pred_init.power_w
                tau = max(self.c.thermal_tau_s, 1e-6)
                r_th = self.c.thermal_resistance_c_per_w or 2.0
                dt1_s = l1_lat_ms / 1000.0
                a1 = math.exp(-dt1_s / tau)
                current_temp_c = a1 * current_temp_c + (1.0 - a1) * (self.c.ambient_c + r_th * pred_init.power_w)

                # Segment 2: Decision stall & reconfiguration with quiescent leakage power (0.05 W)
                dt2_s = (dec_ms + reconfig_lat) / 1000.0
                a2 = math.exp(-dt2_s / tau)
                current_temp_c = a2 * current_temp_c + (1.0 - a2) * (self.c.ambient_c + r_th * 0.05)

                # Segment 3: Remaining layers execution with actual_pred.power_w
                dt3_s = rem_lat_ms / 1000.0
                a3 = math.exp(-dt3_s / tau)
                current_temp_c = a3 * current_temp_c + (1.0 - a3) * (self.c.ambient_c + r_th * actual_pred.power_w)
                last_time_ms = t_completion

                pred_c_tot_energy = (
                    (l1_energy_mj + (lbd.decision.predicted_energy_mj or 0.0) + reconfig_energy)
                    if lbd.decision.predicted_energy_mj is not None else None
                )

                rec = NormalizedDecisionRecord(
                    frame_id=event.frame_id, controller_name="Controller C (Within-Frame)",
                    action=chosen_action, action_name=chosen_action.name, mode=lbd.decision.mode, reason=lbd.decision.reason,
                    predicted_latency_ms=lbd.decision.predicted_latency_ms, predicted_power_w=lbd.decision.predicted_power_w,
                    predicted_energy_mj=lbd.decision.predicted_energy_mj, predicted_temp_c=lbd.decision.predicted_temperature_c,
                    decision_latency_ms=dec_ms, preview_latency_ms=0.0, preview_energy_mj=0.0,
                    reconfig_latency_ms=reconfig_lat, reconfig_energy_mj=reconfig_energy,
                    feasible_action_count=lbd.decision.feasible_actions,
                    planned_sequence=lbd.decision.planned_sequence,
                    predicted_e2e_latency_ms=getattr(lbd.decision, "predicted_e2e_latency_ms", None),
                    predicted_total_energy_mj=pred_c_tot_energy,
                    metadata={"initial_action": initial_action.name, "note": lbd.note, "apply_from_layer": 2,
                              "l1_lat_ms": l1_lat_ms, "rem_lat_ms": rem_lat_ms, "stall_ms": dec_ms + reconfig_lat}
                )

            else:
                # =====================================================================
                # FRAME-LEVEL CONTROLLERS (A, B, FIXED, RULE-BASED)
                # =====================================================================
                preview_lat_ms = 0.0
                preview_energy = 0.0

                if controller_key == "controller_b":
                    pipeline: FramePreviewMPCPipeline = controller_obj
                    preview_lat_ms = self.preview_latency_ms
                    preview_energy = self.preview_energy_mj
                    t_prev_start = max(arrival_ms, prev_host_dec_end)
                    t_prev_end = t_prev_start + preview_lat_ms
                    t_dec_start = t_prev_end
                    telem_b = Telemetry(
                        temperature_c=current_temp_c,
                        observed_latency_ms=last_res.inference_latency_ms if last_res else None,
                        observed_power_w=None,
                        observed_energy_mj=last_res.total_energy_mj if last_res else None,
                        timestamp_s=arrival_ms / 1000.0,
                        host_queue_wait_ms=host_queue_wait_ms,
                        accel_queue_wait_ms=accel_queue_wait_ms,
                        estimated_host_decision_ms=est_host_dec_ms,
                        preview_latency_ms=preview_lat_ms,
                        switch_latency_ms=self.switch_latency_ms
                    )
                    t0 = time.perf_counter()
                    est = pipeline.workload_model.predict(event.image_features)
                    pipeline.forecaster.update(est.workload_scale)
                    horizon_scales = pipeline.forecaster.forecast(
                        max(0, pipeline.controller.c.horizon - 1),
                        pipeline.workload_model.min_target, pipeline.workload_model.max_target)
                    current_wl = {"frame_id": event.frame_id, "workload_scale": est.workload_scale, "workload_scale_std": est.uncertainty}
                    future_wls = [{"workload_scale": x, "workload_scale_std": est.uncertainty} for x in horizon_scales]
                    dec = pipeline.controller.choose(current_wl, telem_b, future_wls)
                    dec_ms = (time.perf_counter() - t0) * 1000.0
                    recent_host_dec_ms.append(dec_ms)
                    t_dec_end = t_dec_start + dec_ms
                    prev_host_dec_end = t_dec_end

                    pred_b_tot_energy = (
                        (dec.predicted_energy_mj + preview_energy)
                        if getattr(dec, "predicted_energy_mj", None) is not None else None
                    )

                    rec = NormalizedDecisionRecord(
                        frame_id=event.frame_id, controller_name="Controller B (Preview)",
                        action=dec.action, action_name=dec.action.name, mode=dec.mode, reason=dec.reason,
                        predicted_latency_ms=dec.predicted_latency_ms, predicted_power_w=dec.predicted_power_w,
                        predicted_energy_mj=dec.predicted_energy_mj, predicted_temp_c=dec.predicted_temperature_c,
                        decision_latency_ms=dec_ms, preview_latency_ms=preview_lat_ms,
                        preview_energy_mj=preview_energy, feasible_action_count=dec.feasible_actions,
                        planned_sequence=dec.planned_sequence,
                        predicted_e2e_latency_ms=getattr(dec, "predicted_e2e_latency_ms", None),
                        predicted_total_energy_mj=pred_b_tot_energy,
                        metadata={"preview_in_calib": est.in_calibration_range, "predicted_wl": est.workload_scale}
                    )

                elif controller_key == "controller_a":
                    ctrl_a: HistoryBasedMPC = controller_obj
                    t_dec_start = max(arrival_ms, prev_host_dec_end)
                    telem_a = Telemetry(
                        temperature_c=current_temp_c,
                        observed_latency_ms=last_res.inference_latency_ms if last_res else None,
                        observed_power_w=None,
                        observed_energy_mj=last_res.total_energy_mj if last_res else None,
                        timestamp_s=arrival_ms / 1000.0,
                        host_queue_wait_ms=host_queue_wait_ms,
                        accel_queue_wait_ms=accel_queue_wait_ms,
                        estimated_host_decision_ms=est_host_dec_ms,
                        preview_latency_ms=0.0,
                        switch_latency_ms=self.switch_latency_ms
                    )
                    t0 = time.perf_counter()
                    dec = ctrl_a.decide_before_next_frame(telem_a)
                    dec_ms = (time.perf_counter() - t0) * 1000.0
                    recent_host_dec_ms.append(dec_ms)
                    t_dec_end = t_dec_start + dec_ms
                    prev_host_dec_end = t_dec_end

                    rec = NormalizedDecisionRecord(
                        frame_id=event.frame_id, controller_name="Controller A (History)",
                        action=dec.action, action_name=dec.action.name, mode=dec.mode, reason=dec.reason,
                        predicted_latency_ms=dec.predicted_latency_ms, predicted_power_w=dec.predicted_power_w,
                        predicted_energy_mj=dec.predicted_energy_mj, predicted_temp_c=dec.predicted_temperature_c,
                        decision_latency_ms=dec_ms, feasible_action_count=dec.feasible_actions,
                        planned_sequence=dec.planned_sequence,
                        predicted_e2e_latency_ms=getattr(dec, "predicted_e2e_latency_ms", None),
                        predicted_total_energy_mj=getattr(dec, "predicted_energy_mj", None)
                    )

                elif controller_key == "fixed_nominal":
                    fb: FixedBaselineController = controller_obj
                    rec = fb.decide(event.frame_id, telem, event.deadline_ms)
                    t_dec_start = arrival_ms
                    t_dec_end = arrival_ms

                elif controller_key == "fixed_low_power":
                    fb: FixedBaselineController = controller_obj
                    rec = fb.decide(event.frame_id, telem, event.deadline_ms)
                    t_dec_start = arrival_ms
                    t_dec_end = arrival_ms

                elif controller_key == "rule_based":
                    rb: RuleBasedBaselineController = controller_obj
                    t_dec_start = max(arrival_ms, prev_host_dec_end)
                    rec = rb.decide(event.frame_id, telem, queue_depth, event.deadline_ms)
                    t_dec_end = t_dec_start + rec.decision_latency_ms
                    prev_host_dec_end = t_dec_end

                chosen_action = rec.action

                # Handle INFEASIBLE_SAFE_STATE:
                # If the controller determined that no configuration is feasible and the fallback
                # itself is unsafe or unvalidated, reject the new arriving frame rather than
                # dispatching an unsafe configuration to the accelerator.
                if rec.mode == "INFEASIBLE_SAFE_STATE":
                    reconfig_lat = 0.0
                    reconfig_energy = 0.0
                    rec.reconfig_latency_ms = 0.0
                    rec.reconfig_energy_mj = 0.0
                    t_dispatch = t_dec_end
                    t_completion = t_dec_end
                    inf_latency = 0.0
                    inf_energy = 0.0
                    tot_energy = preview_energy  # only incurred preview overhead
                    # In-progress accelerator execution continues undisturbed (accel_idle_at_ms unchanged)
                    # Quality is None (frame was not processed)
                    actual_quality = None
                    e2e_lat = t_completion - arrival_ms
                    miss = True  # A rejected frame does not meet its processing deadline
                    res = FrameExecutionResult(
                        frame_id=event.frame_id,
                        controller_name=rec.controller_name,
                        arrival_time_ms=arrival_ms,
                        decision_start_ms=t_dec_start,
                        decision_end_ms=t_dec_end,
                        dispatch_time_ms=t_dispatch,
                        completion_time_ms=t_completion,
                        end_to_end_latency_ms=e2e_lat,
                        deadline_miss=miss,
                        inference_latency_ms=inf_latency,
                        inference_energy_mj=inf_energy,
                        total_energy_mj=tot_energy,
                        quality=actual_quality,
                        temperature_c=current_temp_c,
                        queue_depth_at_arrival=queue_depth,
                        decision=rec,
                        host_queue_depth_at_arrival=host_queue_depth,
                        accel_queue_depth_at_arrival=accel_queue_depth,
                        rejected=True,
                        rejection_reason=rec.reason
                    )
                    results.append(res)
                    # current_action remains unchanged; no observation update
                    continue

                # Reconfiguration between frames
                reconfig_lat = 0.0
                reconfig_energy = 0.0
                if current_action is not None and chosen_action != current_action:
                    reconfig_lat = self.switch_latency_ms
                    reconfig_energy = self.switch_energy_mj
                rec.reconfig_latency_ms = reconfig_lat
                rec.reconfig_energy_mj = reconfig_energy
                if rec.predicted_total_energy_mj is not None:
                    rec.predicted_total_energy_mj += reconfig_energy

                # Dispatch: when both decision is ready AND accelerator is idle
                t_reconfig_start = max(t_dec_end, accel_idle_at_ms)
                t_dispatch = t_reconfig_start + reconfig_lat

                actual_pred = self.predictor.predict(chosen_action, event.actual_workload_scale, telem)
                inf_latency = actual_pred.latency_ms
                inf_energy = actual_pred.energy_mj
                t_completion = t_dispatch + inf_latency
                accel_idle_at_ms = t_completion

                dt_exec_s = (t_completion - t_dispatch) / 1000.0
                tau = max(self.c.thermal_tau_s, 1e-6)
                alpha_exec = math.exp(-dt_exec_s / tau)
                current_temp_c = alpha_exec * current_temp_c + (1.0 - alpha_exec) * (self.c.ambient_c + (self.c.thermal_resistance_c_per_w or 2.0) * actual_pred.power_w)
                last_time_ms = t_completion
                tot_energy = inf_energy + preview_energy + reconfig_energy

            e2e_lat = t_completion - arrival_ms
            miss = e2e_lat > event.deadline_ms
            actual_quality = actual_pred.validated_quality

            res = FrameExecutionResult(
                frame_id=event.frame_id,
                controller_name=rec.controller_name,
                arrival_time_ms=arrival_ms,
                decision_start_ms=t_dec_start,
                decision_end_ms=t_dec_end,
                dispatch_time_ms=t_dispatch,
                completion_time_ms=t_completion,
                end_to_end_latency_ms=e2e_lat,
                deadline_miss=miss,
                inference_latency_ms=inf_latency,
                inference_energy_mj=inf_energy,
                total_energy_mj=tot_energy,
                quality=actual_quality,
                temperature_c=current_temp_c,
                queue_depth_at_arrival=queue_depth,
                decision=rec,
                host_queue_depth_at_arrival=host_queue_depth,
                accel_queue_depth_at_arrival=accel_queue_depth
            )
            results.append(res)
            current_action = chosen_action

            if controller_key == "controller_a":
                obs_rec = WorkloadObservation(
                    frame_id=event.frame_id,
                    workload_scale=event.actual_workload_scale,
                    recent_latency_ms=inf_latency
                )
                controller_obj.observe_completed_frame(obs_rec)

        return results


# ---------------------------------------------------------------------------
# Trace Generation
# ---------------------------------------------------------------------------

def generate_evaluation_trace(n_frames: int = 60, frame_rate_fps: float = 30.0,
                              deadline_ms: float = 50.0, seed: int = 42) -> List[FrameArrivalEvent]:
    """Generates a synthetic perception workload trace with realistic scene complexity shifts."""
    np.random.seed(seed)
    events: List[FrameArrivalEvent] = []
    inter_frame_ms = 1000.0 / frame_rate_fps

    for fid in range(n_frames):
        arr_ms = fid * inter_frame_ms

        # Scene Complexity Phases:
        # Phase 1 (0-14): Clear highway (low complexity)
        # Phase 2 (15-29): Urban intersection (medium-high complexity)
        # Phase 3 (30-39): Heavy congestion / sudden pedestrian swarm (peak burst)
        # Phase 4 (40-59): Return to calm road (recovery)
        if fid < 15:
            base_scale = 0.85 + 0.10 * np.sin(fid / 2.0)
            edge_dens = 0.10 + np.random.uniform(0, 0.05)
            entropy = 0.50 + np.random.uniform(0, 0.05)
            temp_diff = 0.05 + np.random.uniform(0, 0.03)
        elif fid < 30:
            base_scale = 1.35 + 0.15 * np.sin(fid / 3.0)
            edge_dens = 0.25 + np.random.uniform(0, 0.08)
            entropy = 0.75 + np.random.uniform(0, 0.08)
            temp_diff = 0.15 + np.random.uniform(0, 0.05)
        elif fid < 40:
            # Sudden peak burst
            base_scale = 1.70 + np.random.uniform(-0.05, 0.15)
            edge_dens = 0.35 + np.random.uniform(0, 0.05)
            entropy = 0.88 + np.random.uniform(0, 0.04)
            temp_diff = 0.22 + np.random.uniform(0, 0.04)
        else:
            base_scale = 0.95 + 0.10 * np.cos(fid / 3.0)
            edge_dens = 0.12 + np.random.uniform(0, 0.06)
            entropy = 0.55 + np.random.uniform(0, 0.05)
            temp_diff = 0.06 + np.random.uniform(0, 0.04)

        wl_scale = max(0.5, float(base_scale + np.random.normal(0, 0.03)))
        features = {
            "brightness_mean": 0.5 + float(np.random.normal(0, 0.05)),
            "brightness_std": 0.2 + float(np.random.normal(0, 0.02)),
            "edge_density": float(edge_dens),
            "histogram_entropy": float(entropy),
            "temporal_difference": float(temp_diff),
            "aspect_ratio": 16.0 / 9.0,
            "normalized_pixel_count": 1.0
        }
        event = FrameArrivalEvent(
            frame_id=fid,
            arrival_time_ms=arr_ms,
            deadline_ms=deadline_ms,
            actual_workload_scale=wl_scale,
            first_layer_mac_fraction=0.15,
            image_features=features
        )
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Metrics Aggregation
# ---------------------------------------------------------------------------

def compute_summary_metrics(results: List[FrameExecutionResult]) -> Dict[str, Any]:
    if not results:
        return {}
    e2e_lats = [r.end_to_end_latency_ms for r in results]
    energies = [r.total_energy_mj for r in results]
    dec_times = [r.decision.decision_latency_ms for r in results]
    misses = sum(1 for r in results if r.deadline_miss)
    fallback_count = sum(1 for r in results if "FALLBACK" in r.decision.mode)
    rejected_count = sum(1 for r in results if getattr(r, "rejected", False) or r.decision.mode == "INFEASIBLE_SAFE_STATE")
    qualities = [r.quality for r in results if r.quality is not None]
    queue_depths = [r.queue_depth_at_arrival for r in results]
    host_queue_depths = [r.host_queue_depth_at_arrival for r in results]
    accel_queue_depths = [r.accel_queue_depth_at_arrival for r in results]

    e2e_lats.sort()
    dec_times.sort()
    p95_lat = e2e_lats[int(0.95 * len(e2e_lats))]
    p95_dec = dec_times[int(0.95 * len(dec_times))]

    total_span_s = max(1e-3, (results[-1].completion_time_ms - results[0].arrival_time_ms) / 1000.0)
    total_span_ms = total_span_s * 1000.0
    throughput_fps = len(results) / total_span_s

    on_time_completions = len(results) - misses
    on_time_rate_pct = 100.0 * on_time_completions / len(results)
    on_time_throughput_fps = on_time_completions / total_span_s

    # Accelerator busy time: total time the accelerator was actively occupied by frames
    # (For Controller C, includes Layer 1 + decision stall + reconfiguration + remaining layers)
    if results[0].controller_name == "Controller C (Within-Frame)":
        accel_busy_ms = sum(
            r.inference_latency_ms + r.decision.decision_latency_ms + r.decision.reconfig_latency_ms
            for r in results
        )
    else:
        accel_busy_ms = sum(r.inference_latency_ms + r.decision.reconfig_latency_ms for r in results)

    # Host compute time: time host CPU spends running preview and decisions
    host_busy_ms = sum(r.decision.decision_latency_ms + r.decision.preview_latency_ms for r in results)
    host_utilization_pct = min(100.0, 100.0 * host_busy_ms / total_span_ms)
    accel_utilization_pct = min(100.0, 100.0 * accel_busy_ms / total_span_ms)

    return {
        "controller": results[0].controller_name,
        "total_frames": len(results),
        "total_energy_mj": float(sum(energies)),
        "mean_energy_per_frame_mj": float(np.mean(energies)),
        "mean_e2e_latency_ms": float(np.mean(e2e_lats)),
        "median_e2e_latency_ms": float(np.median(e2e_lats)),
        "p95_e2e_latency_ms": float(p95_lat),
        "max_e2e_latency_ms": float(max(e2e_lats)),
        "deadline_misses": int(misses),
        "deadline_miss_rate_pct": float(100.0 * misses / len(results)),
        "on_time_completions": int(on_time_completions),
        "on_time_rate_pct": float(on_time_rate_pct),
        "mean_quality": float(np.mean(qualities)) if qualities else None,
        "mean_decision_ms": float(np.mean(dec_times)),
        "p95_decision_ms": float(p95_dec),
        "fallback_count": int(fallback_count),
        "fallback_rate_pct": float(100.0 * fallback_count / len(results)),
        "rejected_count": int(rejected_count),
        "rejected_rate_pct": float(100.0 * rejected_count / len(results)),
        "final_temp_c": float(results[-1].temperature_c),
        "max_temp_c": float(max(r.temperature_c for r in results)),
        "max_queue_depth": int(max(queue_depths)),
        "mean_queue_depth": float(np.mean(queue_depths)),
        "max_host_queue_depth": int(max(host_queue_depths)),
        "mean_host_queue_depth": float(np.mean(host_queue_depths)),
        "max_accel_queue_depth": int(max(accel_queue_depths)),
        "mean_accel_queue_depth": float(np.mean(accel_queue_depths)),
        "throughput_fps": float(throughput_fps),
        "on_time_throughput_fps": float(on_time_throughput_fps),
        "total_time_span_ms": float(total_span_ms),
        "accel_busy_time_ms": float(accel_busy_ms),
        "accel_idle_time_ms": float(max(0.0, total_span_ms - accel_busy_ms)),
        "accel_utilization_pct": float(accel_utilization_pct),
        "host_busy_time_ms": float(host_busy_ms),
        "host_utilization_pct": float(host_utilization_pct),
        "energy_scope": "accelerator_subsystem (inference + preview + reconfig; host CPU unmetered)"
    }


# Central deadline scenario constants
DEFAULT_DEADLINE_MS = 50.0
STRICT_30FPS_DEADLINE_MS = 33.333


def run_full_comparison(profiles_csv: str = "synthetic_calibrated_profiles.csv",
                        workload_model_json: str = "synthetic_workload_model.json",
                        n_frames: int = 60, deadline_ms: float = DEFAULT_DEADLINE_MS,
                        warm_up: bool = True,
                        output_csv: str = "replay_comparison_results.csv",
                        output_summary_json: str = "replay_aggregate_summary.json",
                        output_summary_csv: str = "replay_aggregate_summary.csv"
                        ) -> Dict[str, Dict[str, Any]]:
    # Load predictor with validated profiles
    predictor = ProfilePerformancePredictor(profiles_csv, allow_uncalibrated_quality=True)
    validated_actions = tuple(predictor.profiles.keys())
    fallback_action = Action(0.50, 1, 4)

    constraints = Constraints(
        deadline_ms=deadline_ms,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        ambient_c=35.0,
        thermal_tau_s=27.0,
        thermal_resistance_c_per_w=2.0,
        horizon=5,
        beam_width=32,
        switch_penalty_mj=DEFAULT_SWITCH_ENERGY_MJ,
        switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS,
    )

    # 1. Instantiate Controllers
    # Controller A
    core_mpc_a = MPCController(predictor, validated_actions, fallback=fallback_action, constraints=constraints)
    ctrl_a = HistoryBasedMPC(core_mpc_a, cold_start_nominal_scale=1.0)

    # Controller B
    ctrl_b = FramePreviewMPCPipeline(
        workload_model_path=workload_model_json,
        profiles_csv=profiles_csv,
        constraints=constraints,
        fallback=fallback_action,
        require_thermal_calibration=False,
        allow_uncalibrated_quality=True
    )

    # Controller C
    core_mpc_c = MPCController(predictor, validated_actions, fallback=fallback_action, constraints=constraints)
    est_c = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(core_mpc_c, est_c, measured_switch_cost_mj=DEFAULT_SWITCH_ENERGY_MJ)

    # Baselines
    nominal_action = Action(1.00, 3, 8)
    low_pwr_action = Action(0.50, 1, 4)
    ctrl_fixed_nom = FixedBaselineController(nominal_action, "Fixed Nominal (W1.00-E3-INT8)")
    ctrl_fixed_low = FixedBaselineController(low_pwr_action, "Fixed Low-Power (W0.50-E1-INT4)")
    ctrl_rule_based = RuleBasedBaselineController(validated_actions, "Rule-Based Baseline")

    # Warm up CPU governor and JIT before replay measurement if requested
    # (Executes 5 dummy optimization calls that are strictly excluded from the 60-frame trace)
    if warm_up:
        dummy_telem = Telemetry(
            temperature_c=35.0, timestamp_s=0.0,
            host_queue_wait_ms=0.0, accel_queue_wait_ms=0.0,
            estimated_host_decision_ms=12.0, preview_latency_ms=0.0,
            switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS
        )
        for _ in range(5):
            ctrl_a.mpc.choose(1.0, dummy_telem)
            ctrl_b.controller.choose(
                {"frame_id": 0, "workload_scale": 1.0, "workload_scale_std": 0.05},
                dummy_telem,
                [{"workload_scale": 1.0, "workload_scale_std": 0.05}] * 4
            )
            ctrl_c.mpc.choose(1.0, dummy_telem)

    # Generate shared trace with central deadline
    trace = generate_evaluation_trace(n_frames=n_frames, deadline_ms=deadline_ms)

    simulator = ReplaySimulator(
        predictor, constraints,
        switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS,
        switch_energy_mj=DEFAULT_SWITCH_ENERGY_MJ,
        preview_latency_ms=DEFAULT_PREVIEW_LATENCY_MS,
        preview_energy_mj=DEFAULT_PREVIEW_ENERGY_MJ
    )

    controllers = [
        ("controller_a", ctrl_a),
        ("controller_b", ctrl_b),
        ("controller_c", ctrl_c),
        ("fixed_nominal", ctrl_fixed_nom),
        ("fixed_low_power", ctrl_fixed_low),
        ("rule_based", ctrl_rule_based)
    ]

    all_summaries: Dict[str, Dict[str, Any]] = {}
    all_frame_rows: List[Dict[str, Any]] = []

    for key, c_obj in controllers:
        res = simulator.run_trace(key, c_obj, trace)
        summary = compute_summary_metrics(res)
        all_summaries[summary["controller"]] = summary

        for r in res:
            all_frame_rows.append({
                "frame_id": r.frame_id,
                "controller": r.controller_name,
                "arrival_ms": f"{r.arrival_time_ms:.2f}",
                "dispatch_ms": f"{r.dispatch_time_ms:.2f}",
                "completion_ms": f"{r.completion_time_ms:.2f}",
                "e2e_latency_ms": f"{r.end_to_end_latency_ms:.2f}",
                "deadline_miss": r.deadline_miss,
                "action": r.decision.action_name,
                "mode": r.decision.mode,
                "reason": r.decision.reason,
                "decision_ms": f"{r.decision.decision_latency_ms:.2f}",
                "preview_ms": f"{r.decision.preview_latency_ms:.2f}",
                "reconfig_ms": f"{r.decision.reconfig_latency_ms:.2f}",
                "total_energy_mj": f"{r.total_energy_mj:.3f}",
                "quality": f"{r.quality:.3f}" if r.quality is not None else "UNKNOWN",
                "temp_c": f"{r.temperature_c:.2f}",
                "queue_depth": r.queue_depth_at_arrival,
                "host_queue_depth": r.host_queue_depth_at_arrival,
                "accel_queue_depth": r.accel_queue_depth_at_arrival,
                "pred_e2e_latency_ms": f"{r.decision.predicted_e2e_latency_ms:.2f}" if r.decision.predicted_e2e_latency_ms is not None else "N/A",
                "pred_total_energy_mj": f"{r.decision.predicted_total_energy_mj:.3f}" if r.decision.predicted_total_energy_mj is not None else "N/A",
                "rejected": getattr(r, "rejected", False),
                "rejection_reason": getattr(r, "rejection_reason", "") or ""
            })

    # Save detailed per-frame CSV
    if all_frame_rows and output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_frame_rows)

    # Save aggregate summary JSON
    if output_summary_json:
        with open(output_summary_json, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)

    # Save aggregate summary CSV
    if output_summary_csv and all_summaries:
        summary_rows = list(all_summaries.values())
        with open(output_summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    return all_summaries


def print_summary_table(summaries: Dict[str, Dict[str, Any]], title: str):
    print("=" * 115)
    print(title)
    print("=" * 115)
    header = (
        f"{'Controller':<30} | {'Energy(mJ)':<10} | {'E2E Lat(ms)':<11} | "
        f"{'Misses(%)':<10} | {'On-Time FPS':<11} | {'Quality':<7} | {'Dec(ms)':<7} | {'MaxHostQ':<8} | {'MaxAccQ':<8}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        q_str = f"{s['mean_quality']:.3f}" if s["mean_quality"] is not None else "N/A"
        print(
            f"{name:<30} | {s['total_energy_mj']:10.2f} | "
            f"{s['mean_e2e_latency_ms']:11.2f} | "
            f"{s['deadline_miss_rate_pct']:9.1f}% | "
            f"{s.get('on_time_throughput_fps', s['throughput_fps']):11.2f} | "
            f"{q_str:<7} | "
            f"{s['mean_decision_ms']:7.2f} | "
            f"{s.get('max_host_queue_depth', s['max_queue_depth']):<8} | "
            f"{s.get('max_accel_queue_depth', 0):<8}"
        )
    print("=" * 115)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Controller Replay Evaluation")
    parser.add_argument("--deadline", type=float, default=None,
                        help="Specific deadline in ms (e.g. 50.0 or 33.333)")
    parser.add_argument("--cold-start", action="store_true",
                        help="Run without CPU governor warm-up (cold-start)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames (default: 60)")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Optional tag appended to output filenames to prevent overwriting baseline results")
    args = parser.parse_args()

    warm_up_flag = not args.cold_start
    warm_str = "WARMED-UP" if warm_up_flag else "COLD-START"

    warm_suffix = "_cold" if args.cold_start else ""
    tag = f"_{args.output_tag}" if args.output_tag else ""

    if args.deadline is not None:
        d = args.deadline
        title = f"REPLAY EVALUATION (DEADLINE: {d:.3f} ms, 60 FRAMES @ 30 FPS, {warm_str})"
        out_csv = f"replay_comparison_{int(d)}ms{warm_suffix}{tag}.csv"
        out_json = f"replay_aggregate_summary_{int(d)}ms{warm_suffix}{tag}.json"
        out_sum_csv = f"replay_aggregate_summary_{int(d)}ms{warm_suffix}{tag}.csv"
        s = run_full_comparison(n_frames=args.frames, deadline_ms=d, warm_up=warm_up_flag,
                                output_csv=out_csv, output_summary_json=out_json, output_summary_csv=out_sum_csv)
        print_summary_table(s, title)
    else:
        # Default execution: run both standard scenarios: 50.0 ms and 33.333 ms
        print("\n>>> EXECUTING SCENARIO 1: 50.0 ms DEADLINE (STANDARD SCENARIO, 60 FRAMES @ 30 FPS, WARMED-UP)")
        out_50_csv = f"replay_comparison_50ms{warm_suffix}{tag}.csv" if tag else "replay_comparison_results.csv"
        out_50_json = f"replay_aggregate_summary_50ms{warm_suffix}{tag}.json" if tag else "replay_aggregate_summary.json"
        out_50_sum_csv = f"replay_aggregate_summary_50ms{warm_suffix}{tag}.csv" if tag else "replay_aggregate_summary.csv"

        s_50 = run_full_comparison(
            n_frames=args.frames, deadline_ms=DEFAULT_DEADLINE_MS, warm_up=True,
            output_csv=out_50_csv,
            output_summary_json=out_50_json,
            output_summary_csv=out_50_sum_csv
        )
        if not tag:
            # Also copy to dedicated 50ms artifact files if running without tag
            with open("replay_comparison_50ms.csv", "w", newline="") as f:
                with open("replay_comparison_results.csv") as src:
                    f.write(src.read())
            with open("replay_aggregate_summary_50ms.json", "w", encoding="utf-8") as f:
                json.dump(s_50, f, indent=2)
            with open("replay_aggregate_summary_50ms.csv", "w", newline="", encoding="utf-8") as f:
                with open("replay_aggregate_summary.csv") as src:
                    f.write(src.read())
        print_summary_table(s_50, "SCENARIO 1: 50.0 ms DEADLINE SUMMARY (WARMED-UP)")

        print("\n>>> EXECUTING SCENARIO 2: 33.333 ms DEADLINE (STRICT 30 FPS SCENARIO, 60 FRAMES @ 30 FPS, WARMED-UP)")
        out_33_csv = f"replay_comparison_33ms{warm_suffix}{tag}.csv"
        out_33_json = f"replay_aggregate_summary_33ms{warm_suffix}{tag}.json"
        out_33_sum_csv = f"replay_aggregate_summary_33ms{warm_suffix}{tag}.csv"
        s_33 = run_full_comparison(
            n_frames=args.frames, deadline_ms=STRICT_30FPS_DEADLINE_MS, warm_up=True,
            output_csv=out_33_csv,
            output_summary_json=out_33_json,
            output_summary_csv=out_33_sum_csv
        )
        print_summary_table(s_33, "SCENARIO 2: 33.333 ms DEADLINE SUMMARY (STRICT 30 FPS, WARMED-UP)")

        print("\nArtifacts generated:")
        if tag:
            print(f"  - 50.0 ms Scenario: {out_50_csv}, {out_50_json}, {out_50_sum_csv}")
            print(f"  - 33.333 ms Scenario: {out_33_csv}, {out_33_json}, {out_33_sum_csv}")
        else:
            print("  - 50.0 ms Scenario: replay_comparison_50ms.csv, replay_aggregate_summary_50ms.json, replay_aggregate_summary_50ms.csv")
            print("  - 33.333 ms Scenario: replay_comparison_33ms.csv, replay_aggregate_summary_33ms.json, replay_aggregate_summary_33ms.csv")
            print("  - Default Baseline: replay_comparison_results.csv, replay_aggregate_summary.json, replay_aggregate_summary.csv")

