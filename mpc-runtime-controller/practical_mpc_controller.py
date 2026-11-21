#!/usr/bin/env python3
"""Hardware-adaptable receding-horizon controller for a configurable DNN accelerator.

This is a controller core, not a hardware driver or a safety certification.
It intentionally contains NO fabricated latency/power/quality table. Integrate a
measured predictor and real telemetry through the interfaces below.

Source-aligned initial configuration space: 3 widths x 3 exits x 2 precisions.
Initial constraint defaults from The-Control-Loop-Notebook: 50 ms deadline,
2 W power cap, 70 C temperature cap, q >= 0.85, thermal tau 27 s.
Ambient temperature defaults to 35 C because that value appears in the notebook's
thermal-model discussion; calibrate all thermal parameters on the target platform.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Set
import math

@dataclass(frozen=True, order=True)
class Action:
    width: float
    exit: int
    bits: int

    @property
    def name(self) -> str:
        return f"W{self.width:.2f}-E{self.exit}-INT{self.bits}"

DEFAULT_ACTIONS = tuple(Action(w, e, b) for w, e, b in product((0.50, 0.75, 1.00), (1, 2, 3), (4, 8)))

# Shared hardware/accelerator overhead defaults
DEFAULT_SWITCH_ENERGY_MJ: float = 0.02
DEFAULT_SWITCH_LATENCY_MS: float = 0.5
DEFAULT_PREVIEW_ENERGY_MJ: float = 0.03
DEFAULT_PREVIEW_LATENCY_MS: float = 1.2

@dataclass(frozen=True)
class Constraints:
    deadline_ms: float = 50.0
    power_cap_w: float = 2.0
    temperature_cap_c: float = 70.0
    quality_floor: float = 0.85
    # Optional per-horizon-step quality requirements; absent means scalar floor.
    quality_floor_horizon: Optional[Tuple[float, ...]] = None
    ambient_c: float = 35.0
    thermal_tau_s: float = 27.0
    # Effective thermal resistance is hardware-specific; MUST be calibrated.
    thermal_resistance_c_per_w: Optional[float] = None
    decision_period_s: float = 0.05
    uncertainty_k: float = 1.0
    switch_penalty_mj: float = DEFAULT_SWITCH_ENERGY_MJ
    switch_latency_ms: float = DEFAULT_SWITCH_LATENCY_MS
    latency_penalty: float = 0.0
    horizon: int = 5
    beam_width: int = 32

@dataclass(frozen=True)
class Prediction:
    latency_ms: float
    power_w: float
    energy_mj: float
    # Offline, labelled validation score/accuracy for this action.
    validated_quality: Optional[float]
    # Standard deviations / uncertainty bounds; use measured values when available.
    latency_std_ms: float = 0.0
    power_std_w: float = 0.0
    energy_std_mj: float = 0.0
    quality_std: float = 0.0

@dataclass
class Telemetry:
    temperature_c: float
    observed_latency_ms: Optional[float] = None
    observed_power_w: Optional[float] = None
    observed_energy_mj: Optional[float] = None
    runtime_quality: Optional[float] = None
    quality_sensor_valid: bool = False
    elapsed_cycles: Optional[int] = None
    watchdog_cycles: Optional[int] = None
    timestamp_s: float = 0.0
    # Scheduling context for causal end-to-end feasibility evaluation
    host_queue_wait_ms: float = 0.0
    accel_queue_wait_ms: float = 0.0
    estimated_host_decision_ms: float = 0.0
    preview_latency_ms: float = 0.0
    switch_latency_ms: float = 0.0
    layer1_latency_ms: float = 0.0
    layer1_mac_fraction: float = 0.0

class Predictor(Protocol):
    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        """Return calibrated predictions for this action and workload."""
        ...

@dataclass
class Decision:
    action: Action
    mode: str
    reason: str
    predicted_latency_ms: Optional[float] = None
    predicted_power_w: Optional[float] = None
    predicted_energy_mj: Optional[float] = None
    predicted_temperature_c: Optional[float] = None
    feasible_actions: int = 0
    planned_sequence: Tuple[str, ...] = field(default_factory=tuple)
    predicted_e2e_latency_ms: Optional[float] = None

class ControllerConfigurationError(ValueError):
    pass

class MPCController:
    """Receding-horizon MPC. `predictor` and hardware I/O are supplied by integrator.

    `validated_actions` must contain only actions whose quality was checked against
    labelled held-out data. The fallback must be among them. Unknown quality does
    not get silently treated as good: an action with no validated quality is never
    admitted by the MPC.
    """
    def __init__(self, predictor: Predictor, validated_actions: Iterable[Action],
                 fallback: Action = Action(0.50, 1, 4), constraints: Constraints = Constraints()):
        self.predictor = predictor
        self.validated_actions = tuple(sorted(set(validated_actions)))
        self.fallback = fallback
        self.c = constraints
        self.validate_constraints(self.c)
        if not self.validated_actions:
            raise ControllerConfigurationError("No validated actions supplied; refusing to control hardware.")
        if fallback not in self.validated_actions:
            raise ControllerConfigurationError("Fallback must be in the validated action set.")
        if self.c.thermal_resistance_c_per_w is None:
            # Do not invent a thermal parameter. Safe operation requires calibration.
            self.thermal_model_ready = False
        else:
            self.thermal_model_ready = True
        self.previous_action: Optional[Action] = None

    def validate_constraints(self, constraints: Constraints) -> None:
        """Validate baseline or decision-scoped constraints without mutating state."""
        if not isinstance(constraints, Constraints):
            raise TypeError("constraints must be a Constraints instance")
        if constraints.horizon != self.c.horizon:
            raise ValueError(
                f"constraints horizon mismatch: expected {self.c.horizon}, "
                f"received {constraints.horizon}"
            )
        if not math.isfinite(constraints.quality_floor):
            raise ValueError("quality_floor must be finite")
        if not 0.0 <= constraints.quality_floor <= 1.0:
            raise ValueError("quality_floor must be within [0, 1]")
        floors = constraints.quality_floor_horizon
        if floors is not None:
            if len(floors) != self.c.horizon:
                raise ValueError(
                    f"quality_floor_horizon length must equal horizon "
                    f"({self.c.horizon}); received {len(floors)}"
                )
            for step, floor in enumerate(floors):
                if not math.isfinite(floor):
                    raise ValueError(f"quality_floor_horizon[{step}] must be finite")
                if not 0.0 <= floor <= 1.0:
                    raise ValueError(
                        f"quality_floor_horizon[{step}] must be within [0, 1]"
                    )

    def _temperature_next(self, current_c: float, power_w: float,
                          constraints: Optional[Constraints] = None) -> Optional[float]:
        c = constraints if constraints is not None else self.c
        if c.thermal_resistance_c_per_w is None:
            return None
        tau = max(c.thermal_tau_s, 1e-6)
        a = math.exp(-c.decision_period_s / tau)
        equilibrium = c.ambient_c + c.thermal_resistance_c_per_w * power_w
        return a * current_c + (1.0 - a) * equilibrium

    def _robust(self, p: Prediction,
                constraints: Optional[Constraints] = None):
        c = constraints if constraints is not None else self.c
        k = c.uncertainty_k
        return (p.latency_ms + k*p.latency_std_ms,
                p.power_w + k*p.power_std_w,
                p.energy_mj + k*p.energy_std_mj,
                None if p.validated_quality is None else p.validated_quality-k*p.quality_std)

    def _hard_override(self, t: Telemetry,
                       constraints: Optional[Constraints] = None) -> Optional[str]:
        c = constraints if constraints is not None else self.c
        if t.watchdog_cycles is not None and t.elapsed_cycles is not None and t.elapsed_cycles >= t.watchdog_cycles:
            return "watchdog budget exceeded"
        if t.temperature_c >= c.temperature_cap_c:
            return "temperature cap reached"
        if t.observed_power_w is not None and t.observed_power_w >= c.power_cap_w:
            return "measured power cap reached"
        q_floor = (c.quality_floor_horizon[0]
                   if c.quality_floor_horizon is not None else c.quality_floor)
        if t.quality_sensor_valid and t.runtime_quality is not None and t.runtime_quality < q_floor:
            return "runtime quality monitor below floor"
        return None

    def _estimate_e2e_latency(self, action: Action, step: int, seq: Tuple[Action, ...],
                              robust_inf_lat: float, telemetry: Telemetry,
                              constraints: Optional[Constraints] = None) -> float:
        """Estimate E2E latency including queues, preview, host decision and reconfiguration."""
        c = constraints if constraints is not None else self.c
        switch_lat = getattr(telemetry, 'switch_latency_ms', 0.0) or c.switch_latency_ms
        if step == 0:
            prev_act = self.previous_action
            reconfig_lat = switch_lat if (prev_act is not None and action != prev_act) else 0.0
            l1_frac = getattr(telemetry, 'layer1_mac_fraction', 0.0)
            if l1_frac > 0.0:
                l1_lat = getattr(telemetry, 'layer1_latency_ms', 0.0)
                accel_q = getattr(telemetry, 'accel_queue_wait_ms', 0.0)
                host_q = getattr(telemetry, 'host_queue_wait_ms', 0.0)
                est_dec = getattr(telemetry, 'estimated_host_decision_ms', 0.0)
                t_host_ready = max(accel_q + l1_lat, host_q) + est_dec
                return t_host_ready + reconfig_lat + robust_inf_lat
            host_q = getattr(telemetry, 'host_queue_wait_ms', 0.0)
            prev_lat = getattr(telemetry, 'preview_latency_ms', 0.0)
            est_dec = getattr(telemetry, 'estimated_host_decision_ms', 0.0)
            accel_q = getattr(telemetry, 'accel_queue_wait_ms', 0.0)
            t_host_ready = host_q + prev_lat + est_dec
            t_accel_ready = accel_q
            return max(t_host_ready, t_accel_ready) + reconfig_lat + robust_inf_lat
        prev_act = seq[-1] if seq else self.previous_action
        reconfig_lat = switch_lat if (prev_act is not None and action != prev_act) else 0.0
        prev_lat = getattr(telemetry, 'preview_latency_ms', 0.0)
        est_dec = getattr(telemetry, 'estimated_host_decision_ms', 0.0)
        return prev_lat + est_dec + reconfig_lat + robust_inf_lat

    def _beam_sort_key(self, state: Tuple[float, float, Optional[Action], Tuple[Action, ...]],
                       pred_step0: Dict[Action, Prediction]) -> Tuple[float, int, float, float, str]:
        """Strict deterministic tie-breaking:
        1. Cumulative energy score (rounded to microjoules 1e-6 mJ).
        2. Hysteresis: prefer current action (churn_key=0) to avoid unnecessary switching.
        3. Quality margin: higher validated robust quality preferred.
        4. Latency margin: lower E2E latency preferred.
        5. Lexical order on action name for 100% deterministic reproducibility.
        """
        score, temp, first_action, seq = state
        score_key = round(score, 6)
        churn_key = 0 if (self.previous_action is not None and first_action == self.previous_action) else 1
        p = pred_step0.get(first_action) if first_action is not None else None
        qual = p.validated_quality if (p and p.validated_quality is not None) else 0.0
        qual_key = -round(qual, 4)
        lat = p.latency_ms if p else 0.0
        lat_key = round(lat, 4)
        name_key = first_action.name if first_action is not None else ""
        return (score_key, churn_key, qual_key, lat_key, name_key)

    def choose(self, workload_now: object, telemetry: Telemetry,
               workload_forecast: Sequence[object] = (),
               constraints_override: Optional[Constraints] = None) -> Decision:
        # Runtime constraints are local to this decision; self.c is never replaced.
        c = self.c if constraints_override is None else constraints_override
        if constraints_override is not None:
            self.validate_constraints(constraints_override)

        rejection_reasons: Set[str] = set()
        rejection_counts: Dict[int, Dict[str, int]] = {}

        def reject(step: int, category: str, message: str) -> None:
            rejection_reasons.add(message)
            per_step = rejection_counts.setdefault(step, {})
            per_step[category] = per_step.get(category, 0) + 1

        self.last_rejection_counts = rejection_counts
        self.last_rejection_reasons = tuple()

        hard_override = self._hard_override(telemetry, c)
        if hard_override:
            self.previous_action = self.fallback
            return Decision(self.fallback, "SAFETY_FALLBACK", hard_override)

        thermal_mode = c.thermal_resistance_c_per_w is not None

        workloads = [workload_now]
        if workload_forecast:
            workloads.extend(list(workload_forecast[:max(0, c.horizon - 1)]))
        while len(workloads) < c.horizon:
            workloads.append(workloads[-1])

        pred: List[Dict[Action, Prediction]] = []
        for wl in workloads:
            row: Dict[Action, Prediction] = {}
            for action in self.validated_actions:
                try:
                    prediction = self.predictor.predict(action, wl, telemetry)
                    values = [prediction.latency_ms, prediction.power_w, prediction.energy_mj]
                    if any(not math.isfinite(value) or value < 0 for value in values):
                        continue
                    row[action] = prediction
                except (ValueError, RuntimeError, KeyError, NotImplementedError):
                    continue
            pred.append(row)

        step_candidates: List[List[Tuple[Action, Prediction, float, float, float, Optional[float]]]] = []
        for step in range(c.horizon):
            valid_candidates = []
            q_floor = (c.quality_floor_horizon[step]
                       if c.quality_floor_horizon is not None else c.quality_floor)
            for action, prediction in pred[step].items():
                lat, power, energy, quality = self._robust(prediction, c)
                if power > c.power_cap_w:
                    reject(step, "power", f"power ({power:.2f}W > {c.power_cap_w:.2f}W)")
                    continue
                if quality is None:
                    reject(step, "quality_unvalidated", "quality_unvalidated")
                    continue
                if quality < q_floor:
                    reject(step, "quality", f"quality ({quality:.3f} < {q_floor:.3f})")
                    continue
                valid_candidates.append((action, prediction, lat, power, energy, quality))
            step_candidates.append(valid_candidates)

        pred_step0 = pred[0]
        prev_act = self.previous_action
        action_meta: Dict[Action, Tuple[int, float, float, str]] = {}
        for action in self.validated_actions:
            prediction = pred_step0.get(action)
            quality = (prediction.validated_quality
                       if prediction and prediction.validated_quality is not None else 0.0)
            latency = prediction.latency_ms if prediction else 0.0
            churn = 0 if prev_act is not None and action == prev_act else 1
            action_meta[action] = (churn, -round(quality, 4), round(latency, 4), action.name)

        switch_lat = getattr(telemetry, 'switch_latency_ms', 0.0) or c.switch_latency_ms
        host_q = getattr(telemetry, 'host_queue_wait_ms', 0.0)
        accel_q = getattr(telemetry, 'accel_queue_wait_ms', 0.0)
        prev_lat = getattr(telemetry, 'preview_latency_ms', 0.0)
        est_dec = getattr(telemetry, 'estimated_host_decision_ms', 0.0)
        l1_frac = getattr(telemetry, 'layer1_mac_fraction', 0.0)
        l1_lat = getattr(telemetry, 'layer1_latency_ms', 0.0)
        if l1_frac > 0.0:
            step0_t_host_ready = max(accel_q + l1_lat, host_q) + est_dec
            is_l1 = True
        else:
            step0_t_host_ready = host_q + prev_lat + est_dec
            is_l1 = False

        c_deadline = c.deadline_ms
        c_temp_cap = c.temperature_cap_c
        c_switch_penalty = c.switch_penalty_mj
        c_lat_penalty = c.latency_penalty
        c_beam_width = max(1, c.beam_width)

        # Beam state: (cumulative score, temperature, first action, action sequence).
        beam = [(0.0, telemetry.temperature_c, None, tuple())]
        first_step_feasible: Set[Action] = set()

        for step in range(c.horizon):
            next_beam = []
            candidates = step_candidates[step]
            if step == 0:
                for score, temp, first, sequence in beam:
                    for action, prediction, lat, power, energy, quality in candidates:
                        reconfig = switch_lat if (prev_act is not None and action != prev_act) else 0.0
                        if is_l1:
                            e2e_lat = step0_t_host_ready + reconfig + lat
                        else:
                            e2e_lat = max(step0_t_host_ready, accel_q) + reconfig + lat
                        if e2e_lat > c_deadline:
                            reject(step, "e2e_latency",
                                   f"e2e_latency ({e2e_lat:.1f}ms > {c_deadline:.1f}ms)")
                            continue
                        tnext = self._temperature_next(temp, power, c) if thermal_mode else None
                        if thermal_mode and (tnext is None or tnext >= c_temp_cap):
                            reject(step, "temperature",
                                   f"temperature ({tnext:.1f}C >= {c_temp_cap:.1f}C)")
                            continue
                        first_step_feasible.add(action)
                        step_score = energy
                        if c_lat_penalty > 0.0:
                            step_score += c_lat_penalty * (e2e_lat / c_deadline) ** 2
                        if prev_act is not None and prev_act != action:
                            step_score += c_switch_penalty
                        next_beam.append((
                            score + step_score, temp if tnext is None else tnext,
                            action, (action,)
                        ))
            else:
                for score, temp, first, sequence in beam:
                    last_action = sequence[-1]
                    for action, prediction, lat, power, energy, quality in candidates:
                        reconfig = switch_lat if last_action != action else 0.0
                        e2e_lat = prev_lat + est_dec + reconfig + lat
                        if e2e_lat > c_deadline:
                            reject(step, "e2e_latency",
                                   f"e2e_latency ({e2e_lat:.1f}ms > {c_deadline:.1f}ms)")
                            continue
                        tnext = self._temperature_next(temp, power, c) if thermal_mode else None
                        if thermal_mode and (tnext is None or tnext >= c_temp_cap):
                            reject(step, "temperature",
                                   f"temperature ({tnext:.1f}C >= {c_temp_cap:.1f}C)")
                            continue
                        step_score = energy
                        if c_lat_penalty > 0.0:
                            step_score += c_lat_penalty * (e2e_lat / c_deadline) ** 2
                        if last_action != action:
                            step_score += c_switch_penalty
                        next_beam.append((
                            score + step_score, temp if tnext is None else tnext,
                            first, sequence + (action,)
                        ))

            if not next_beam:
                break
            next_beam.sort(key=lambda state: (
                round(state[0], 6), action_meta[state[2]][0],
                action_meta[state[2]][1], action_meta[state[2]][2],
                action_meta[state[2]][3]
            ))
            beam = next_beam[:c_beam_width]

        self.last_rejection_counts = {
            step: dict(counts) for step, counts in rejection_counts.items()
        }
        self.last_rejection_reasons = tuple(sorted(rejection_reasons))

        complete_sequences = [state for state in beam if len(state[3]) == c.horizon]
        if complete_sequences:
            best_sequence = min(complete_sequences, key=lambda state: (
                round(state[0], 6), action_meta[state[2]][0],
                action_meta[state[2]][1], action_meta[state[2]][2],
                action_meta[state[2]][3]
            ))
            score, temp, first_action, sequence = best_sequence
            prediction = pred[0][first_action]
            lat, power, energy, _ = self._robust(prediction, c)
            chosen_e2e = self._estimate_e2e_latency(
                first_action, 0, tuple(), lat, telemetry, c
            )
            tnext = self._temperature_next(telemetry.temperature_c, power, c) if thermal_mode else None
            self.previous_action = first_action
            return Decision(
                first_action, "MPC", "Lowest-energy feasible sequence over receding horizon",
                lat, power, energy, tnext, len(first_step_feasible),
                tuple(action.name for action in sequence), chosen_e2e
            )

        # All surviving states are at the same depth. Execute the first action
        # of the best-scoring partial sequence, then observe and replan.
        if beam and first_step_feasible and len(beam[0][3]) > 0:
            score, temp, first_action, sequence = beam[0]
            prediction = pred[0][first_action]
            lat, power, energy, _ = self._robust(prediction, c)
            chosen_e2e = self._estimate_e2e_latency(
                first_action, 0, tuple(), lat, telemetry, c
            )
            tnext = self._temperature_next(telemetry.temperature_c, power, c) if thermal_mode else None
            self.previous_action = first_action
            reason = (
                f"Future horizon infeasible after {len(sequence)} feasible step(s); "
                "executing first action and replanning."
            )
            return Decision(
                first_action, "PARTIAL_HORIZON_FEASIBLE", reason,
                lat, power, energy, tnext, len(first_step_feasible),
                tuple(action.name for action in sequence), chosen_e2e
            )

        # Select compliant fallback that satisfies active quality floor
        safe_fallback = self.fallback
        q_floor_0 = (c.quality_floor_horizon[0]
                     if c.quality_floor_horizon is not None else c.quality_floor)

        try:
            fb_pred = self.predictor.predict(self.fallback, workloads[0], telemetry)
            _, _, _, q_fb = self._robust(fb_pred, c)
            if q_fb is None or q_fb < q_floor_0:
                compliant_candidates = []
                for act in self.validated_actions:
                    try:
                        p_act = self.predictor.predict(act, workloads[0], telemetry)
                        l_a, p_a, e_a, q_a = self._robust(p_act, c)
                        if q_a is not None and q_a >= q_floor_0:
                            compliant_candidates.append((p_a <= c.power_cap_w, e_a, act))
                    except Exception:
                        continue
                if compliant_candidates:
                    compliant_candidates.sort(key=lambda x: (not x[0], x[1]))
                    safe_fallback = compliant_candidates[0][2]
        except Exception:
            pass

        self.previous_action = safe_fallback
        reasons_str = "; ".join(self.last_rejection_reasons) if self.last_rejection_reasons else "horizon infeasibility"
        if safe_fallback not in self.validated_actions:
            return Decision(
                safe_fallback, "INFEASIBLE_SAFE_STATE",
                f"No feasible configuration satisfies hard safety constraints ({reasons_str}). "
                f"Fallback '{safe_fallback.name}' is unvalidated!"
            )

        fallback_violations = []
        try:
            fallback_prediction = self.predictor.predict(safe_fallback, workloads[0], telemetry)
            lat_fb, power_fb, _, quality_fb = self._robust(fallback_prediction, c)
            e2e_fb = self._estimate_e2e_latency(
                safe_fallback, 0, (), lat_fb, telemetry, c
            )
            if e2e_fb > c.deadline_ms:
                fallback_violations.append(
                    f"Fallback E2E latency {e2e_fb:.1f}ms exceeds deadline {c.deadline_ms:.1f}ms"
                )
            if power_fb > c.power_cap_w:
                fallback_violations.append(
                    f"Fallback power {power_fb:.2f}W exceeds cap {c.power_cap_w:.2f}W"
                )
            q_floor_0 = (c.quality_floor_horizon[0]
                        if c.quality_floor_horizon is not None else c.quality_floor)
            if quality_fb is None:
                fallback_violations.append("Fallback quality is unvalidated (None)")
            elif quality_fb < q_floor_0:
                fallback_violations.append(
                    f"Fallback quality {quality_fb:.3f} below floor {q_floor_0:.3f}"
                )
            if thermal_mode:
                tnext_fb = self._temperature_next(telemetry.temperature_c, power_fb, c)
                if tnext_fb is not None and tnext_fb >= c.temperature_cap_c:
                    fallback_violations.append(
                        f"Fallback temperature {tnext_fb:.1f}C exceeds cap {c.temperature_cap_c:.1f}C"
                    )
        except Exception as exc:
            fallback_violations.append(f"Fallback prediction failed ({exc})")

        if fallback_violations:
            violation_text = "; ".join(fallback_violations)
            return Decision(
                safe_fallback, "INFEASIBLE_SAFE_STATE",
                f"No fully feasible predicted sequence across the complete horizon ({reasons_str}). "
                f"Fallback '{safe_fallback.name}' is also unsafe: [{violation_text}]."
            )
        return Decision(
            safe_fallback, "SAFETY_FALLBACK",
            f"No fully feasible predicted sequence across the complete horizon ({reasons_str}); "
            f"validated safe fallback '{safe_fallback.name}' selected."
        )

# Integration sketch: implement Predictor.predict() using the actual simulator or
# board telemetry/model. A driver should apply only Decision.action, then return
# real counters/sensors as the next Telemetry. Never send commands from this module.
