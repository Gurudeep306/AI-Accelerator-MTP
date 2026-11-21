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
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple
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
    beam_width: int = 128

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

    def _temperature_next(self, current_c: float, power_w: float) -> Optional[float]:
        if self.c.thermal_resistance_c_per_w is None:
            return None
        tau = max(self.c.thermal_tau_s, 1e-6)
        a = math.exp(-self.c.decision_period_s / tau)
        equilibrium = self.c.ambient_c + self.c.thermal_resistance_c_per_w * power_w
        return a * current_c + (1.0 - a) * equilibrium

    def _robust(self, p: Prediction):
        k = self.c.uncertainty_k
        return (p.latency_ms + k*p.latency_std_ms,
                p.power_w + k*p.power_std_w,
                p.energy_mj + k*p.energy_std_mj,
                None if p.validated_quality is None else p.validated_quality-k*p.quality_std)

    def _hard_override(self, t: Telemetry) -> Optional[str]:
        if t.watchdog_cycles is not None and t.elapsed_cycles is not None and t.elapsed_cycles >= t.watchdog_cycles:
            return "watchdog budget exceeded"
        if t.temperature_c >= self.c.temperature_cap_c:
            return "temperature cap reached"
        if t.observed_power_w is not None and t.observed_power_w >= self.c.power_cap_w:
            return "measured power cap reached"
        if t.quality_sensor_valid and t.runtime_quality is not None and t.runtime_quality < self.c.quality_floor:
            return "runtime quality monitor below floor"
        return None

    def _estimate_e2e_latency(self, action: Action, step: int, seq: Tuple[Action, ...],
                              robust_inf_lat: float, telemetry: Telemetry) -> float:
        """Estimate end-to-end latency accounting for host decision, preview, reconfig, and queues."""
        switch_lat = getattr(telemetry, 'switch_latency_ms', 0.0) or self.c.switch_latency_ms
        if step == 0:
            prev_act = self.previous_action
            reconfig_lat = switch_lat if (prev_act is not None and action != prev_act) else 0.0

            l1_frac = getattr(telemetry, 'layer1_mac_fraction', 0.0)
            if l1_frac > 0.0:
                l1_lat = getattr(telemetry, 'layer1_latency_ms', 0.0)
                accel_q = getattr(telemetry, 'accel_queue_wait_ms', 0.0)
                host_q = getattr(telemetry, 'host_queue_wait_ms', 0.0)
                est_dec = getattr(telemetry, 'estimated_host_decision_ms', 0.0)
                # Within-frame: Layer 1 runs on accelerator, host decides for Layer 2+, then reconfig + remaining layers
                t_host_ready = max(accel_q + l1_lat, host_q) + est_dec
                return t_host_ready + reconfig_lat + robust_inf_lat
            else:
                host_q = getattr(telemetry, 'host_queue_wait_ms', 0.0)
                prev_lat = getattr(telemetry, 'preview_latency_ms', 0.0)
                est_dec = getattr(telemetry, 'estimated_host_decision_ms', 0.0)
                accel_q = getattr(telemetry, 'accel_queue_wait_ms', 0.0)
                # Frame-level: dispatch occurs when decision is ready AND accelerator is idle
                t_host_ready = host_q + prev_lat + est_dec
                t_accel_ready = accel_q
                dispatch_delay = max(t_host_ready, t_accel_ready) + reconfig_lat
                return dispatch_delay + robust_inf_lat
        else:
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
               workload_forecast: Sequence[object] = ()) -> Decision:
        override = self._hard_override(telemetry)
        if override:
            self.previous_action = self.fallback
            return Decision(self.fallback, "SAFETY_FALLBACK", override)
        if not self.thermal_model_ready:
            thermal_mode = False
        else:
            thermal_mode = True

        workloads = [workload_now]
        if workload_forecast:
            workloads.extend(list(workload_forecast[:max(0, self.c.horizon-1)]))
        while len(workloads) < self.c.horizon:
            workloads.append(workloads[-1])

        # Predict for each action at each horizon step; no static/synthetic values assumed.
        pred: List[Dict[Action, Prediction]] = []
        for wl in workloads:
            row: Dict[Action, Prediction] = {}
            for action in self.validated_actions:
                try:
                    p = self.predictor.predict(action, wl, telemetry)
                    values = [p.latency_ms, p.power_w, p.energy_mj]
                    if any((not math.isfinite(v) or v < 0) for v in values):
                        continue
                    row[action] = p
                except (ValueError, RuntimeError, KeyError, NotImplementedError):
                    continue
            pred.append(row)

        # Beam search over feasible action sequences, preserving thermal state.
        # State: score, current temp, first action, sequence.
        beam = [(0.0, telemetry.temperature_c, None, tuple())]
        first_step_feasible = set()
        rejection_reasons = set()

        for step in range(self.c.horizon):
            next_beam = []
            for score, temp, first, seq in beam:
                for action, p in pred[step].items():
                    lat, power, energy, quality = self._robust(p)
                    # Check end-to-end latency constraint
                    e2e_lat = self._estimate_e2e_latency(action, step, seq, lat, telemetry)
                    if e2e_lat > self.c.deadline_ms:
                        rejection_reasons.add(f"e2e_latency ({e2e_lat:.1f}ms > {self.c.deadline_ms:.1f}ms)")
                        continue
                    if power > self.c.power_cap_w:
                        rejection_reasons.add(f"power ({power:.2f}W > {self.c.power_cap_w:.2f}W)")
                        continue
                    if quality is None:
                        rejection_reasons.add("quality_unvalidated")
                        continue
                    if quality < self.c.quality_floor:
                        rejection_reasons.add(f"quality ({quality:.3f} < {self.c.quality_floor:.3f})")
                        continue
                    tnext = self._temperature_next(temp, power) if thermal_mode else None
                    if thermal_mode and (tnext is None or tnext >= self.c.temperature_cap_c):
                        rejection_reasons.add(f"temperature ({tnext:.1f}C >= {self.c.temperature_cap_c:.1f}C)")
                        continue
                    if step == 0:
                        first_step_feasible.add(action)

                    # Main objective: total energy (inference + switching overhead)
                    step_score = energy
                    if self.c.latency_penalty > 0.0:
                        step_score += self.c.latency_penalty * (e2e_lat / self.c.deadline_ms) ** 2
                    if seq and seq[-1] != action:
                        step_score += self.c.switch_penalty_mj
                    elif not seq and self.previous_action is not None and self.previous_action != action:
                        step_score += self.c.switch_penalty_mj
                    next_beam.append((score + step_score, temp if tnext is None else tnext,
                                      action if first is None else first, seq + (action,)))
            if not next_beam:
                break
            next_beam.sort(key=lambda s: self._beam_sort_key(s, pred[0]))
            # Deduplicate same first action and temperature bucket for diversity.
            beam = next_beam[:max(1, self.c.beam_width)]

        # A partial sequence is not a valid horizon solution. If any horizon step
        # has no feasible continuation, fall back rather than silently accepting it.
        complete_sequences = [state for state in beam if len(state[3]) == self.c.horizon]
        if not complete_sequences or not first_step_feasible:
            self.previous_action = self.fallback
            reasons_str = "; ".join(sorted(rejection_reasons)) if rejection_reasons else "horizon infeasibility"

            # Check whether the fallback action itself is validated
            if self.fallback not in self.validated_actions:
                return Decision(
                    self.fallback, "INFEASIBLE_SAFE_STATE",
                    f"No feasible configuration satisfies hard safety constraints ({reasons_str}). "
                    f"Fallback '{self.fallback.name}' is unvalidated!"
                )

            fallback_violations = []
            try:
                p_fb = self.predictor.predict(self.fallback, workloads[0], telemetry)
                lat_fb, pwr_fb, _, qual_fb = self._robust(p_fb)
                e2e_fb = self._estimate_e2e_latency(self.fallback, 0, (), lat_fb, telemetry)
                if e2e_fb > self.c.deadline_ms:
                    fallback_violations.append(f"Fallback E2E latency {e2e_fb:.1f}ms exceeds deadline {self.c.deadline_ms:.1f}ms")
                if pwr_fb > self.c.power_cap_w:
                    fallback_violations.append(f"Fallback power {pwr_fb:.2f}W exceeds cap {self.c.power_cap_w:.2f}W")
                if qual_fb is None:
                    fallback_violations.append("Fallback quality is unvalidated (None)")
                elif qual_fb < self.c.quality_floor:
                    fallback_violations.append(f"Fallback quality {qual_fb:.3f} below floor {self.c.quality_floor:.3f}")
                if thermal_mode:
                    tnext_fb = self._temperature_next(telemetry.temperature_c, pwr_fb)
                    if tnext_fb is not None and tnext_fb >= self.c.temperature_cap_c:
                        fallback_violations.append(f"Fallback temperature {tnext_fb:.1f}C exceeds cap {self.c.temperature_cap_c:.1f}C")
            except Exception as e:
                fallback_violations.append(f"Fallback prediction failed ({e})")

            if fallback_violations:
                viol_str = "; ".join(fallback_violations)
                return Decision(
                    self.fallback, "INFEASIBLE_SAFE_STATE",
                    f"No fully feasible predicted sequence across the complete horizon ({reasons_str}). "
                    f"Fallback '{self.fallback.name}' is also unsafe: [{viol_str}]."
                )

            return Decision(
                self.fallback, "SAFETY_FALLBACK",
                f"No fully feasible predicted sequence across the complete horizon ({reasons_str}); "
                f"validated safe fallback '{self.fallback.name}' selected."
            )

        best_seq = min(complete_sequences, key=lambda s: self._beam_sort_key(s, pred[0]))
        score, temp, first_action, sequence = best_seq
        p = pred[0][first_action]
        lat, power, energy, _ = self._robust(p)
        chosen_e2e = self._estimate_e2e_latency(first_action, 0, tuple(), lat, telemetry)
        tnext = self._temperature_next(telemetry.temperature_c, power) if thermal_mode else None
        self.previous_action = first_action
        return Decision(first_action, "MPC", "Lowest-energy feasible sequence over receding horizon",
                        lat, power, energy, tnext, len(first_step_feasible), tuple(a.name for a in sequence),
                        chosen_e2e)

# Integration sketch: implement Predictor.predict() using the actual simulator or
# board telemetry/model. A driver should apply only Decision.action, then return
# real counters/sensors as the next Telemetry. Never send commands from this module.
