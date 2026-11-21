"""All tunable parameters in one place, with their provenance.

Tags used below:
  [measured]   taken from a measurement file shipped with the project
  [derived]    computed from measured numbers (derivation in the comment)
  [assumption] a modelling choice; change it and re-run to test sensitivity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class AcceleratorParams:
    # ---------------- action space -----------------------------------------
    # DVFS ladder [assumption]: shaped like an embedded-GPU ladder (~0.3-1.1 GHz).
    freq_ghz: Tuple[float, ...] = (0.31, 0.51, 0.71, 0.92, 1.12)
    volt: Tuple[float, ...] = (0.62, 0.71, 0.80, 0.90, 1.00)
    precisions: Tuple[str, ...] = ("FP32", "FP16", "INT8")
    sparsities: Tuple[float, ...] = (0.0, 0.2, 0.4)
    exits: Tuple[str, ...] = ("head1", "head2", "final")   # Anytime-YOLO exits [measured]

    # ---------------- latency ----------------------------------------------
    # Network time on this edge accelerator at f_max/FP32 = GTX-1650 network
    # median x edge_slowdown [assumption]. Host pre/post-processing (incl. NMS)
    # = image-to-box median - network median [derived from profiles].
    edge_slowdown: float = 1.5
    # FP16/INT8 network-time factors [derived]: YOLO11n TensorRT on T4,
    # end-to-end 14.47/12.97/10.49 ms minus ~3 ms fixed overhead.
    precision_latency: Tuple[float, ...] = (1.00, 0.87, 0.65)
    # Structured-sparsity speed-up [assumption].
    sparsity_latency: Tuple[float, ...] = (1.00, 0.92, 0.84)
    nms_ms_per_object: float = 0.04            # [assumption] post-processing grows with objects
    spike_probability: float = 0.02            # [assumption] OS/memory contention spike
    spike_factor: float = 1.8
    # Reconfiguration overhead added to the first frame after a switch [assumption]:
    # DVFS step (PLL/voltage settle) vs precision/sparsity change (swap to a different
    # pre-built engine). Changing the Anytime-YOLO exit is free (same engine).
    dvfs_switch_ms: float = 1.0
    engine_swap_ms: float = 12.0

    # ---------------- accuracy ---------------------------------------------
    # mAP@50 / recall per exit come from the profile JSON [measured].
    # INT8 mAP ratio 0.1399/0.1484 = 0.943 [measured, YOLO11n T4]; FP16 ~ lossless [assumption].
    precision_accuracy: Tuple[float, ...] = (1.000, 0.995, 0.943)
    sparsity_accuracy: Tuple[float, ...] = (1.00, 0.97, 0.92)   # [assumption]
    crowd_occlusion_penalty: float = 0.30      # recall loss at max crowding [assumption]

    # ---------------- power --------------------------------------------------
    idle_power_w: float = 2.0                  # [assumption]
    dyn_power_max_w: float = 22.0              # busy power at f_max, V_max, FP32 [assumption]
    # Board-power ratios FP16/INT8 vs FP32: 37.7/40.0, 34.8/40.0 [measured, T4]
    precision_power: Tuple[float, ...] = (1.000, 0.942, 0.870)
    sparsity_power: Tuple[float, ...] = (1.00, 0.96, 0.92)       # [assumption]
    leak_power_25c_w: float = 1.2              # [assumption]
    leak_temp_coeff: float = 0.025             # exp. leakage-temperature coupling [assumption]

    # ---------------- thermal (1st-order RC, exact discretisation) ---------
    thermal_resistance: float = 3.2            # degC/W [assumption]
    thermal_capacitance: float = 25.0          # J/degC -> tau = 80 s [assumption]
    temp_soft_limit: float = 80.0              # controller constraint
    temp_throttle: float = 85.0                # hardware emergency throttle (forces f_min)
    temp_release: float = 78.0                 # throttle released below this


@dataclass(frozen=True)
class RobotParams:
    nominal_speed: float = 1.2                 # m/s, typical service robot in crowds
    max_brake: float = 1.5                     # m/s^2
    max_accel: float = 1.0                     # m/s^2
    actuation_delay_s: float = 0.10            # control + actuation latency
    body_clearance_m: float = 0.60             # robot radius + pedestrian radius
    safety_buffer_m: float = 0.20
    corridor_half_width_m: float = 0.7         # robot radius + pedestrian radius + 0.1 m
    sensing_range_m: float = 8.0
    density_radius_m: float = 2.5
    hazard_distance_m: float = 3.0             # in-path pedestrian closer than this = hazard
    # Crowd-density speed cap: linear from v_max to floor at density_ref [policy assumption, kept from v1]
    crowd_density_ref: float = 1.5
    crowd_speed_floor: float = 0.25
    zone_speed_limit: float = 1.5


@dataclass(frozen=True)
class TaskParams:
    camera_fps: float = 30.0
    deadline_ms: float = 50.0
    queue_capacity: int = 4                    # frames; older frames are dropped
    max_miss_probability: float = 0.05         # shield: allowed P(pedestrian missed in a step)
    # Tracker confirmation: a new pedestrian is acted on after `confirm_hits`
    # detections (SORT/ByteTrack-style min-hits) [assumption]. Expected
    # time-to-detect = latency + queue wait + (confirm_hits/recall - 1)/fps.
    confirm_hits: int = 2


@dataclass(frozen=True)
class CostWeights:
    """Shared stage cost. MPC minimises it, the RL agent receives -cost as reward,
    and every controller is reported on each term separately."""
    progress: float = 1.0
    energy_per_w: float = 0.04
    hazard_miss: float = 20.0
    safety_violation: float = 10.0             # latency-attributable speed-permit violation
    collision: float = 0.0                     # frontal contacts are reported, not optimised:
                                               # in a non-reactive replay they are crowd-induced
    deadline_miss: float = 2.0
    thermal_over_soft: float = 0.4             # per degC above the soft limit
    throttle: float = 5.0
    switch: float = 0.05
    frame_drop: float = 0.5                    # x fraction of frames dropped
    shield_intervention: float = 0.5           # RL training only: penalty when shield overrides


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.4                            # ETH/UCY annotation period (2.5 Hz)
    ambient_c: float = 35.0                    # robot electronics bay
    initial_temp_c: float = 50.0


@dataclass(frozen=True)
class Params:
    accel: AcceleratorParams = field(default_factory=AcceleratorParams)
    robot: RobotParams = field(default_factory=RobotParams)
    task: TaskParams = field(default_factory=TaskParams)
    cost: CostWeights = field(default_factory=CostWeights)
    sim: SimParams = field(default_factory=SimParams)


DEFAULT = Params()
