#!/usr/bin/env python3
"""Audit tests for quality constraints, certification boundaries, fallback enforcement, and mode labeling (Section 7)."""
import pytest
from practical_mpc_controller import (
    Action, Constraints, MPCController, Telemetry, DEFAULT_ACTIONS
)
from frame_preview_mpc import ProfilePerformancePredictor
from history_mpc_controller import HistoryBasedMPC

def test_missing_quality_strictly_blocks_mpc_feasibility():
    """Verify that when raw navnet_source_profiles.csv has no validated quality,
    MPC strictly rejects all candidate sequences and triggers SAFETY_FALLBACK.
    Never treats unknown quality as feasible.
    """
    # Load raw profiles with allow_uncalibrated_quality=True (all quality=None)
    predictor = ProfilePerformancePredictor("navnet_source_profiles.csv", allow_uncalibrated_quality=True)
    fallback = Action(0.5, 1, 4)
    # Strict quality floor constraint
    c = Constraints(quality_floor=0.85, horizon=5)
    ctrl = MPCController(predictor, tuple(predictor.profiles.keys()), fallback=fallback, constraints=c)
    telem = Telemetry(temperature_c=35.0)

    decision = ctrl.choose(1.0, telem, [1.0] * 4)
    assert decision.action == fallback
    assert decision.mode == "INFEASIBLE_SAFE_STATE"
    assert "No fully feasible predicted sequence" in decision.reason
    assert decision.feasible_actions == 0

def test_unverified_fallback_is_not_counted_as_feasible_plan():
    """Verify that when fallback is triggered with unverified/impossible conditions,
    the controller reports INFEASIBLE_SAFE_STATE so the evaluation harness never marks it as a feasible plan.
    """
    predictor = ProfilePerformancePredictor("navnet_source_profiles.csv", allow_uncalibrated_quality=True)
    fallback = Action(0.5, 1, 4)
    # Impossible deadline = 0.1 ms
    c = Constraints(deadline_ms=0.1, horizon=5)
    ctrl = MPCController(predictor, tuple(predictor.profiles.keys()), fallback=fallback, constraints=c)
    telem = Telemetry(temperature_c=35.0)

    decision = ctrl.choose(1.0, telem, [1.0] * 4)
    assert decision.mode == "INFEASIBLE_SAFE_STATE"
    # Mode must clearly indicate failure, not MPC
    assert decision.mode != "MPC"
