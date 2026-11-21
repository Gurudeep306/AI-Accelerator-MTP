#!/usr/bin/env python3
"""Unit tests for workload input standardization and validation (Requirement 1)."""
import pytest
import math
from practical_mpc_controller import Action, Telemetry
from frame_preview_mpc import ProfilePerformancePredictor

def test_profile_performance_predictor_workload_formats(tmp_path):
    csv_file = tmp_path / "test_profiles.csv"
    csv_file.write_text(
        "width,exit,bits,latency_ms,energy_mj,power_w,validated_quality\n"
        "0.5,1,4,2.0,0.2,0.1,0.90\n"
    )
    predictor = ProfilePerformancePredictor(csv_file)
    action = Action(0.5, 1, 4)
    telem = Telemetry(temperature_c=40.0)

    # 1. Accepts float
    p_float = predictor.predict(action, 1.5, telem)
    assert math.isclose(p_float.latency_ms, 3.0)
    assert math.isclose(p_float.energy_mj, 0.3)
    assert math.isclose(p_float.power_w, 0.1)
    assert math.isclose(p_float.validated_quality, 0.90)

    # 2. Accepts int
    p_int = predictor.predict(action, 2, telem)
    assert math.isclose(p_int.latency_ms, 4.0)

    # 3. Accepts dict with workload_scale and optional std
    p_dict = predictor.predict(action, {"workload_scale": 1.5, "workload_scale_std": 0.1}, telem)
    assert math.isclose(p_dict.latency_ms, 3.0)
    assert p_dict.latency_std_ms > 0

    # 4. Invalid inputs raise appropriate exceptions
    with pytest.raises(ValueError, match="workload dict must contain 'workload_scale'"):
        predictor.predict(action, {"wrong_key": 1.0}, telem)

    with pytest.raises(TypeError, match="workload must be a numeric scale or a dict"):
        predictor.predict(action, "invalid_str", telem)

    with pytest.raises(ValueError, match="workload_scale must be finite and positive"):
        predictor.predict(action, -1.0, telem)

    with pytest.raises(ValueError, match="workload_scale must be finite and positive"):
        predictor.predict(action, 0.0, telem)

    with pytest.raises(ValueError, match="workload_scale must be finite and positive"):
        predictor.predict(action, float("nan"), telem)

    with pytest.raises(ValueError, match="workload_scale must be finite and positive"):
        predictor.predict(action, float("inf"), telem)
