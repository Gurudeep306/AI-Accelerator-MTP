#!/usr/bin/env python3
"""Standardized Workload Prediction Interface for Configurable DNN Accelerators.

Defines the common workload prediction representations and scopes for:
- Controller A (Historical observations)
- Controller B (Spatial preview classification)
- Controller C (Within-frame observation and remaining-work adaptation)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# Nominal R3 Conv2d reference MACs (640x384)
NOMINAL_R3_GMAC: float = 2.623196160

# Static Resolution Tier Definitions
TIER_GMAC_MAP = {
    0: {"name": "R1_320x192", "gmac": 0.655799040, "width": 320, "height": 192},
    1: {"name": "R2_480x288", "gmac": 1.475547840, "width": 480, "height": 288},
    2: {"name": "R3_640x384", "gmac": 2.623196160, "width": 640, "height": 384},
}


class PredictionScope(Enum):
    FULL_FRAME = "full_frame"             # Prediction covering entire upcoming frame (Controllers A, B)
    REMAINING_FRAME = "remaining_frame"   # Prediction covering remaining layers after Layer 0 (Controller C)
    HORIZON_STEP = "horizon_step"         # Forecast for future full frames (t+1..t+H-1)


@dataclass(frozen=True)
class WorkloadPrediction:
    """Standardized workload prediction across all stages of execution.

    Units:
        expected_gmac: Giga-MACs (10^9 multiply-accumulate operations).
        gmac_std: Standard deviation / uncertainty bound in GMACs.
        overhead_ms: CPU/GPU latency incurred to generate this prediction.
    """
    expected_gmac: float
    gmac_std: float = 0.0
    scope: PredictionScope = PredictionScope.FULL_FRAME
    overhead_ms: float = 0.0
    confidence: float = 1.0
    layer_boundary: int = 0
    tier_idx: Optional[int] = None
    tier_name: Optional[str] = None
    label_provenance: str = "ANALYTICAL_GRAPH"

    def __post_init__(self):
        if not math.isfinite(self.expected_gmac) or self.expected_gmac <= 0.0:
            raise ValueError(f"expected_gmac must be finite and positive, got {self.expected_gmac}")
        if not math.isfinite(self.gmac_std) or self.gmac_std < 0.0:
            raise ValueError(f"gmac_std must be finite and non-negative, got {self.gmac_std}")
        if self.overhead_ms < 0.0:
            raise ValueError(f"overhead_ms cannot be negative, got {self.overhead_ms}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    @property
    def workload_scale(self) -> float:
        """Workload normalized relative to nominal full R3 frame (2.6232 GMAC)."""
        return self.expected_gmac / NOMINAL_R3_GMAC

    @property
    def workload_scale_std(self) -> float:
        """Uncertainty normalized relative to nominal full R3 frame."""
        return self.gmac_std / NOMINAL_R3_GMAC

    @property
    def remaining_ratio(self) -> float:
        """Ratio of remaining workload.

        For FULL_FRAME and HORIZON_STEP, returns 1.0.
        For REMAINING_FRAME after YOLOv8n Layer 0 (1.0118% of MACs), returns 0.989882.
        """
        if self.scope in (PredictionScope.FULL_FRAME, PredictionScope.HORIZON_STEP):
            return 1.0
        # YOLOv8n Layer 0 accounts for exactly 1.0118% of Conv2d MACs
        return 1.0 - 0.010118

    def __float__(self) -> float:
        """Support float(prediction) returning workload_scale for backward compatibility."""
        return self.workload_scale

    def __getitem__(self, key: str) -> Any:
        """Support dict-like indexing for backward compatibility with legacy predictors."""
        if key == "workload_scale":
            return self.workload_scale
        elif key == "workload_scale_std":
            return self.workload_scale_std
        elif key == "expected_gmac":
            return self.expected_gmac
        elif key == "gmac_std":
            return self.gmac_std
        elif key == "scope":
            return self.scope.value
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in ("workload_scale", "workload_scale_std", "expected_gmac", "gmac_std", "scope")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_gmac": self.expected_gmac,
            "gmac_std": self.gmac_std,
            "workload_scale": self.workload_scale,
            "workload_scale_std": self.workload_scale_std,
            "scope": self.scope.value,
            "overhead_ms": self.overhead_ms,
            "confidence": self.confidence,
            "layer_boundary": self.layer_boundary,
            "tier_idx": self.tier_idx,
            "tier_name": self.tier_name,
            "label_provenance": self.label_provenance,
        }
