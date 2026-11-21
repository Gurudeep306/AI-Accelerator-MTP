"""Fixed and rule-based controllers (the two baselines of the pipeline)."""
from __future__ import annotations

from accel.model import ActionSpace
from config.params import Params


class Controller:
    name = "controller"

    def reset(self):
        pass

    def act(self, obs) -> int:
        raise NotImplementedError


class FixedController(Controller):
    """Holds one operating point (v1 default: f3 / FP16 / 20% sparsity, full network)."""
    name = "Fixed"

    def __init__(self, params: Params, f=3, prec="FP16", sp=0.2, exit_="final"):
        self.a = ActionSpace(params.accel).index(f, prec, sp, exit_)

    def act(self, obs):
        return self.a


class RuleBasedController(Controller):
    """v1 rule set extended with the Anytime-YOLO exit and robot-perception cues.

    Priority: thermal protection > imminent hazard > predicted hazard / dense
    crowd > moderate scene > energy saving.
    """
    name = "Rule-based"

    def __init__(self, params: Params):
        self.P = params
        s = ActionSpace(params.accel)
        self.thermal = s.index(2, "FP16", 0.4, "head2")
        self.critical = s.index(5, "FP16", 0.0, "final")
        self.high = s.index(4, "FP16", 0.2, "final")
        self.balanced = s.index(3, "FP16", 0.2, "final")
        self.eco = s.index(2, "INT8", 0.4, "head2")

    def act(self, obs):
        pc, a = obs.percept, self.P.accel
        if obs.temp >= a.temp_soft_limit - 5.0:
            return self.thermal
        if pc.hazard or pc.closing_speed > 1.5 and pc.clearance_m < 2 * self.P.robot.hazard_distance_m:
            return self.critical
        if pc.min_clearance_forecast < self.P.robot.hazard_distance_m or pc.n_view >= 8:
            return self.high
        if pc.n_view >= 3:
            return self.balanced
        return self.eco
