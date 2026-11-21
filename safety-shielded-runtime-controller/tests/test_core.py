"""Unit tests: run with  python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from accel.model import AcceleratorModel
from config.params import DEFAULT as P
from control.baselines import FixedController, RuleBasedController
from control.dqn import BranchingDuelingMLP, DQNAgent, featurise
from control.mpc import MPCController
from control.shield import Shield
from robot.safety import Percept, permitted_speed, stopping_distance, stopping_speed
from robot.scenes import ALL_SEQUENCES, equation_scene, load_eth_ucy
from sim.env import CrowdAcceleratorEnv
from sim.runner import run_episode, summarise


class TestAcceleratorModel(unittest.TestCase):
    def setUp(self):
        self.m = AcceleratorModel(P.accel, P.task)
        self.s = self.m.space

    def test_action_space_roundtrip(self):
        self.assertEqual(self.s.n, 135)
        a = self.s.index(4, "INT8", 0.2, "head2")
        self.assertEqual(self.s.describe(a), "f4/INT8/sp0.2/head2")

    def test_profile_anchor(self):
        # f_max / FP32 / dense / final  ==  measured GTX-1650 network median x edge slowdown + host time
        a = self.s.index(5, "FP32", 0.0, "final")
        lat, net = self.m.latency_ms(a)
        self.assertAlmostEqual(float(lat), 9.16825 * 1.5 + (12.24075 - 9.16825), places=3)

    def test_monotonicity(self):
        lo, hi = self.s.index(1, "FP16", 0.0, "final"), self.s.index(5, "FP16", 0.0, "final")
        self.assertGreater(self.m.latency_ms(lo)[0], self.m.latency_ms(hi)[0])
        self.assertLess(self.m.a_busy_w[lo], self.m.a_busy_w[hi])
        e1, e3 = self.s.index(3, "FP16", 0.0, "head1"), self.s.index(3, "FP16", 0.0, "final")
        self.assertLess(self.m.a_recall[e1], self.m.a_recall[e3])

    def test_thermal_steady_state(self):
        a, t = self.s.index(3, "FP16", 0.2, "final"), 40.0
        for _ in range(3000):
            out = self.m.step(a, t, 0.0, 0.4, 35.0)
            t = float(out["temp"])
        self.assertAlmostEqual(t, 35.0 + P.accel.thermal_resistance * float(out["power_w"]), places=2)

    def test_queue_saturates_when_too_slow(self):
        a = self.s.index(1, "FP32", 0.0, "final")          # ~53 ms > 33 ms frame interval
        q = 0.0
        for _ in range(20):
            out = self.m.step(a, 50.0, q, 0.4, 35.0)
            q = float(out["queue"])
        self.assertEqual(q, P.task.queue_capacity)
        self.assertGreater(float(out["drops"]), 0)

    def test_vectorised_step(self):
        out = self.m.step(np.arange(self.s.n), 50.0, 0.0, 0.4, 35.0)
        self.assertEqual(out["power_w"].shape, (self.s.n,))


class TestRobotSafety(unittest.TestCase):
    def test_stopping_speed_inverts_stopping_distance(self):
        v = float(stopping_speed(2.0, 0.3, P.robot))
        d = float(stopping_distance(v, 200.0, P.robot))   # tau = 0.2 + 0.1 actuation = 0.3
        self.assertAlmostEqual(d, 2.0 - P.robot.safety_buffer_m, places=6)

    def test_slower_detection_lowers_permitted_speed(self):
        pc = Percept(1.0, 0.0, 0.0, 1, True, False, 0.0, 1.0)
        fast, slow = permitted_speed(pc, np.array([150.0, 500.0]), P.robot)
        self.assertGreater(fast, slow)


class TestScenes(unittest.TestCase):
    def test_all_eth_ucy_sequences_load(self):
        for s in ALL_SEQUENCES:
            sc = load_eth_ucy(s)
            self.assertGreater(len(sc), 400)
            self.assertGreater(sc.n_pedestrians, 100)

    def test_equation_scene(self):
        sc = equation_scene(duration_s=60, seed=1)
        self.assertEqual(len(sc), 150)


class TestClosedLoop(unittest.TestCase):
    def test_controllers_run_and_shield_holds_thermal(self):
        sc = load_eth_ucy("crowds_zara01")
        for ctrl in (FixedController(P, 5, "FP32", 0.0, "final"), RuleBasedController(P)):
            env = CrowdAcceleratorEnv(sc, P, seed=0, ambient_c=45.0, max_steps=400)
            s = summarise(run_episode(env, ctrl, Shield(P)))
            self.assertEqual(s["throttle_s"], 0.0)      # shield prevents hardware throttling
        env = CrowdAcceleratorEnv(sc, P, seed=0, ambient_c=45.0, max_steps=400)
        s = summarise(run_episode(env, FixedController(P, 5, "FP32", 0.0, "final"), None))
        self.assertGreater(s["throttle_s"], 0.0)        # ...which does happen without it

    def test_mpc_short_run(self):
        sc = load_eth_ucy("biwi_hotel")
        env = CrowdAcceleratorEnv(sc, P, seed=0, max_steps=15)
        s = summarise(run_episode(env, MPCController(P, env=env), Shield(P)))
        self.assertEqual(s["steps"], 15)


class TestDQN(unittest.TestCase):
    def test_gradient_check(self):
        agent = DQNAgent(P)
        net = BranchingDuelingMLP(17, agent.space, hidden=16, seed=1)
        x = np.random.default_rng(0).standard_normal((3, 17)).astype(np.float64)
        for k in net.p:
            net.p[k] = net.p[k].astype(np.float64)
        dq = np.random.default_rng(1).standard_normal((3, agent.nA))
        q, cache = net.forward(x)
        g = net.backward(cache, dq)
        for k in ("W1", "Wf", "Wa", "bv"):
            idx = (0,) * net.p[k].ndim
            old = net.p[k][idx]
            net.p[k][idx] = old + 1e-6
            lp = (net.forward(x)[0] * dq).sum()
            net.p[k][idx] = old - 1e-6
            lm = (net.forward(x)[0] * dq).sum()
            net.p[k][idx] = old
            self.assertAlmostEqual(g[k][idx], (lp - lm) / 2e-6, places=4)

    def test_featurise_dim(self):
        env = CrowdAcceleratorEnv(load_eth_ucy("biwi_eth"), P)
        self.assertEqual(featurise(env.reset(), env.space, P).shape, (17,))


if __name__ == "__main__":
    unittest.main()
