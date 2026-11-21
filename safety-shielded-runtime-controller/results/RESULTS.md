# Results

All numbers are from the closed-loop simulator replaying the held-out ETH/UCY crowds (6 sequences, leave-one-scene-out for the RL agent). Accelerator power, temperature and latency are model estimates anchored to the measured Anytime-YOLO / YOLO11n profiles; they are not hardware measurements.

## Main comparison — nominal (35 C) (18 episodes per controller)

| Controller | Cost ↓ | Energy J/m ↓ | Power W ↓ | Speed m/s ↑ | Hazard t-detect ms ↓ | Hazard miss % ↓ | Deadline miss % ↓ | Max T °C ↓ | Throttled % ↓ | Violations /km ↓ | Switches /min | Shield interv. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | -0.228 | 10.64 | 9.19 | 0.863 | 214 | 1.46 | 0.02 | 65.6 | 0.00 | 0.16 | 12.8 | 7.2 |
| Rule-based | -0.225 | 11.34 | 9.81 | 0.865 | 199 | 1.19 | 0.02 | 75.1 | 0.00 | 0.00 | 20.6 | 0.1 |
| MPC | -0.334 | 7.20 | 6.24 | 0.866 | 219 | 1.57 | 0.15 | 58.1 | 0.00 | 0.16 | 20.7 | 0.0 |
| RL (DQfD) | -0.336 | 7.20 | 6.25 | 0.868 | 219 | 1.58 | 0.14 | 58.3 | 0.00 | 0.16 | 20.5 | 0.0 |

## Main comparison — hot (45 C) (18 episodes per controller)

| Controller | Cost ↓ | Energy J/m ↓ | Power W ↓ | Speed m/s ↑ | Hazard t-detect ms ↓ | Hazard miss % ↓ | Deadline miss % ↓ | Max T °C ↓ | Throttled % ↓ | Violations /km ↓ | Switches /min | Shield interv. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | -0.190 | 11.72 | 10.12 | 0.863 | 214 | 1.46 | 0.02 | 80.0 | 0.00 | 0.16 | 13.0 | 7.3 |
| Rule-based | -0.190 | 11.06 | 9.57 | 0.865 | 216 | 1.67 | 0.07 | 75.2 | 0.00 | 0.00 | 34.7 | 2.2 |
| MPC | -0.306 | 8.01 | 6.94 | 0.866 | 219 | 1.57 | 0.15 | 71.1 | 0.00 | 0.16 | 20.8 | 0.0 |
| RL (DQfD) | -0.305 | 8.04 | 6.98 | 0.868 | 219 | 1.59 | 0.14 | 71.1 | 0.00 | 0.16 | 21.3 | 0.0 |

## Paired improvement vs Fixed (mean over episodes, bootstrap 95% CI)

| Controller | Condition | Δ cost (neg = better) | Δ energy % | Δ hazard time-to-detect % |
|---|---|---|---|---|
| Rule-based | nominal (35 C) | +0.022 [-0.005, +0.047] | +13.9 [+5.1, +22.6] | -7.5 [-7.9, -7.0] |
| Rule-based | hot (45 C) | +0.022 [-0.006, +0.050] | -0.4 [-5.7, +5.1] | -0.0 [-1.4, +1.5] |
| MPC | nominal (35 C) | -0.094 [-0.111, -0.078] | -29.5 [-32.6, -26.4] | +2.2 [+1.9, +2.6] |
| MPC | hot (45 C) | -0.102 [-0.120, -0.083] | -28.8 [-31.9, -25.8] | +2.2 [+1.9, +2.6] |
| RL (DQfD) | nominal (35 C) | -0.095 [-0.112, -0.078] | -29.4 [-32.5, -26.4] | +2.0 [+1.7, +2.4] |
| RL (DQfD) | hot (45 C) | -0.101 [-0.119, -0.082] | -28.5 [-31.5, -25.7] | +2.0 [+1.5, +2.5] |

## Mean stage cost per held-out sequence (nominal / hot)

| Sequence | Fixed | Rule-based | MPC | RL (DQfD) |
|---|---|---|---|---|
| biwi_eth | -0.463 / -0.424 | -0.518 / -0.476 | -0.607 / -0.579 | -0.606 / -0.576 |
| biwi_hotel | -0.401 / -0.356 | -0.435 / -0.399 | -0.526 / -0.494 | -0.533 / -0.500 |
| crowds_zara01 | -0.381 / -0.347 | -0.389 / -0.356 | -0.495 / -0.468 | -0.493 / -0.468 |
| crowds_zara02 | +0.071 / +0.108 | +0.136 / +0.169 | -0.008 / +0.022 | -0.004 / +0.035 |
| students001 | +0.366 / +0.388 | +0.454 / +0.487 | +0.323 / +0.342 | +0.321 / +0.340 |
| students003 | +0.047 / +0.071 | +0.120 / +0.149 | -0.015 / +0.006 | -0.017 / +0.005 |

## Synthetic equation-based stress scene (never used for training)

### nominal (35 C)

| Controller | Cost ↓ | Energy J/m ↓ | Power W ↓ | Speed m/s ↑ | Hazard t-detect ms ↓ | Hazard miss % ↓ | Deadline miss % ↓ | Max T °C ↓ | Throttled % ↓ | Violations /km ↓ | Switches /min | Shield interv. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | -0.274 | 9.80 | 9.02 | 0.920 | 217 | 1.49 | 0.13 | 64.2 | 0.00 | 0.00 | 20.4 | 11.9 |
| Rule-based | -0.281 | 10.66 | 9.81 | 0.921 | 197 | 1.14 | 0.00 | 70.7 | 0.00 | 0.00 | 23.8 | 0.0 |
| MPC | -0.393 | 6.77 | 6.24 | 0.920 | 217 | 1.54 | 0.04 | 56.3 | 0.00 | 0.00 | 28.0 | 0.0 |
| RL (DQfD) | -0.390 | 6.83 | 6.29 | 0.920 | 217 | 1.54 | 0.09 | 56.6 | 0.00 | 0.00 | 27.6 | 0.0 |

### hot (45 C)

| Controller | Cost ↓ | Energy J/m ↓ | Power W ↓ | Speed m/s ↑ | Hazard t-detect ms ↓ | Hazard miss % ↓ | Deadline miss % ↓ | Max T °C ↓ | Throttled % ↓ | Violations /km ↓ | Switches /min | Shield interv. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | -0.243 | 10.63 | 9.79 | 0.920 | 217 | 1.49 | 0.13 | 77.2 | 0.00 | 0.00 | 20.4 | 11.9 |
| Rule-based | -0.248 | 10.23 | 9.42 | 0.920 | 217 | 1.66 | 0.18 | 75.2 | 0.00 | 0.00 | 38.6 | 4.3 |
| MPC | -0.369 | 7.43 | 6.84 | 0.920 | 218 | 1.54 | 0.04 | 68.6 | 0.00 | 0.00 | 28.2 | 0.0 |
| RL (DQfD) | -0.367 | 7.50 | 6.90 | 0.920 | 217 | 1.53 | 0.04 | 68.9 | 0.00 | 0.00 | 27.5 | 0.0 |


## Ablations (seed 0, held-out ETH/UCY)

`[no shield]` = proposals applied without the safety layer; `MPC-oracle` = MPC with the true future crowd instead of a constant-velocity forecast.

### nominal (35 C)

| Controller | Cost ↓ | Energy J/m ↓ | Power W ↓ | Speed m/s ↑ | Hazard t-detect ms ↓ | Hazard miss % ↓ | Deadline miss % ↓ | Max T °C ↓ | Throttled % ↓ | Violations /km ↓ | Switches /min | Shield interv. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | -0.225 | 10.68 | 9.22 | 0.863 | 215 | 1.46 | 0.00 | 65.5 | 0.00 | 0.49 | 12.8 | 7.2 |
| Rule-based | -0.223 | 11.38 | 9.84 | 0.865 | 199 | 1.19 | 0.03 | 75.1 | 0.00 | 0.00 | 20.8 | 0.1 |
| MPC | -0.329 | 7.21 | 6.25 | 0.866 | 220 | 1.58 | 0.27 | 58.1 | 0.00 | 0.49 | 20.7 | 0.0 |
| RL (DQfD) | -0.329 | 7.24 | 6.27 | 0.866 | 220 | 1.59 | 0.24 | 58.2 | 0.00 | 0.49 | 20.5 | 0.0 |
| Fixed [no shield] | -0.229 | 10.78 | 9.34 | 0.866 | 212 | 1.39 | 0.00 | 65.8 | 0.00 | 1.96 | 0.0 | 0.0 |
| MPC-oracle | -0.338 | 7.14 | 6.23 | 0.873 | 221 | 1.59 | 0.25 | 58.1 | 0.00 | 0.49 | 20.1 | 0.0 |
| RL (DQN, scratch) | -0.221 | 8.59 | 7.49 | 0.873 | 235 | 1.89 | 0.90 | 61.9 | 0.00 | 0.49 | 111.6 | 0.0 |
| RL (DQfD, unmasked) [no shield] | +0.277 | 6.96 | 5.91 | 0.849 | 381 | 5.99 | 9.57 | 55.1 | 0.00 | 5.99 | 24.0 | 0.0 |
| Rule-based [no shield] | -0.223 | 11.38 | 9.84 | 0.865 | 199 | 1.19 | 0.03 | 75.1 | 0.00 | 0.00 | 20.9 | 0.0 |

### hot (45 C)

| Controller | Cost ↓ | Energy J/m ↓ | Power W ↓ | Speed m/s ↑ | Hazard t-detect ms ↓ | Hazard miss % ↓ | Deadline miss % ↓ | Max T °C ↓ | Throttled % ↓ | Violations /km ↓ | Switches /min | Shield interv. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | -0.188 | 11.76 | 10.15 | 0.863 | 215 | 1.46 | 0.00 | 80.0 | 0.00 | 0.49 | 13.0 | 7.3 |
| Rule-based | -0.189 | 11.07 | 9.57 | 0.865 | 217 | 1.67 | 0.12 | 75.2 | 0.00 | 0.00 | 34.1 | 2.1 |
| MPC | -0.301 | 8.02 | 6.95 | 0.866 | 220 | 1.58 | 0.27 | 71.1 | 0.00 | 0.49 | 20.8 | 0.0 |
| RL (DQfD) | -0.297 | 8.08 | 7.00 | 0.866 | 220 | 1.61 | 0.25 | 70.9 | 0.00 | 0.49 | 21.9 | 0.0 |
| Fixed [no shield] | -0.183 | 11.88 | 10.28 | 0.866 | 212 | 1.39 | 0.00 | 80.4 | 0.00 | 1.96 | 0.0 | 0.0 |
| MPC-oracle | -0.310 | 7.94 | 6.93 | 0.873 | 221 | 1.59 | 0.25 | 71.1 | 0.00 | 0.49 | 20.1 | 0.0 |
| RL (DQN, scratch) | -0.158 | 10.30 | 8.89 | 0.863 | 233 | 1.85 | 0.93 | 78.6 | 0.00 | 0.00 | 106.3 | 0.0 |
| RL (DQfD, unmasked) [no shield] | +0.369 | 7.70 | 6.49 | 0.842 | 393 | 6.83 | 9.39 | 67.4 | 0.00 | 7.54 | 18.4 | 0.0 |
| Rule-based [no shield] | -0.191 | 11.07 | 9.57 | 0.865 | 216 | 1.65 | 0.05 | 75.2 | 0.00 | 0.49 | 30.9 | 0.0 |


## Best static operating point (chosen in hindsight on the test scenes)

- nominal (35 C): `f2/INT8/sp0.0/head2` → cost -0.288, energy 7.58 J/m, hazard t-detect 226 ms, throttled 0.0%
- hot (45 C): `f2/INT8/sp0.0/head2` → cost -0.259, energy 8.41 J/m, hazard t-detect 226 ms, throttled 0.0%
