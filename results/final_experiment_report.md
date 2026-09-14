# MaleCNS recurrent-depth chess experiment

Status: **complete_short_game_protocol**

The frozen reference checkpoint uses the rate dynamics at recurrent depth 16. The low-Elo opponent scale comes from the measured calibration in `results/low_elo_calibration`, with Minic Level 0 and Gaia Skill Level 1 recorded as engine settings rather than guessed ratings.

## Current evidence

- Serious-run records: 402 of 402 expected; W/D/L = 0/402/0.
- Full-legal diagnostic records: 60.
- Median Fly move latency: 12.808 seconds; recurrent passes: 175312.
- Depth sweep rows: 7 across depths 1, 2, 4, 8, 16, 32, and 64.
- Control sweep rows: 28 across the original, topology-shuffled, sign-shuffled, and recurrent-attenuated graphs.
- Control rating rows: 28 across the same graph variants and depths.
- Every recorded game includes PGN, exact calibrated opponent setting, Fly move latency, recurrent-pass count, and candidate-score margin.

## Interpretation boundary

The serious rating allocation uses the configured candidate policy and short games per prepared opening. The diagnostic and position-quality sweeps score all legal moves and use the calibrated opening set. These artifacts verify the end-to-end real-connectome path and telemetry, but a short-game rating is not a substitute for an opening-diverse, long-game strength estimate.

Key artifacts:

- `results/low_elo_calibration/low_elo_calibration.csv`
- `data/teacher_dataset_v1.csv`
- `data/chess_activations_depth16_v1.npz`
- `results/training/v1_depth16_rate_large/readout_checkpoint.npz`
- `results\sweep_full_8pos_v2/depth_sweep/metrics.csv` and `results\sweep_full_8pos_v2/depth_sweep/rating_vs_depth.png`
- `results\sweep_full_8pos_v2/control_sweep/metrics.csv` and `results\sweep_full_8pos_v2/control_sweep/control_depth_curves.png`
- `results\first_fly_rating\serious_402_fulllegal/games.csv` and `results\first_fly_rating\serious_402_fulllegal/games.pgn`
