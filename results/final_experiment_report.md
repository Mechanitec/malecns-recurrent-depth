# MaleCNS recurrent-depth chess experiment

Status: **complete_bounded_protocol**

The frozen reference checkpoint uses the rate dynamics at recurrent depth 16. The low-Elo opponent scale comes from the measured calibration in `results/low_elo_calibration`, with Minic Level 0 and Gaia Skill Level 1 recorded as engine settings rather than guessed ratings.

## Current evidence

- Serious-run records: 400 of 400 expected; W/D/L = 0/400/0.
- Median Fly move latency: 2.644 seconds; recurrent passes: 25600.
- Depth sweep rows: 7 across depths 1, 2, 4, 8, 16, 32, and 64.
- Control sweep rows: 28 across the original, topology-shuffled, sign-shuffled, and recurrent-attenuated graphs.
- Every recorded game includes PGN, exact calibrated opponent setting, Fly move latency, recurrent-pass count, and candidate-score margin.

## Interpretation boundary

The current local runs use four tactical candidates and two-ply games in the long-run throughput allocation. The compact sweep uses the same bound and four-ply paired games. These artifacts verify the end-to-end real-connectome path and telemetry, but they are not a substitute for an opening-diverse, all-legal-move, long-game rating run. Expand those bounds before making a strong chess-strength or connectome-specific depth claim.

Key artifacts:

- `results/low_elo_calibration/low_elo_calibration.csv`
- `data/teacher_dataset_v1.csv`
- `data/chess_activations_depth16_v1.npz`
- `results/training/v1_depth16_rate_large/readout_checkpoint.npz`
- `results/depth_sweep/metrics.csv` and `results/depth_sweep/rating_vs_depth.png`
- `results/control_sweep/metrics.csv` and `results/control_sweep/control_depth_curves.png`
- `results\first_fly_rating\serious_400_bounded/games.csv` and `results\first_fly_rating\serious_400_bounded/games.pgn`
