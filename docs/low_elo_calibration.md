# Low-Elo calibration

The project uses a project-specific **MaleCNS Chess Rating (MCR)** below the native Stockfish range. The intended empirical anchors are:

- `minic_0` = **0 MCR** (random legal mover)
- `gaia_580` = **580 MCR** (Gaia Chess 4 Skill Level 1, published estimate 580)

Do not assign invented Elo values to Minic levels 1–30. Their positions on the scale are fitted from games.

## Match design

`scripts/calibrate_low_elo.py schedule` creates a connected schedule with:

1. every adjacent Minic level (`0 vs 1`, `1 vs 2`, ..., `29 vs 30`),
2. five-level skip links (`0 vs 5`, `5 vs 10`, ...),
3. Gaia-580 anchor games against Minic levels 20, 22, 24, 26, 28 and 30.

The default is 40 opening pairs per matchup. Each pair contains two games with colors reversed. The default schedule is therefore **42 matchups × 40 pairs × 2 = 3,360 games**.

Completed game rows must contain at least `white`, `black`, `white_score` (`1`, `0.5`, or `0`), and preferably `pair_id` for pair-bootstrap confidence intervals.

## Fit

The fitter uses a Bradley–Terry logistic model with an independently fitted White-advantage term. Latent strengths are affinely normalized so that Minic 0 is exactly 0 and Gaia Level 1 is exactly 580.

Confidence intervals are obtained by bootstrap-resampling whole color-swapped opening pairs, not individual games. This preserves within-opening dependence.

Measured Minic values can contain small sampling inversions. For the runtime difficulty lookup, weighted pooled-adjacent-violators isotonic regression produces a monotonic Minic level table while retaining the unsmoothed `raw_mcr` column for scientific inspection.

## Commands

```bash
python scripts/calibrate_low_elo.py schedule \
  --openings 40 \
  --output results/low_elo/calibration_schedule.csv

# Execute the games. A FEN file is strongly recommended so every matchup
# sees the same color-swapped opening positions. The runner checkpoints after
# every game and resumes by game_index.
python scripts/calibrate_low_elo.py run \
  --schedule results/low_elo/calibration_schedule.csv \
  --minic /path/to/minic \
  --gaia /path/to/gaiachess \
  --openings-fen data/calibration_openings.fen \
  --output results/low_elo/calibration_games.csv

python scripts/calibrate_low_elo.py fit \
  --games results/low_elo/calibration_games.csv \
  --bootstrap 500 \
  --output results/low_elo/low_elo_calibration.csv
```

## Reproducibility requirements

Freeze engine versions, use one thread, disable pondering and Gaia opening-book variation, use a fixed opening suite, and preserve the same strength-mode settings throughout the calibration. Gaia's published 580 value is an estimated external anchor, not an absolute physical rating.

## Synthetic validation

The statistical code should be tested with simulated match outcomes whose latent ratings are known before empirical engine results are trusted. Synthetic outputs are test artifacts only and must never be committed or presented as measured engine calibration.

The `run` subcommand requires the `chess` optional dependency plus local Minic and Gaia executables. The schedule generator and statistical fitter do not require external engine binaries.
