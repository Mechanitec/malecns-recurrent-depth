# Fast low-Elo opponent ladder

Alfil is retained only as a legacy/cross-check engine. It is not the default for high-volume tournaments because its per-move latency is too high for the number of games needed to estimate MaleCNS Elo precisely.

## Default ladder

| Nominal rating | Opponent | Setting |
| ---: | --- | --- |
| 0 | Minic | `Level=0` random mover |
| 580 | Gaia 4 | `Skill Level=1` |
| 700 | Gaia 4 | `Skill Level=2` |
| 820 | Gaia 4 | `Skill Level=3` |
| 940 | Gaia 4 | `Skill Level=4` |
| 1060 | Gaia 4 | `Skill Level=5` |
| 1180 | Gaia 4 | `Skill Level=6` |
| 1300 | Gaia 4 | `Skill Level=7` |
| Stockfish floor+ | Stockfish | `UCI_LimitStrength` + `UCI_Elo` |

Gaia's ratings are published estimates, not exact physical constants. The benchmark records the engine name and nominal rating for every game so later cross-calibration can adjust the common rating scale without losing provenance.

Minic levels 1-30 are intentionally **not** assigned invented Elo labels here. The Minic project itself describes its low-level Elo fit as approximate. A future calibration run can estimate those levels against Gaia/Stockfish anchors and then fill the 0-580 gap densely.

## Why this is faster

- Minic `Level=0` returns a random legal move instead of performing a normal search.
- Gaia levels 1-19 are strength-capped by search/evaluation behavior rather than requiring a long wall-clock thinking time.
- Stockfish remains the high-rating reference.
- The independent full-strength Stockfish observer can still evaluate every live position; it is separate from the playing opponent.

## Example

```bash
python scripts/run_chess_benchmark.py \
  --minic C:/Engines/Minic/minic.exe \
  --gaia C:/Engines/Gaia/gaiachess.exe \
  --stockfish C:/Engines/Stockfish/stockfish.exe \
  --elos 0,580,700,820,940,1060,1180,1300,1320,1400 \
  --games-per-elo 100 \
  --output results/fly_fast_ladder
```

For an existing trained fly, add the normal `--checkpoint`, MaleCNS Feather paths, and `--sensory-indices` arguments.

## Legacy Alfil

The old Alfil path is still available for validation:

```bash
python scripts/run_chess_benchmark.py \
  --legacy-alfil \
  --alfil C:/Engines/Alfil/Alfil.exe \
  --stockfish C:/Engines/Stockfish/stockfish.exe \
  --elos 0,200,400,600,800,1000,1200,1320
```

Do not mix fast-ladder nominal ratings and Alfil nominal ratings in one Elo report without explicit cross-calibration.
