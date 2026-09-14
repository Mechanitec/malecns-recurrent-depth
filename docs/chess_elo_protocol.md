# Chess rating experimental protocol

## Goal

Measure playing strength of a frozen MaleCNS-derived recurrent system only when the game protocol produces informative outcomes, and use position-quality metrics for rapid recurrent-depth research when full games are too expensive.

Chess rating is a benchmark of the **complete artificial system**: connectome + dynamics + encoder + projector + readout. It is not the biological fruit fly's Elo.

## Opponent ladder

The preferred high-throughput ladder is now:

- **Minic** for very weak measured/calibrated settings;
- **Gaia** for its low published strength anchors and calibration bridge;
- **Stockfish** at/above the installed build's supported `UCI_Elo` floor.

Legacy Alfil support may remain for compatibility, but it is not the default bulk-tournament engine because throughput was inadequate.

Low-strength Minic settings must not be assigned guessed Elo values. Use the measured calibration table in `results/low_elo_calibration/` and preserve the exact engine setting plus calibrated rating in every game record.

Stockfish's supported Elo range is detected from the local binary at runtime.

## Calibration convention

The weak-engine calibration uses actual games on a connected common scale. Preserve both raw calibration outputs and uncertainty.

Do not force two independent arbitrary anchors onto standard Elo. If a zero-based project display scale is useful, label it separately (for example `mcr0`) and do not call it official/FIDE Elo.

## Fly move selection

For each legal move:

1. encode board + candidate move;
2. project to the frozen sensory population;
3. run the same MaleCNS-derived recurrent system for depth `D`;
4. read a scalar score from the frozen readout population;
5. choose the highest-scoring legal move.

This all-legal candidate interface is required for headline strength and position-quality experiments.

## Frozen factors in a depth experiment

Freeze:

- MaleCNS graph/preprocessing;
- neural dynamics/gains;
- chess feature definition;
- sensory population;
- projector seed/fanout/amplitude;
- readout population/weights/checkpoint;
- opponent/evaluator binaries and settings;
- opening suite;
- random seeds where applicable;
- rating/adjudication policy.

Only recurrent depth changes.

## Short-game protocol warning

A game stopped after a tiny number of plies and then assigned `1/2-1/2` is **not an informative normal game result**.

The repository's earlier 2-ply/4-ply compact protocols were useful for plumbing/throughput diagnostics, but their repeated ~410/~445 rating values must not be interpreted as actual chess strength because unresolved games were overwhelmingly or entirely forced draws.

Do not repeat a large forced-draw tournament merely to reduce the numerical confidence interval around a non-informative outcome mechanism.

## Serious rating requirements

A serious rating run should use one of the following predeclared outcome mechanisms:

1. **Natural game completion** with a sufficiently large `max_plies`;
2. a validated objective adjudication rule based on an independent evaluator, with explicit win/draw thresholds and persistence criteria applied symmetrically;
3. another documented outcome rule demonstrated not to collapse most games to artificial 0.5 scores.

The independent evaluator must never choose moves or leak analysis into either player.

Use paired openings/color reversal and reuse the same opening suite for model/depth comparisons.

## Recommended game-count tiers

After the outcome protocol is informative:

- smoke: 20-40 games;
- development: 100-200 games;
- serious estimate: 400+ games;
- publication-quality comparisons: paired openings plus bootstrap/sequential-power analysis.

Game count does not compensate for a broken adjudication mechanism.

## Elo estimator

Given opponent ratings `R_i` and fly scores `s_i in {0, 0.5, 1}`, estimate `R` from:

```text
sum_i [s_i - E(R, R_i)] = 0
E(R, R_i) = 1 / (1 + 10^((R_i - R)/400))
```

For serious claims, bootstrap paired openings rather than treating color-swapped games as fully independent.

## Position-quality experiment is currently primary

Because full all-legal MaleCNS moves are computationally expensive, the current research priority is a paired held-out **position-depth study** rather than another short-game rating run.

Primary position metric:

```text
regret_cp = teacher_best_cp - teacher_cp(selected_move)
```

Use the exact same unseen positions across all depths and graph controls. This directly tests whether recurrent depth improves decisions without requiring thousands of completed games.

Once the recurrent-depth optimum and a stronger decoder are established, return to long-game rating with an informative completion/adjudication rule.

## Interpretation

The strongest eventual rating result is differential:

```text
same checkpoint + same openings + same opponents + same outcome policy
vary only recurrent depth
```

A connectome-specific claim additionally requires the real MaleCNS graph to show a stronger depth benefit than matched topology/sign/recurrent-strength controls.
