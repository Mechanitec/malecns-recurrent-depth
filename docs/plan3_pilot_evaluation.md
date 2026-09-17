# Plan 3 pilot evaluation and rerun requirements

## Status

The first completed Plan 3 run is retained as a historical exploratory pilot. It is useful for validating the implementation and for identifying failure modes, but it is **not** sufficient for a positive chess-learning claim.

The primary reward/aversive run used the real MaleCNS graph, D8 recurrent processing, a frozen linear decoder, and 11,119 existing KC-to-MBON plastic edges. It completed 803 updates while preserving graph topology and synaptic signs.

## What the pilot showed

Relative to the frozen-synapse control at D8:

| stage | reward/aversive top-1 | frozen top-1 | difference |
|---|---:|---:|---:|
| movement | 41.7% | 16.7% | +25.0 pp |
| endgames | 18.8% | 15.6% | +3.1 pp |
| tactics | 22.0% | 28.0% | -6.0 pp |
| mates | 23.7% | 23.7% | 0.0 pp |

The sequential positive-mate evaluation solved/preserved only 1 of 9 tested sequences (11.1%). The pilot therefore supports, at most, a narrow simple-task movement signal. It does not demonstrate general chess transfer or meaningful forced-mate skill.

Biologically, the run stayed well bounded. Topology and signs remained fixed, only the predeclared KC-to-MBON edges were modified, and almost no plastic edges reached the 0.5x/2.0x weight bounds. This is evidence that the local-plasticity machinery can change network behavior without globally rewriting the connectome.

## Problems found during post-run audit

### 1. Movement teaching signal did not match the predeclared protocol

`plan3.md` specifies a direct legality target for Stage 1:

```text
legal candidate   -> +1
illegal candidate -> -1
```

The pilot instead reused the centipawn-regret mapping. With synthetic movement labels of +100/-100 cp, an illegal move produced a raw signal of about +0.076 rather than -1. The baseline could later make the centered advantage negative, but this was not the planned Stage-1 experiment.

**Fixed on the PR branch:** movement lessons now use the direct `is_positive` target. Replayed movement lessons also retain the movement-specific rule even when replayed during a later curriculum stage.

### 2. The shuffled control was not causally matched

The original shuffled control used only 404 updates and trained a different decoder checkpoint from the primary/frozen runs. It also used a rolling signal buffer rather than an exact within-stage permutation. Consequently, its strong movement/mate numbers cannot be interpreted as evidence against the biological teaching signal.

**Fixed on the PR branch:**

- frozen and shuffled controls must provide `--decoder-checkpoint` and therefore reuse the exact primary decoder;
- shuffled controls must provide `--shuffled-signal-reference-log` from the primary run;
- the exact primary centered advantages are deterministically permuted within each training stage;
- update counts must match exactly or the shuffled run aborts;
- replay and signal shuffling use separate RNG streams so the replay examples remain matched across conditions.

The next valid shuffled control should use the full matched curriculum rather than the earlier half-size exploratory shortcut.

### 3. The reported pairwise metric was incorrect

`pairwise_ranking_accuracy_d8` was populated with top-1 accuracy instead of an actual pairwise concordance calculation.

**Fixed on the PR branch:** validation now computes all non-tied teacher-ordered candidate pairs. Teacher ties are ignored and predicted ties count as half correct.

### 4. Retention-drop sign was reversed

The pilot field `drop_from_own_stage` was computed as retained minus original, so a positive value could mean an improvement despite being named a drop.

**Fixed on the PR branch:** it is now `own_stage_top1 - retained_top1`, so positive values mean forgetting.

### 5. Mate coverage is too small for a strong conclusion

The final sequential validation contained only nine positive mate sequences (six mate-in-1 and three mate-in-2) and no mate-in-3 sequence. The existing result must remain descriptive only.

Before a strong mate-learning claim, the rerun must report explicit coverage counts for mate-in-1, mate-in-2, and mate-in-3 and perform full-sequence evaluation against best defense.

## Plan 3.1 rerun protocol

The next experiment should preserve the locked research goal and the same biologically localized KC-to-MBON plasticity concept, but use the corrected protocol.

1. Run the **primary reward/aversive** condition first and save its frozen decoder plus `training_log.csv`.
2. Run the **frozen** condition using the exact primary decoder checkpoint.
3. Run the **shuffled** condition using the exact primary decoder checkpoint and the primary training log as the signal reference.
4. Use identical curriculum limits, seed, replay policy, graph, manifests, Stockfish settings, and D8 depth for all three conditions.
5. Verify decoder hashes are identical across all conditions.
6. Verify the shuffled condition has exactly the same per-stage applied-advantage multiset as the primary condition, only permuted.
7. Report real pairwise ranking accuracy in addition to top-1/top-3 and robust regret statistics.
8. Treat median regret, trimmed mean regret, P90/P95 regret, >500 cp blunder rate, >1000 cp blunder rate, and lost-mate rate as primary descriptive error metrics; raw mean cp regret is heavily distorted by mate scores.
9. Repeat the corrected three-condition experiment over multiple independent seeds before interpreting a small effect as reliable. Five seeds is preferred; three is the minimum useful replication target for the next pass.
10. Do not tune the plasticity constants against the held-out confirmation set.

## Interpretation boundary

A successful rerun would show that a biologically localized reward-modulated plasticity rule changes held-out chess behavior more usefully than both a frozen network and a signal-matched shuffled control while preserving topology/sign constraints.

A failed rerun would still be scientifically useful: it would indicate that the current KC-to-MBON eligibility/reward rule is insufficient for chess credit assignment under the present encoder, recurrent dynamics, and frozen decoder. It would not by itself falsify the broader locked recurrent-depth research goal.
