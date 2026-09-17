# Plan 3 neuromodulated curriculum results

The primary run used the real MaleCNS graph, D8 training, localized existing KC-to-MBON plasticity, and a decoder frozen before plasticity.

- Training updates: `803`; elapsed: `2697.8 s`.
- Plastic edges: `11119`; topology preserved: `True`; signs preserved: `True`.
- Sequential mate evaluation: `9` positive-mate validation positions; solve rate `0.1111111111111111`.
- The mate result is descriptive, not a claim of general chess strength; the available positive mate validation set is small.

## Stage metrics

```text
   stage  validation_positions  top1_accuracy_d8  top3_accuracy_d8  pairwise_ranking_accuracy_d8  mean_regret_cp_d8  median_regret_cp_d8  trimmed_mean_regret_cp_d8  p90_regret_cp_d8  p95_regret_cp_d8  blunder_rate_gt500cp_d8  blunder_rate_gt1000cp_d8  mate_blunder_count_d8  mean_teacher_rank_d8  legal_selection_rate_d8                                                                                                                       movement_piece_accuracy_d8  mate_solve_rate_d8                                                                                                                                                                                                                                                                                                                                             plasticity
movement                    36          0.416667          0.861111                      0.416667         116.666667                200.0                 117.241379             200.0            200.00                 0.000000                  0.000000                      0              4.111111                 0.416667 {"bishop": 0.6666666666666666, "king": 0.16666666666666666, "knight": 0.3333333333333333, "pawn": 0.0, "queen": 0.8333333333333334, "rook": 0.5}            0.416667                                     {'update_count': 144.0, 'rms_log_ratio': 0.06881942473173899, 'median_abs_log_ratio': 0.004839962110783791, 'p95_abs_log_ratio': 0.10035239145588011, 'fraction_changed_gt_5pct': 0.11349941541505532, 'fraction_changed_gt_10pct': 0.05144347513265581, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
endgames                    32          0.187500          0.718750                      0.187500        9428.218750                 34.5                 125.360000             670.9          99253.95                 0.218750                  0.093750                      0              2.937500                      NaN                                                                                                                                               {}            0.187500 {'update_count': 301.0, 'rms_log_ratio': 0.11099701144461985, 'median_abs_log_ratio': 0.021619915989168712, 'p95_abs_log_ratio': 0.22935972343868738, 'fraction_changed_gt_5pct': 0.30020685313427464, 'fraction_changed_gt_10pct': 0.15774799892076627, 'fraction_lower_bound': 0.0002698084360104326, 'fraction_upper_bound': 0.0007194891626944869}
 tactics                    50          0.220000          0.700000                      0.220000         375.240000                201.0                 268.600000            1112.0           1573.65                 0.260000                  0.120000                      0              2.760000                      NaN                                                                                                                                               {}            0.220000       {'update_count': 556.0, 'rms_log_ratio': 0.150628032807489, 'median_abs_log_ratio': 0.04077649401973485, 'p95_abs_log_ratio': 0.3345982567948883, 'fraction_changed_gt_5pct': 0.4453637917078874, 'fraction_changed_gt_10pct': 0.2709776058998111, 'fraction_lower_bound': 0.0010792337440417303, 'fraction_upper_bound': 0.0034175735227988126}
   mates                    38          0.236842          0.763158                      0.236842       31764.473684                321.0               25951.935484          100322.1         100532.10                 0.394737                  0.368421                      0              2.973684                      NaN                                                                                                                                               {}            0.236842                      {'update_count': 803.0, 'rms_log_ratio': 0.1817806664514006, 'median_abs_log_ratio': 0.05572624628069417, 'p95_abs_log_ratio': 0.44226166615318835, 'fraction_changed_gt_5pct': 0.5281050454177534, 'fraction_changed_gt_10pct': 0.3485025631801421, 'fraction_lower_bound': 0.00044968072668405434, 'fraction_upper_bound': 0.0}
```

## Curriculum coverage

```text
          rows  positions  train_positions  validation_positions
movement  4465        480              360                   120
endgames  2079        420              320                   100
tactics   3234        650              500                   150
mates      944        190              152                    38
```

## Artifact verification

- Required artifacts present: `True`.
- Plot artifacts present: `True`.
