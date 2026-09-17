# Plan 3 neuromodulated curriculum

```text
   stage  validation_positions  top1_accuracy_d8  mean_regret_cp_d8  mate_solve_rate_d8                                                                                                                                                                                                                                                                                                          plasticity
movement                     1               0.0              200.0                 0.0               {'update_count': 3.0, 'rms_log_ratio': 0.011820908214450955, 'median_abs_log_ratio': 0.00013072651305350277, 'p95_abs_log_ratio': 0.007579568803637493, 'fraction_changed_gt_5pct': 0.023746701846965697, 'fraction_changed_gt_10pct': 0.0, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
endgames                     1               0.0              132.0                 0.0   {'update_count': 7.0, 'rms_log_ratio': 0.025252545639447, 'median_abs_log_ratio': 0.0006245628634690149, 'p95_abs_log_ratio': 0.05126012436906948, 'fraction_changed_gt_5pct': 0.055408970976253295, 'fraction_changed_gt_10pct': 0.023746701846965697, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
 tactics                     1               0.0              171.0                 0.0 {'update_count': 12.0, 'rms_log_ratio': 0.03560480036905091, 'median_abs_log_ratio': 0.0020220320319941205, 'p95_abs_log_ratio': 0.07427892766045002, 'fraction_changed_gt_5pct': 0.07387862796833773, 'fraction_changed_gt_10pct': 0.026385224274406333, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
   mates                     1               0.0                2.0                 0.0   {'update_count': 17.0, 'rms_log_ratio': 0.042403835100347044, 'median_abs_log_ratio': 0.002346864823678715, 'p95_abs_log_ratio': 0.08808578609097169, 'fraction_changed_gt_5pct': 0.10026385224274406, 'fraction_changed_gt_10pct': 0.0395778364116095, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
```

Control: `shuffled`. The decoder was fit before plasticity, and only predeclared existing KC-to-MBON edges were updated.
