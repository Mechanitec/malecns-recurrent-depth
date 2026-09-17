# Plan 3 neuromodulated curriculum

```text
   stage  validation_positions  top1_accuracy_d8  mean_regret_cp_d8  mate_solve_rate_d8                                                                                                                                                                                                                                                                                                           plasticity
movement                     3          1.000000           0.000000            1.000000 {'update_count': 13.0, 'rms_log_ratio': 0.026359930106686927, 'median_abs_log_ratio': 0.0014293610883214637, 'p95_abs_log_ratio': 0.060353900562654966, 'fraction_changed_gt_5pct': 0.0712401055408971, 'fraction_changed_gt_10pct': 0.018469656992084433, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
endgames                     3          1.000000           0.000000            1.000000   {'update_count': 29.0, 'rms_log_ratio': 0.040712235689138794, 'median_abs_log_ratio': 0.0033065432177208703, 'p95_abs_log_ratio': 0.07604567705869039, 'fraction_changed_gt_5pct': 0.10290237467018469, 'fraction_changed_gt_10pct': 0.0395778364116095, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
 tactics                     3          0.333333         534.000000            0.333333   {'update_count': 48.0, 'rms_log_ratio': 0.048246027890424384, 'median_abs_log_ratio': 0.006986352519951059, 'p95_abs_log_ratio': 0.09810714394029194, 'fraction_changed_gt_5pct': 0.1345646437994723, 'fraction_changed_gt_10pct': 0.052770448548812667, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
   mates                     3          0.000000       66702.333333            0.000000     {'update_count': 69.0, 'rms_log_ratio': 0.05664117240881307, 'median_abs_log_ratio': 0.009753150378568578, 'p95_abs_log_ratio': 0.11351162746888883, 'fraction_changed_gt_5pct': 0.19788918205804748, 'fraction_changed_gt_10pct': 0.0712401055408971, 'fraction_lower_bound': 0.0, 'fraction_upper_bound': 0.0}
```

Control: `reward_aversive`. The decoder was fit before plasticity, and only predeclared existing KC-to-MBON edges were updated.
