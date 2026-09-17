# Plan 3 control comparison

All conditions use the same real MaleCNS graph, exact KC-to-MBON edge scope, frozen decoder procedure, and curriculum limits unless noted by the run metadata.

```text
      condition    stage  top1_accuracy_d8  top3_accuracy_d8  median_regret_cp_d8  mean_regret_cp_d8  blunder_rate_gt1000cp_d8  mate_solve_rate_d8  training_positions   elapsed_s  plastic_edge_count
reward_aversive movement          0.416667          0.861111                200.0         116.666667                  0.000000            0.416667                 803 2697.830887               11119
reward_aversive endgames          0.187500          0.718750                 34.5        9428.218750                  0.093750            0.187500                 803 2697.830887               11119
reward_aversive  tactics          0.220000          0.700000                201.0         375.240000                  0.120000            0.220000                 803 2697.830887               11119
reward_aversive    mates          0.236842          0.763158                321.0       31764.473684                  0.368421            0.236842                 803 2697.830887               11119
         frozen movement          0.166667          0.861111                200.0         166.666667                  0.000000            0.166667                 803 2677.667513               11119
         frozen endgames          0.156250          0.687500                 34.5        9429.750000                  0.093750            0.156250                 803 2677.667513               11119
         frozen  tactics          0.280000          0.700000                165.5         358.360000                  0.120000            0.280000                 803 2677.667513               11119
         frozen    mates          0.236842          0.710526                321.0       34371.473684                  0.368421            0.236842                 803 2677.667513               11119
       shuffled movement          0.722222          0.777778                  0.0          55.555556                  0.000000            0.722222                 404 1465.537752               11119
       shuffled endgames          0.187500          0.625000                 12.0        6302.000000                  0.062500            0.187500                 404 1465.537752               11119
       shuffled  tactics          0.240000          0.680000                210.0         281.120000                  0.080000            0.240000                 404 1465.537752               11119
       shuffled    mates          0.315789          0.842105                132.0       31562.526316                  0.315789            0.315789                 404 1465.537752               11119
```

Interpret the reward/aversive condition against both frozen and shuffled-signal controls; these are exploratory held-out curriculum metrics, not a chess Elo estimate.
