# Supplementary data for manuscript HEENG-7124

These CSV files report existing experiment records used in the revised manuscript, *Operational and Hydrologic Diagnostics of Multi-Agent Reinforcement Learning Policies for Cascade Reservoir Operation: A Case Study of the Upper Yangtze River*.

| File | Contents |
|---|---|
| `Supplementary_Table_S1.csv` | All 19 Stage B parameter configurations and diagnostic summaries over three training seeds. |
| `Supplementary_Table_S2.csv` | All 24 Stage A screening configurations, their training seeds and operational scores. |
| `Supplementary_Table_S3.csv` | The eight Stage C trained policies, their roles, selected training seeds, Stage B scores and Stage C scores. |
| `stage_c_sequence_provenance.csv` | The five Stage C source hydrologic years, their sequence order, hydrologic classes and annual inflow exceedance frequencies. |

## Fields and units

- Configuration IDs link the parameter settings and policy records across the tables.
- `learning_rate` and `critic_learning_rate` are the actor and critic learning rates; `clip_ratio`, `entropy_weight`, `discount_factor` and `gae_lambda` are the clipping constraint, entropy weight, discount factor and generalized advantage estimation parameter.
- In S1, `mean`, `std` and `cv` denote the three-seed mean, sample standard deviation and standard deviation divided by the mean. Read zero or undefined CV values together with the corresponding means.
- `return` and `operational_score` denote cumulative policy reward. They are dimensionless scores, distinct from energy generation.
- `power` totals are in 10^8 kWh. Spillage quantities, including cumulative `storage_spill_conflict`, are in 10^8 m3.
- `low_level_pressure_days` is lower-bound water-level pressure (LBP), in reservoir-days, using the fixed 10% normal-minus-dead-level margin.
- `mean_action_correction` is the mean absolute difference between proposed and executed normalized actions. `any_violation_rate` is the fraction of evaluation time steps with a constraint violation. Both are dimensionless.
- In S3, `source_seed` is the selected training seed, `source_return` its Stage B score, and `eval_return_mean` the reported Stage C score. Representative policies are selected from the seed whose Stage B score is nearest the three-seed median.
- Annual inflow exceedance frequencies in the sequence metadata are percentages. Stage C concatenates the 2021, 2000, 1969, 1976 and 2006 hydrologic years in very-wet to very-dry order.

Stage B uses the common fixed, continuous five-year evaluation sequence. Stage C evaluates existing trained policies on the stated separate mixed five-year sequence without retraining. These files provide configuration summaries and source-year metadata; they do not contain raw reservoir workbooks, raw gauging inflows or trained policy models.
