# GitHub Upload Checklist

Before uploading this folder to a public repository, check the following.

- Confirm no raw reservoir-system workbook is included.
- Confirm no raw inflow workbook is included.
- Confirm no trained checkpoint file is included.
- Confirm no raw evaluator workbook or large run log is included.
- Confirm any copied manifest files have local absolute paths redacted if
  needed.
- Confirm `code/config/reservoir_happo_public_template.py` is used as the public
  fixed-setting template instead of a private machine-specific full config.
- Confirm `configs/stage_b_repeated_sensitivity_matrix.csv` has 19 Stage B
  configurations.
- Confirm Stage B seed protocol lists seeds 100, 200, and 300.
- Confirm Stage C is described as checkpoint validation with no retraining.
- Confirm `total_spillage` and `flood_season_spillage` remain neutral spillage
  diagnostics.
- Confirm manuscript-facing role names match the main text:
  Baseline, Operationally acceptable candidate, High-return candidate,
  Conservative candidate, High-clipping risk, Long-horizon risk, and
  Instability case.

Recommended optional additions, only if public release is allowed:

- redacted Stage B aggregate response dataset;
- Stage C scorecard;
- figure-source CSV files;
- exact package/version file for the Python environment.
