# Public Reproducibility Package

This folder contains the minimum public materials accompanying the manuscript
*"Hydrologic diagnostics of multi-agent reinforcement learning policies for
cascade reservoir operation: A case study of the upper Yangtze River"*.

The package supports configuration-level review and auditability of the
training-parameter sensitivity analysis reported in the manuscript.

## Scope

Included materials document:

- the fixed reservoir-control boundary used by all experiments;
- the six training parameters varied in the manuscript;
- the Stage 0, Stage A, and Stage B configuration matrices;
- the Stage C mixed-sequence checkpoint-validation scenario definition;
- representative-role names used in the manuscript;
- the seed and checkpoint-selection protocol;
- a minimal response-dataset schema and export-release manifest;
- source-code files that should be included when the repository is made public.

This package does not include raw reservoir workbooks, raw inflow workbooks,
trained checkpoints, raw evaluator workbooks, or large run logs. Those files may
be released separately only if data-license, repository-size, and
confidentiality constraints allow.

## Folder Contents

`configs/`

- `fixed_control_settings.csv`: fixed system, observation, reward, action, and
  export settings.
- `learning_parameter_space.csv`: six training parameters, defaults,
  and candidate values.
- `stage0_stageA_stageB_configuration_matrix.csv`: full deterministic
  configuration matrix for the verification, screening, and main evidence
  stages.
- `stage_b_repeated_sensitivity_matrix.csv`: the 19 Stage B configurations used
  for the main repeated sensitivity analysis.
- `stage_c_mixed_sequence_scenarios.json`: public Stage C scenario metadata.
- `representative_roles_and_checkpoints.csv`: manuscript-facing role names and
  their corresponding internal configuration identifiers.
- `seed_and_checkpoint_protocol.json`: seed rule and Stage C checkpoint
  selection rule.
- `acceptability_rule_summary.json`: metric-direction and acceptability-scope
  summary.

`schema/`

- `response_dataset_schema_minimal.csv`: field groups needed to interpret the
  Stage B and Stage C response datasets.
- `export_release_manifest.csv`: what is included, optional, or excluded from
  the minimum public package.

`manifests/`

- `source_code_manifest.csv`: repository files that should accompany this
  public package.
- `package_manifest.json`: high-level package metadata and exclusions.

`code/`

- `config/`: public copies of the PaperFour configuration generator,
  evaluation launcher, learning-parameter utility, and a redacted fixed-setting
  template.
- `analysis/`: public copies of the response-dataset, evidence-pack, and Stage C
  scorecard builders.
- `figures/scripts/`: public copies of the figure data-preparation and plotting
  scripts.

`statements/`

- `data_model_code_availability_statement.md`: draft availability statement for
  submission.
- `reproducible_results_plan.md`: minimal reproducible-results plan aligned with
  the manuscript boundary.
- `github_upload_checklist.md`: pre-upload checks.

## Recommended Public Repository Layout

When uploading to GitHub, keep this folder with the manuscript materials and
also include the source files listed in `manifests/source_code_manifest.csv`.
If raw data cannot be released, keep the restricted files out of the public
repository and retain the explicit restricted-data notes in the availability
statement.

The scripts in `code/` are included for auditability. Full reruns still require
the restricted reservoir and inflow workbooks, trained checkpoints for Stage C,
and the project runtime environment.

## Interpretation Boundary

The package supports a fixed five-reservoir HAPPO sensitivity study. It does not
support claims of a transferable training parameter recommendation, cross-basin
transferability, broad hydrologic robustness, or algorithm superiority over
other reservoir-operation methods.
