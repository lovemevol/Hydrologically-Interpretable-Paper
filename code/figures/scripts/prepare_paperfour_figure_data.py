from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover - optional runtime dependency
    EventAccumulator = None  # type: ignore[assignment]

from _paperfour_plot_paths import FIGURE_DATA_DIR, STAGE_B_OUTPUT_DIR, STAGE_C_OUTPUT_DIR, ensure_figure_dirs


LEARNING_PARAMS = [
    "learning_rate",
    "critic_learning_rate",
    "clip_ratio",
    "entropy_weight",
    "discount_factor",
    "gae_lambda",
]

PRIMARY_CONFIGS = ["default", "gamma099_clip02", "gamma0999_clip02", "risk_lhs009"]
RESERVOIR_ORDER = ["乌东德", "白鹤滩", "溪洛渡", "向家坝", "三峡"]
RESERVOIR_LABELS = {
    "乌东德": "Wudongde",
    "白鹤滩": "Baihetan",
    "溪洛渡": "Xiluodu",
    "向家坝": "Xiangjiaba",
    "三峡": "Three Gorges",
}

METRIC_META = {
    "return": ("Evaluator return", "reward", "higher_better", "algorithm performance"),
    "power": ("Generation", "10^8 kWh", "higher_better", "water benefit"),
    "low_level_pressure_days": ("Lower-bound pressure days", "reservoir-days", "lower_better", "water-level pressure"),
    "boundary_pressure_spill": ("Boundary-pressure spillage", "10^8 m3", "lower_better", "conditional spill risk"),
    "mean_action_correction": ("Action correction", "normalized action", "lower_better", "executability"),
    "any_violation_rate": ("Any violation rate", "fraction", "lower_better", "operational constraint"),
    "non_flood_spill": ("Non-flood-season spillage", "10^8 m3", "lower_better", "conditional spill risk"),
}

TRAINING_METRIC_META = {
    "eval_return": ("evaluator_step/eval_episode_return_mean", "reward", "higher_better"),
    "approx_kl": ("learner_step/agent{agent}_approx_kl_avg", "unitless", "diagnostic"),
    "clipfrac": ("learner_step/agent{agent}_clipfrac_avg", "fraction", "diagnostic"),
    "entropy_loss": ("learner_step/agent{agent}_entropy_loss_avg", "loss", "diagnostic"),
    "value_loss": ("learner_step/agent{agent}_value_loss_avg", "loss", "diagnostic"),
    "policy_loss": ("learner_step/agent{agent}_policy_loss_avg", "loss", "diagnostic"),
}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _season(month: int) -> str:
    if 6 <= int(month) <= 9:
        return "flood"
    if 1 <= int(month) <= 5:
        return "dry"
    return "recovery"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _scale_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    span = values.max() - values.min()
    if not np.isfinite(span) or abs(span) < 1e-12:
        return pd.Series(0.5, index=values.index)
    return (values - values.min()) / span


def _event_scalars(accumulator: EventAccumulator, tag: str) -> pd.DataFrame:
    records = accumulator.Scalars(tag)
    if not records:
        return pd.DataFrame(columns=["step", "value"])
    return pd.DataFrame({"step": [item.step for item in records], "value": [item.value for item in records]})


def prepare_training_curves(response: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    columns = [
        "config_id",
        "run_name",
        "seed",
        "step",
        "relative_step",
        "metric",
        "value",
        "unit",
        "indicator_direction",
    ]
    rows: list[pd.DataFrame] = []
    if EventAccumulator is None:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output_dir / "fig_training_curves.csv", index=False, encoding="utf-8-sig")
        return frame

    for _, run in response.iterrows():
        serial_dir = Path(str(run["run_dir"])) / "log" / "serial"
        if not serial_dir.exists():
            continue
        try:
            accumulator = EventAccumulator(str(serial_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception:
            continue
        available_tags = set(accumulator.Tags().get("scalars", []))
        for metric, (tag_pattern, unit, direction) in TRAINING_METRIC_META.items():
            if "{agent}" not in tag_pattern:
                if tag_pattern not in available_tags:
                    continue
                metric_frame = _event_scalars(accumulator, tag_pattern)
            else:
                agent_frames = []
                for agent in range(5):
                    tag = tag_pattern.format(agent=agent)
                    if tag in available_tags:
                        one_agent = _event_scalars(accumulator, tag)
                        one_agent["agent"] = agent
                        agent_frames.append(one_agent)
                if not agent_frames:
                    continue
                metric_frame = (
                    pd.concat(agent_frames, ignore_index=True)
                    .groupby("step", as_index=False)
                    .agg(value=("value", "mean"))
                )
            if metric_frame.empty:
                continue
            metric_frame["config_id"] = run["config_id"]
            metric_frame["run_name"] = run["run_name"]
            metric_frame["seed"] = run["seed"]
            metric_frame["metric"] = metric
            metric_frame["unit"] = unit
            metric_frame["indicator_direction"] = direction
            rows.append(metric_frame)

    if rows:
        frame = pd.concat(rows, ignore_index=True)
        max_step = frame.groupby(["run_name", "metric"])["step"].transform("max").replace(0, np.nan)
        frame["relative_step"] = frame["step"] / max_step
        frame = frame[columns].sort_values(["config_id", "seed", "metric", "step"])
    else:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(output_dir / "fig_training_curves.csv", index=False, encoding="utf-8-sig")
    return frame


def prepare_fig1(output_dir: Path) -> None:
    nodes = pd.DataFrame(
        [
            ("not_applicable", "params", "learning_rate", "Learning rate", "HAPPO update magnitude", "algorithm", "none", "controlled variable"),
            ("not_applicable", "params", "clip_ratio", "PPO clip ratio", "Policy update trust region", "algorithm", "none", "controlled variable"),
            ("not_applicable", "params", "discount_factor", "Discount factor", "Temporal credit horizon", "algorithm", "none", "controlled variable"),
            ("not_applicable", "learner", "approx_kl", "Approx. KL", "Policy update drift", "algorithm", "unitless", "diagnostic"),
            ("not_applicable", "learner", "clipfrac", "Clip fraction", "Frequency of clipped updates", "algorithm", "fraction", "diagnostic"),
            ("not_applicable", "execution", "action_correction", "Action correction", "Controller feasibility pressure", "operation", "normalized action", "lower_better"),
            ("not_applicable", "hydrology", "low_level_pressure", "Lower-bound pressure", "Reservoir lower-bound operating stress", "water", "reservoir-days", "lower_better"),
            ("not_applicable", "hydrology", "conditional_spill", "Conditional spillage", "Non-flood and boundary-pressure spillage", "water", "10^8 m3", "lower_better"),
            ("not_applicable", "benefit", "generation_return", "Return and generation", "Performance and water-resource benefit", "water", "reward / 10^8 kWh", "higher_better"),
        ],
        columns=["config_id", "layer", "node_id", "label", "description", "domain", "unit", "indicator_direction"],
    )
    edges = pd.DataFrame(
        [
            ("not_applicable", "learning_rate", "approx_kl", "larger update step changes policy drift", "mechanism edge", "diagnostic"),
            ("not_applicable", "clip_ratio", "clipfrac", "clip threshold changes clipped-update frequency", "mechanism edge", "diagnostic"),
            ("not_applicable", "discount_factor", "generation_return", "credit horizon shapes inter-seasonal trade-off", "mechanism edge", "diagnostic"),
            ("not_applicable", "approx_kl", "action_correction", "unstable learning can increase infeasible actions", "mechanism edge", "lower_better downstream"),
            ("not_applicable", "clipfrac", "action_correction", "aggressive updates can increase correction pressure", "mechanism edge", "lower_better downstream"),
            ("not_applicable", "action_correction", "low_level_pressure", "execution pressure is reflected in reservoir state stress", "mechanism edge", "lower_better downstream"),
            ("not_applicable", "low_level_pressure", "conditional_spill", "spill interpretation depends on level and season", "mechanism edge", "conditional risk"),
            ("not_applicable", "conditional_spill", "generation_return", "hydrologic diagnostics contextualize return", "mechanism edge", "diagnostic"),
        ],
        columns=["config_id", "source", "target", "mechanism_note", "unit", "indicator_direction"],
    )
    nodes.to_csv(output_dir / "fig1_framework_nodes.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(output_dir / "fig1_framework_edges.csv", index=False, encoding="utf-8-sig")


def prepare_fig1_dynamic_evidence(training_curves: pd.DataFrame, daily: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    if not training_curves.empty:
        selected = training_curves[
            training_curves["config_id"].isin(PRIMARY_CONFIGS)
            & training_curves["metric"].isin(["eval_return", "approx_kl", "clipfrac"])
        ].copy()
        if not selected.empty:
            selected["step_bin"] = (pd.to_numeric(selected["relative_step"], errors="coerce") * 80).round() / 80
            grouped = (
                selected.groupby(["config_id", "metric", "step_bin", "unit", "indicator_direction"], as_index=False)
                .agg(value=("value", "mean"))
                .rename(columns={"step_bin": "relative_time"})
            )
            rows.append(grouped)
    if not daily.empty:
        hydro = daily[daily["config_id"].isin(PRIMARY_CONFIGS)].copy()
        hydro["relative_time"] = hydro.groupby("config_id").cumcount() / hydro.groupby("config_id")["date"].transform("count").clip(lower=1)
        hydro = (
            hydro.assign(metric="low_pressure_reservoir_count", value=hydro["low_pressure_reservoir_count"], unit="reservoir count", indicator_direction="lower_better")
            [["config_id", "metric", "relative_time", "unit", "indicator_direction", "value"]]
        )
        hydro["relative_time"] = (hydro["relative_time"] * 80).round() / 80
        hydro = hydro.groupby(["config_id", "metric", "relative_time", "unit", "indicator_direction"], as_index=False).agg(value=("value", "mean"))
        rows.append(hydro)

    if rows:
        frame = pd.concat(rows, ignore_index=True)
        frame["value_scaled"] = frame.groupby("metric")["value"].transform(_scale_series)
    else:
        frame = pd.DataFrame(columns=["config_id", "metric", "relative_time", "unit", "indicator_direction", "value", "value_scaled"])
    frame.to_csv(output_dir / "fig1_dynamic_evidence.csv", index=False, encoding="utf-8-sig")


def prepare_fig2(output_dir: Path) -> None:
    fixed = pd.DataFrame(
        [
            ("not_applicable", "reservoir_system", "Five-reservoir cascade workbook", "5-reservoir cascade", "fixed", "system"),
            ("not_applicable", "algorithm", "HAPPO", "Heterogeneous-agent PPO", "fixed", "algorithm"),
            ("not_applicable", "observation", "Fixed observation setting", "Local variables + shared cascade context", "fixed", "algorithm"),
            ("not_applicable", "reward", "default_reward", "Default reward; no reward retuning", "fixed", "experiment boundary"),
            ("not_applicable", "action_space", "discrete-100", "Fixed discrete action dimension", "fixed", "execution"),
        ],
        columns=["config_id", "component", "value", "description", "indicator_direction", "unit"],
    )
    stages = pd.DataFrame(
        [
            ("not_applicable", "Stage 0", "smoke test", 3, 1, "config and export validation", "runs", "workflow count"),
            ("not_applicable", "Stage A", "screening", 24, 1, "coarse sensitivity screening", "runs", "workflow count"),
            ("not_applicable", "Stage B", "repeated sensitivity analysis", 19, 3, "sampled acceptable values and counterexamples", "runs", "workflow count"),
            ("not_applicable", "Stage C", "mixed-sequence checkpoint validation", 8, 1, "representative checkpoint validation", "evaluations", "workflow count"),
        ],
        columns=["config_id", "stage", "purpose", "config_count", "seed_or_eval_count", "output_role", "unit", "indicator_direction"],
    )
    stages["data_units"] = stages["config_count"] * stages["seed_or_eval_count"]
    stages["source_layer"] = ["runner", "runner", "runner + evaluator", "offline evaluator"]
    ranges = pd.DataFrame(
        [
            ("all_stage_b_configs", "learning_rate", "1e-4, 3e-4, 5e-4, 1e-3", "actor learning rate", "controlled variable", "unitless"),
            ("all_stage_b_configs", "critic_learning_rate", "1e-4, 3e-4, 5e-4, 1e-3", "critic learning rate", "controlled variable", "unitless"),
            ("all_stage_b_configs", "clip_ratio", "0.1, 0.2, 0.3", "PPO clipping threshold", "controlled variable", "unitless"),
            ("all_stage_b_configs", "entropy_weight", "0.004, 0.008, 0.016, 0.032", "exploration regularization", "controlled variable", "unitless"),
            ("all_stage_b_configs", "discount_factor", "0.99, 0.995, 0.999", "temporal credit horizon", "controlled variable", "unitless"),
            ("all_stage_b_configs", "gae_lambda", "0.90, 0.95, 0.98", "advantage smoothing", "controlled variable", "unitless"),
        ],
        columns=["config_id", "parameter", "candidate_values", "algorithm_meaning", "indicator_direction", "unit"],
    )
    output_chain = pd.DataFrame(
        [
            ("all_stage_b_configs", "manifest + learner logs", "learning dynamics", "KL, clipfrac, entropy, value loss", "diagnostic", "mixed"),
            ("all_stage_b_configs", "evaluator Excel", "reservoir operation", "levels, outflow, spill, action correction", "diagnostic", "daily"),
            ("all_stage_b_configs", "response dataset", "sensitivity analysis", "importance, interactions, sampled acceptable values", "diagnostic", "table"),
            ("all_stage_b_configs", "Stage C scorecard", "checkpoint validation", "return and hydrologic pressure under mixed sequence", "diagnostic", "table"),
        ],
        columns=["config_id", "input_source", "analysis_layer", "output_metrics", "indicator_direction", "unit"],
    )
    fixed.to_csv(output_dir / "fig2_fixed_system.csv", index=False, encoding="utf-8-sig")
    stages.to_csv(output_dir / "fig2_stage_design.csv", index=False, encoding="utf-8-sig")
    ranges.to_csv(output_dir / "fig2_parameter_ranges.csv", index=False, encoding="utf-8-sig")
    output_chain.to_csv(output_dir / "fig2_output_chain.csv", index=False, encoding="utf-8-sig")


def prepare_fig4(spearman: pd.DataFrame, response: pd.DataFrame, output_dir: Path) -> None:
    selected_metrics = ["return", "low_level_pressure_days", "mean_action_correction", "any_violation_rate", "boundary_pressure_spill"]
    frame = spearman[(spearman["level"] == "config") & spearman["metric"].isin(selected_metrics)].copy()
    frame["config_id"] = "all_stage_b_configs"
    frame["metric_label"] = frame["metric"].map(lambda metric: METRIC_META[metric][0])
    frame["unit"] = frame["metric"].map(lambda metric: METRIC_META[metric][1])
    frame["indicator_direction"] = frame["metric"].map(lambda metric: METRIC_META[metric][2])
    frame["metric_group"] = frame["metric"].map(lambda metric: METRIC_META[metric][3])
    frame["abs_spearman"] = frame["spearman"].abs()
    frame.to_csv(output_dir / "fig4_parameter_importance.csv", index=False, encoding="utf-8-sig")

    rows = []
    for parameter in LEARNING_PARAMS:
        grouped = (
            response.groupby(parameter, dropna=False)
            .agg(
                n=("run_name", "count"),
                return_mean=("return", "mean"),
                return_std=("return", "std"),
                low_level_pressure_days_mean=("low_level_pressure_days", "mean"),
                mean_action_correction_mean=("mean_action_correction", "mean"),
                any_violation_rate_mean=("any_violation_rate", "mean"),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            for metric in ["return", "low_level_pressure_days", "mean_action_correction", "any_violation_rate"]:
                metric_col = f"{metric}_mean"
                rows.append(
                    {
                        "config_id": "all_stage_b_configs",
                        "parameter": parameter,
                        "parameter_value": row[parameter],
                        "metric": metric,
                        "metric_label": METRIC_META[metric][0],
                        "metric_value": row[metric_col],
                        "n": row["n"],
                        "unit": METRIC_META[metric][1],
                        "indicator_direction": METRIC_META[metric][2],
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "fig4_parameter_direction.csv", index=False, encoding="utf-8-sig")


def _interaction_frame(interactions: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    rows = []
    for _, row in interactions.iterrows():
        params = {row["parameter_a"], row["parameter_b"]}
        if params != {a, b}:
            continue
        value_a = row["value_a"] if row["parameter_a"] == a else row["value_b"]
        value_b = row["value_b"] if row["parameter_b"] == b else row["value_a"]
        record = row.to_dict()
        record[a] = float(value_a)
        record[b] = float(value_b)
        record["config_id"] = "grouped_stage_b_configs"
        rows.append(record)
    return pd.DataFrame(rows)


def prepare_fig5(interactions: pd.DataFrame, response: pd.DataFrame, output_dir: Path) -> None:
    frame = _interaction_frame(interactions, "clip_ratio", "discount_factor")
    if frame.empty:
        frame = response.copy()
        frame["config_id"] = frame["config_id"].astype(str)
    for metric in ["return", "low_level_pressure_days", "mean_action_correction", "any_violation_rate"]:
        if f"{metric}_mean" in frame.columns:
            frame[f"{metric}_unit"] = METRIC_META[metric][1]
            frame[f"{metric}_direction"] = METRIC_META[metric][2]
    frame["unit"] = "reward, normalized action, reservoir-days, or fraction"
    frame["indicator_direction"] = "return higher_better; correction/pressure/violation lower_better"
    frame.to_csv(output_dir / "fig5_clip_gamma_response.csv", index=False, encoding="utf-8-sig")

    config_level = (
        response.groupby(["config_id", "clip_ratio", "discount_factor"], as_index=False)
        .agg(
            return_mean=("return", "mean"),
            return_std=("return", "std"),
            low_level_pressure_days_mean=("low_level_pressure_days", "mean"),
            mean_action_correction_mean=("mean_action_correction", "mean"),
            any_violation_rate_mean=("any_violation_rate", "mean"),
            ld_approx_kl_mean=("ld_approx_kl_mean_agent_mean", "mean"),
            ld_clipfrac_mean=("ld_clipfrac_mean_agent_mean", "mean"),
        )
    )
    config_level["unit"] = "reward, reservoir-days, normalized action, or fraction"
    config_level["indicator_direction"] = "return higher_better; update diagnostics neutral; pressure/correction lower_better"
    config_level.to_csv(output_dir / "fig5_clip_gamma_3d.csv", index=False, encoding="utf-8-sig")


def prepare_fig6(response: pd.DataFrame, output_dir: Path, reservoir_daily: pd.DataFrame | None = None) -> None:
    rows = []
    season_columns = {
        "dry": ("dry_season_power", "dry_season_spill_neutral"),
        "flood": ("flood_season_power", "flood_season_spill_neutral"),
        "recovery": ("recovery_season_power", "recovery_season_spill_neutral"),
    }
    for _, row in response.iterrows():
        for season, (power_col, spill_col) in season_columns.items():
            rows.append(
                {
                    "config_id": row["config_id"],
                    "run_name": row["run_name"],
                    "seed": row["seed"],
                    "discount_factor": row["discount_factor"],
                    "gae_lambda": row["gae_lambda"],
                    "season": season,
                    "season_power": row.get(power_col, np.nan),
                    "season_spill_neutral": row.get(spill_col, np.nan),
                    "return": row.get("return", np.nan),
                    "low_level_pressure_days": row.get("low_level_pressure_days", np.nan),
                    "low_limit_margin_mean": row.get("low_limit_margin_mean", np.nan),
                    "remaining_storage_mean": row.get("remaining_storage_mean", np.nan),
                    "unit": "10^8 kWh, 10^8 m3, fraction, or reservoir-days",
                    "indicator_direction": "seasonal power higher_better; seasonal spill neutral spillage diagnostic; pressure lower_better",
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "fig6_gamma_gae_seasonal_response.csv", index=False, encoding="utf-8-sig")

    if reservoir_daily is None or reservoir_daily.empty:
        seasonal_pressure = pd.DataFrame(
            columns=["config_id", "stage_c_role", "season", "low_pressure_days", "remaining_storage_ratio_mean", "low_limit_margin_mean", "unit", "indicator_direction"]
        )
    else:
        seasonal_pressure = (
            reservoir_daily.groupby(["config_id", "stage_c_role", "season"], dropna=False)
            .agg(
                low_pressure_days=("low_pressure_flag", "sum"),
                remaining_storage_ratio_mean=("remaining_storage_ratio", "mean"),
                low_limit_margin_mean=("low_limit_margin", "mean"),
            )
            .reset_index()
        )
        seasonal_pressure["unit"] = "reservoir-days or fraction"
        seasonal_pressure["indicator_direction"] = "pressure lower_better; storage and margin diagnostic"
    seasonal_pressure.to_csv(output_dir / "fig6_stage_c_seasonal_pressure.csv", index=False, encoding="utf-8-sig")


def _read_reservoir_timeseries(excel_path: Path, config_id: str, role: str, run_name: str) -> pd.DataFrame:
    xl = pd.ExcelFile(excel_path)
    frames = []
    for sheet_name in xl.sheet_names:
        if sheet_name in {"汇总", "Summary"}:
            continue
        frame = pd.read_excel(excel_path, sheet_name=sheet_name)
        if "日期" not in frame.columns:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["日期"], errors="coerce")
        frame["config_id"] = config_id
        frame["stage_c_role"] = role
        frame["run_name"] = run_name
        frame["reservoir"] = sheet_name
        frame["reservoir_label"] = RESERVOIR_LABELS.get(sheet_name, sheet_name)
        frame["reservoir_order"] = RESERVOIR_ORDER.index(sheet_name) if sheet_name in RESERVOIR_ORDER else 99
        high_low_span = pd.to_numeric(frame["高限水位(m)"], errors="coerce") - pd.to_numeric(frame["低限水位(m)"], errors="coerce")
        frame["normalized_water_state"] = _safe_ratio(
            pd.to_numeric(frame["当日水位(m)"], errors="coerce") - pd.to_numeric(frame["低限水位(m)"], errors="coerce"),
            high_low_span,
        ).clip(-0.5, 1.5)
        frame["low_pressure_flag"] = (pd.to_numeric(frame["距低限归一化裕度"], errors="coerce") <= 0.05).astype(int)
        frame["any_violation"] = (
            pd.to_numeric(frame.get("生态违规", 0), errors="coerce").fillna(0)
            + pd.to_numeric(frame.get("保证出力违规", 0), errors="coerce").fillna(0)
            + pd.to_numeric(frame.get("航运违规", 0), errors="coerce").fillna(0)
        ).gt(0).astype(int)
        keep = {
            "date": "date",
            "config_id": "config_id",
            "stage_c_role": "stage_c_role",
            "run_name": "run_name",
            "reservoir": "reservoir",
            "reservoir_label": "reservoir_label",
            "reservoir_order": "reservoir_order",
            "normalized_water_state": "normalized_water_state",
            "低限水位(m)": "low_level_m",
            "当日水位(m)": "water_level_m",
            "高限水位(m)": "high_level_m",
            "距低限归一化裕度": "low_limit_margin",
            "距高限归一化裕度": "high_limit_margin",
            "剩余调节库容占比": "remaining_storage_ratio",
            "当日入流(m³/s)": "inflow_m3s",
            "当日出流(m³/s)": "outflow_m3s",
            "弃水流量(m³/s)": "spill_flow_m3s",
            "当日发电量(亿kW·h)": "generation_100mkwh",
            "动作修正幅度": "action_correction",
            "智能体动作": "agent_action",
            "实际动作": "actual_action",
            "low_pressure_flag": "low_pressure_flag",
            "any_violation": "any_violation",
        }
        available = {source: target for source, target in keep.items() if source in frame.columns}
        frames.append(frame[list(available)].rename(columns=available))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare_fig7_fig8(stage_c_index: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = stage_c_index[stage_c_index["config_id"].isin(PRIMARY_CONFIGS)].copy()
    selected["config_id"] = pd.Categorical(selected["config_id"], categories=PRIMARY_CONFIGS, ordered=True)
    selected = selected.sort_values("config_id")
    frames = []
    for _, row in selected.iterrows():
        frames.append(
            _read_reservoir_timeseries(
                Path(row["latest_evaluator_export"]),
                str(row["config_id"]),
                str(row["stage_c_role"]),
                str(row["run_name"]),
            )
        )
    long_frame = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    long_frame["month"] = long_frame["date"].dt.month
    long_frame["season"] = long_frame["month"].map(_season)
    long_frame["day_index"] = long_frame.groupby(["config_id", "reservoir"]).cumcount()
    long_frame["unit"] = "daily reservoir metric"
    long_frame["indicator_direction"] = "generation higher_better; pressure/correction/violation lower_better; spillage neutral spillage diagnostic"
    long_frame.to_csv(output_dir / "fig7_stage_c_reservoir_daily.csv", index=False, encoding="utf-8-sig")

    daily = (
        long_frame.groupby(["config_id", "stage_c_role", "run_name", "date", "season"], dropna=False)
        .agg(
            normalized_water_state_mean=("normalized_water_state", "mean"),
            normalized_water_state_min=("normalized_water_state", "min"),
            normalized_water_state_max=("normalized_water_state", "max"),
            low_limit_margin_mean=("low_limit_margin", "mean"),
            high_limit_margin_mean=("high_limit_margin", "mean"),
            remaining_storage_ratio_mean=("remaining_storage_ratio", "mean"),
            inflow_m3s_sum=("inflow_m3s", "sum"),
            outflow_m3s_sum=("outflow_m3s", "sum"),
            spill_flow_m3s_sum=("spill_flow_m3s", "sum"),
            generation_100mkwh_sum=("generation_100mkwh", "sum"),
            action_correction_mean=("action_correction", "mean"),
            low_pressure_reservoir_count=("low_pressure_flag", "sum"),
            violation_reservoir_count=("any_violation", "sum"),
        )
        .reset_index()
    )
    daily["unit"] = "daily system aggregate"
    daily["indicator_direction"] = "return higher_better; pressure/correction/violation lower_better; spillage neutral spillage diagnostic"
    daily.to_csv(output_dir / "fig7_mechanism_timeseries.csv", index=False, encoding="utf-8-sig")

    heterogeneity = (
        long_frame.groupby(["config_id", "stage_c_role", "reservoir", "reservoir_label", "reservoir_order"], dropna=False)
        .agg(
            generation_100mkwh_sum=("generation_100mkwh", "sum"),
            remaining_storage_ratio_mean=("remaining_storage_ratio", "mean"),
            action_correction_mean=("action_correction", "mean"),
            low_pressure_days=("low_pressure_flag", "sum"),
            violation_days=("any_violation", "sum"),
            normalized_water_state_mean=("normalized_water_state", "mean"),
            low_limit_margin_mean=("low_limit_margin", "mean"),
            high_limit_margin_mean=("high_limit_margin", "mean"),
        )
        .reset_index()
        .sort_values(["reservoir_order", "config_id"])
    )
    heterogeneity["unit"] = "10^8 kWh, fraction, or days"
    heterogeneity["indicator_direction"] = "generation higher_better; storage diagnostic; correction/pressure/violation lower_better"
    heterogeneity.to_csv(output_dir / "fig8_reservoir_heterogeneity.csv", index=False, encoding="utf-8-sig")
    return daily, long_frame


def prepare_fig10(stage_c_score: pd.DataFrame, output_dir: Path) -> None:
    frame = stage_c_score.copy()
    frame["unit"] = "percent delta or raw hydrologic metric"
    frame["indicator_direction"] = "return higher_better; pressure/correction/conditional spill lower_better; total spillage neutral spillage diagnostic"
    risk_cols = [
        "low_level_pressure_days_delta_vs_default_pct",
        "boundary_pressure_spill_delta_vs_default_pct",
        "non_flood_spill_delta_vs_default_pct",
    ]
    frame["risk_dashboard_score"] = frame[risk_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    frame.to_csv(output_dir / "fig10_mixed_sequence_validation.csv", index=False, encoding="utf-8-sig")


def prepare_fig9(config_summary: pd.DataFrame, output_dir: Path) -> None:
    frame = config_summary.copy()
    frame["unit"] = "reward, fraction, days, 10^8 m3"
    frame["indicator_direction"] = "return higher_better; correction/pressure/violation lower_better; total spillage neutral spillage diagnostic"
    frame["action_executability_risk"] = frame["mean_action_correction_mean"]
    frame["constraint_risk"] = frame["any_violation_rate_mean"]
    frame["water_level_pressure"] = frame["low_level_pressure_days_mean"]
    frame["acceptability_score"] = (
        _scale_series(frame["return_mean"])
        + (1.0 - _scale_series(frame["low_level_pressure_days_mean"]))
        + (1.0 - _scale_series(frame["mean_action_correction_mean"]))
        + (1.0 - _scale_series(frame["any_violation_rate_mean"]))
    ) / 4.0
    frame.to_csv(output_dir / "fig9_pareto_diagnostics.csv", index=False, encoding="utf-8-sig")


def prepare_homogeneous_sources(output_dir: Path) -> None:
    """Write long-format source tables used by the homomorphic main figures."""
    # Fig.4: metric-specific lollipop panels.
    fig4 = _read(output_dir / "fig4_parameter_importance.csv").copy()
    fig4_long = fig4.assign(
        figure_id="Fig.4",
        panel_id=fig4["metric"],
        facet_variable=fig4["metric_label"],
        x_value=fig4["spearman"],
        y_value=fig4["parameter"],
        parameter=fig4["parameter"],
        parameter_label=fig4["parameter"].map(
            {
                "learning_rate": "actor LR",
                "critic_learning_rate": "critic LR",
                "clip_ratio": "clip ratio",
                "entropy_weight": "entropy",
                "discount_factor": "discount",
                "gae_lambda": "GAE lambda",
            }
        ),
    )[
        [
            "figure_id",
            "panel_id",
            "facet_variable",
            "x_value",
            "y_value",
            "metric",
            "unit",
            "indicator_direction",
            "config_id",
            "parameter",
            "parameter_label",
            "spearman",
            "abs_spearman",
        ]
    ]
    fig4_long.to_csv(output_dir / "fig4_homogeneous_lollipop.csv", index=False, encoding="utf-8-sig")

    # Fig.5: clip-gamma contour panels.
    fig4 = _read(output_dir / "fig5_clip_gamma_3d.csv").copy()
    fig4_specs = [
        ("return_mean", "return", "reward", "higher_better"),
        ("low_level_pressure_days_mean", "low_level_pressure_days", "reservoir-days", "lower_better"),
        ("mean_action_correction_mean", "mean_action_correction", "normalized action", "lower_better"),
        ("any_violation_rate_mean", "any_violation_rate", "fraction", "lower_better"),
    ]
    rows = []
    for value_col, metric, unit, direction in fig4_specs:
        one = fig4[["config_id", "clip_ratio", "discount_factor", value_col]].copy()
        one = one.rename(columns={value_col: "metric_value"})
        one["figure_id"] = "Fig.5"
        one["panel_id"] = metric
        one["facet_variable"] = metric
        one["x_value"] = one["clip_ratio"]
        one["y_value"] = one["discount_factor"]
        one["metric"] = metric
        one["unit"] = unit
        one["indicator_direction"] = direction
        rows.append(one)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "fig5_homogeneous_contour.csv", index=False, encoding="utf-8-sig")

    # Fig.6: season x metric point-line panels.
    fig5 = _read(output_dir / "fig6_gamma_gae_seasonal_response.csv").copy()
    fig5_specs = [
        ("season_power", "seasonal_generation", "10^8 kWh", "higher_better"),
        ("low_level_pressure_days", "low_level_pressure_days", "reservoir-days", "lower_better"),
        ("season_spill_neutral", "seasonal_spill_diagnostic", "10^8 m3", "neutral_diagnostic"),
    ]
    rows = []
    for value_col, metric, unit, direction in fig5_specs:
        one = fig5[["config_id", "run_name", "seed", "discount_factor", "gae_lambda", "season", value_col]].copy()
        one = one.rename(columns={value_col: "metric_value"})
        one["figure_id"] = "Fig.6"
        one["panel_id"] = metric + "_" + one["season"].astype(str)
        one["facet_variable"] = one["season"]
        one["x_value"] = one["discount_factor"]
        one["y_value"] = one["metric_value"]
        one["metric"] = metric
        one["unit"] = unit
        one["indicator_direction"] = direction
        rows.append(one)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "fig6_homogeneous_seasonal.csv", index=False, encoding="utf-8-sig")

    # Fig.7: representative operation traces.
    fig6 = _read(output_dir / "fig7_mechanism_timeseries.csv").copy()
    fig6_specs = [
        ("normalized_water_state_mean", "water_state", "normalized level", "diagnostic"),
        ("outflow_m3s_sum", "outflow", "m3/s", "diagnostic"),
        ("spill_flow_m3s_sum", "spill_diagnostic", "m3/s", "neutral_diagnostic"),
        ("action_correction_mean", "action_correction", "normalized action", "lower_better"),
        ("low_pressure_reservoir_count", "low_pressure_event", "reservoir count", "lower_better"),
        ("violation_reservoir_count", "violation_event", "reservoir count", "lower_better"),
    ]
    rows = []
    for value_col, metric, unit, direction in fig6_specs:
        one = fig6[["config_id", "stage_c_role", "run_name", "date", "season", value_col]].copy()
        one = one.rename(columns={value_col: "metric_value"})
        one["figure_id"] = "Fig.7"
        one["panel_id"] = one["config_id"]
        one["facet_variable"] = one["config_id"]
        one["x_value"] = one["date"]
        one["y_value"] = one["metric_value"]
        one["metric"] = metric
        one["unit"] = unit
        one["indicator_direction"] = direction
        rows.append(one)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "fig7_homogeneous_timeseries.csv", index=False, encoding="utf-8-sig")

    # Fig.8: reservoir x configuration matrices.
    fig7 = _read(output_dir / "fig8_reservoir_heterogeneity.csv").copy()
    fig7_specs = [
        ("generation_100mkwh_sum", "generation", "10^8 kWh", "higher_better"),
        ("low_pressure_days", "low_pressure_days", "days", "lower_better"),
        ("action_correction_mean", "action_correction", "normalized action", "lower_better"),
        ("remaining_storage_ratio_mean", "remaining_storage", "fraction", "diagnostic"),
    ]
    rows = []
    for value_col, metric, unit, direction in fig7_specs:
        one = fig7[["config_id", "stage_c_role", "reservoir", "reservoir_label", "reservoir_order", value_col]].copy()
        one = one.rename(columns={value_col: "metric_value"})
        one["figure_id"] = "Fig.8"
        one["panel_id"] = metric
        one["facet_variable"] = one["reservoir_label"]
        one["x_value"] = one["config_id"]
        one["y_value"] = one["reservoir_label"]
        one["metric"] = metric
        one["unit"] = unit
        one["indicator_direction"] = direction
        rows.append(one)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "fig8_homogeneous_matrix.csv", index=False, encoding="utf-8-sig")

    # Fig.10: Baseline-relative lollipop panels.
    fig8 = _read(output_dir / "fig10_mixed_sequence_validation.csv").copy()
    fig8_specs = [
        ("eval_return_delta_vs_default_pct", "return_delta", "% vs Baseline", "higher_better"),
        ("low_level_pressure_days_delta_vs_default_pct", "low_pressure_delta", "% vs Baseline", "lower_better"),
        ("boundary_pressure_spill_delta_vs_default_pct", "boundary_pressure_spill_delta", "% vs Baseline", "lower_better"),
        ("non_flood_spill_delta_vs_default_pct", "non_flood_spill_delta", "% vs Baseline", "lower_better"),
    ]
    rows = []
    for value_col, metric, unit, direction in fig8_specs:
        one = fig8[["config_id", "stage_c_role", value_col]].copy()
        one = one.rename(columns={value_col: "metric_value"})
        one["figure_id"] = "Fig.10"
        one["panel_id"] = metric
        one["facet_variable"] = metric
        one["x_value"] = one["metric_value"]
        one["y_value"] = one["config_id"]
        one["metric"] = metric
        one["unit"] = unit
        one["indicator_direction"] = direction
        rows.append(one)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "fig10_homogeneous_lollipop.csv", index=False, encoding="utf-8-sig")

    # Fig.9: homogeneous return-risk scatter projections.
    fig9 = _read(output_dir / "fig9_pareto_diagnostics.csv").copy()
    fig9_specs = [
        ("low_level_pressure_days_mean", "low_level_pressure", "reservoir-days", "lower_better"),
        ("mean_action_correction_mean", "action_correction", "normalized action", "lower_better"),
        ("any_violation_rate_mean", "violation_rate", "fraction", "lower_better"),
        ("boundary_pressure_spill_mean", "boundary_pressure_spill", "10^8 m3", "lower_better"),
    ]
    rows = []
    for value_col, metric, unit, direction in fig9_specs:
        one = fig9[["config_id", "clip_ratio", "discount_factor", "return_mean", value_col]].copy()
        one = one.rename(columns={value_col: "risk_metric_value"})
        one["figure_id"] = "Fig.9"
        one["panel_id"] = metric
        one["facet_variable"] = metric
        one["x_value"] = one["risk_metric_value"]
        one["y_value"] = one["return_mean"]
        one["metric"] = metric
        one["unit"] = unit
        one["indicator_direction"] = direction
        rows.append(one)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "fig9_homogeneous_scatter.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare source data for the manuscript figures.")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    ensure_figure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    response = _read(STAGE_B_OUTPUT_DIR / "stage_b_response_dataset.csv")
    config_summary = _read(STAGE_B_OUTPUT_DIR / "stage_b_config_summary.csv")
    spearman = _read(STAGE_B_OUTPUT_DIR / "stage_b_parameter_spearman.csv")
    interactions = _read(STAGE_B_OUTPUT_DIR / "stage_b_interactions.csv")
    stage_c_score = _read(STAGE_C_OUTPUT_DIR / "stage_c_scenario_scorecard.csv")
    stage_c_index = _read(STAGE_C_OUTPUT_DIR / "stage_c_behavior_timeseries_index.csv")

    prepare_fig1(args.output_dir)
    prepare_fig2(args.output_dir)
    training_curves = prepare_training_curves(response, args.output_dir)
    prepare_fig4(spearman, response, args.output_dir)
    prepare_fig5(interactions, response, args.output_dir)
    daily, reservoir_daily = prepare_fig7_fig8(stage_c_index, args.output_dir)
    prepare_fig1_dynamic_evidence(training_curves, daily, args.output_dir)
    prepare_fig6(response, args.output_dir, reservoir_daily)
    prepare_fig10(stage_c_score, args.output_dir)
    prepare_fig9(config_summary, args.output_dir)
    prepare_homogeneous_sources(args.output_dir)

    outputs = sorted(path.name for path in args.output_dir.glob("fig*.csv"))
    print(f"Prepared {len(outputs)} manuscript figure data files in {args.output_dir}")
    for name in outputs:
        print(f" - {name}")


if __name__ == "__main__":
    main()
