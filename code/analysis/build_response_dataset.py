from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover - optional dependency in some analysis-only environments.
    EventAccumulator = None  # type: ignore[assignment]


CURRENT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = CURRENT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agent.reservoir_multi_agents.analysis.experiment_analysis_paperOne.analysis_core import (  # noqa: E402
    PROJECT_DIR,
    collect_run_summary_records,
    discover_result_roots,
)
from agent.reservoir_multi_agents.analysis.experiment_analysis_paperOne.hydrology_paper_analysis import (  # noqa: E402
    FLOOD_SEASON_MONTHS,
    add_conditional_spill_metrics,
    _flow_sum_to_volume_yi_m3,
    _infer_months,
    _read_export_time_series_frames,
)


LEARNING_PARAMS = (
    "learning_rate",
    "critic_learning_rate",
    "clip_ratio",
    "entropy_weight",
    "discount_factor",
    "gae_lambda",
)

BASE_METRIC_COLUMNS = {
    "return": "result_eval_value",
    "power": "metric_\u603b\u53d1\u7535\u91cf",
    "total_spill_neutral": "metric_\u603b\u5f03\u6c34\u4f53\u79ef",
    "flood_spill_neutral": "metric_\u6c5b\u671f\u5f03\u6c34\u4f53\u79ef",
    "non_flood_spill": "metric_\u975e\u6c5b\u671f\u5f03\u6c34\u4f53\u79ef",
    "spill_rate_neutral": "metric_\u5f03\u6c34\u7387",
    "non_flood_spill_rate": "metric_\u975e\u6c5b\u671f\u5f03\u6c34\u7387",
    "storage_spill_conflict": "metric_\u53ef\u84c4\u5f03\u6c34\u51b2\u7a81\u6307\u6570",
    "boundary_pressure_spill": "metric_\u8fb9\u754c\u538b\u529b\u5f03\u6c34\u4f53\u79ef",
    "low_level_pressure_days": "metric_\u4f4e\u6c34\u4f4d\u538b\u529b\u5929\u6570",
    "navigation_penalty": "metric_\u603b\u822a\u8fd0\u60e9\u7f5a",
}

EXCEL_COLUMNS = {
    "date": "\u65e5\u671f",
    "power": "\u5f53\u65e5\u53d1\u7535\u91cf(\u4ebfkW\xb7h)",
    "spill_flow": "\u5f03\u6c34\u6d41\u91cf(m\xb3/s)",
    "agent_action": "\u667a\u80fd\u4f53\u52a8\u4f5c",
    "actual_action": "\u5b9e\u9645\u52a8\u4f5c",
    "action_correction": "\u52a8\u4f5c\u4fee\u6b63\u5e45\u5ea6",
    "high_margin": "\u8ddd\u9ad8\u9650\u5f52\u4e00\u5316\u88d5\u5ea6",
    "low_margin": "\u8ddd\u4f4e\u9650\u5f52\u4e00\u5316\u88d5\u5ea6",
    "remaining_storage": "\u5269\u4f59\u8c03\u8282\u5e93\u5bb9\u5360\u6bd4",
    "ecology_violation": "\u751f\u6001\u8fdd\u89c4",
    "guarantee_violation": "\u4fdd\u8bc1\u51fa\u529b\u8fdd\u89c4",
    "navigation_violation": "\u822a\u8fd0\u8fdd\u89c4",
}

DRY_SEASON_MONTHS = {1, 2, 3, 4, 5}
RECOVERY_SEASON_MONTHS = {10, 11, 12}
AGENT_COUNT = 5

LEARNER_AGENT_METRICS = {
    "approx_kl": "approx_kl_avg",
    "clipfrac": "clipfrac_avg",
    "entropy_loss": "entropy_loss_avg",
    "value_loss": "value_loss_avg",
    "policy_loss": "policy_loss_avg",
}
SCALAR_STATS = ("final", "mean", "trend")
AGENT_AGGREGATIONS = ("agent_mean", "agent_max", "agent_std")


def _learning_dynamic_columns() -> tuple[str, ...]:
    columns = [
        "ld_eval_return_final",
        "ld_eval_return_mean",
        "ld_eval_return_trend",
    ]
    for metric_name in LEARNER_AGENT_METRICS:
        for stat_name in SCALAR_STATS:
            for aggregation in AGENT_AGGREGATIONS:
                columns.append(f"ld_{metric_name}_{stat_name}_{aggregation}")
    return tuple(columns)


EXECUTION_METRICS = (
    "mean_action_correction",
    "max_action_correction",
    "action_correction_rate",
    "mean_action_volatility",
    "mean_agent_action_volatility",
    "ecology_violation_rate",
    "guarantee_violation_rate",
    "navigation_violation_rate",
    "any_violation_rate",
    "high_limit_margin_mean",
    "high_limit_margin_min",
    "low_limit_margin_mean",
    "low_limit_margin_min",
    "remaining_storage_mean",
    "dry_season_power",
    "flood_season_power",
    "recovery_season_power",
    "dry_season_spill_neutral",
    "flood_season_spill_neutral",
    "recovery_season_spill_neutral",
)

LEARNING_DYNAMIC_METRICS = _learning_dynamic_columns()

RISK_METRICS = (
    "non_flood_spill",
    "storage_spill_conflict",
    "boundary_pressure_spill",
    "low_level_pressure_days",
    "mean_action_correction",
    "any_violation_rate",
)

SUMMARY_METRICS = (
    "return",
    "power",
    "total_spill_neutral",
    "flood_spill_neutral",
    "non_flood_spill",
    "spill_rate_neutral",
    "non_flood_spill_rate",
    "storage_spill_conflict",
    "boundary_pressure_spill",
    "low_level_pressure_days",
    "navigation_penalty",
    *EXECUTION_METRICS,
    *LEARNING_DYNAMIC_METRICS,
)

SENSITIVITY_RESPONSE_METRICS = (
    "return",
    "power",
    "non_flood_spill",
    "boundary_pressure_spill",
    "low_level_pressure_days",
    "mean_action_correction",
    "action_correction_rate",
    "any_violation_rate",
    "ld_eval_return_final",
    "ld_approx_kl_mean_agent_mean",
    "ld_clipfrac_mean_agent_mean",
    "ld_entropy_loss_mean_agent_mean",
    "ld_value_loss_mean_agent_mean",
    "ld_policy_loss_mean_agent_mean",
)

INTERACTION_PAIRS = (
    ("clip_ratio", "discount_factor"),
    ("discount_factor", "gae_lambda"),
    ("entropy_weight", "discount_factor"),
)

STAGE_C_RECOMMENDED_CONFIGS = {
    "default": "default_reference",
    "gamma099_clip02": "robust_top_and_best_return",
    "return_lhs008": "high_return_low_clip_candidate",
    "return_lhs012": "stage_a_return_leader_check",
    "clip03_gamma995": "high_clip_risk_contrast",
    "gamma0999_clip02": "long_horizon_risk_contrast",
    "risk_lhs009": "failure_counterexample",
    "entropy004_robust": "high_screening_but_high_cv_uncertainty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PaperFour response dataset from manifests, summaries, hydrology diagnostics, and learner dynamics."
    )
    parser.add_argument(
        "result_roots",
        nargs="*",
        help="Result roots to scan. Defaults to auto-discovering result* directories under reservoir_multi_agents.",
    )
    parser.add_argument("--base-dir", type=Path, default=PROJECT_DIR, help=f"Base directory. Default: {PROJECT_DIR}")
    parser.add_argument("--stage", default="B", help="Manifest stage to keep. Default: B")
    parser.add_argument("--data-type", choices=["collector", "evaluator"], default="evaluator")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "output",
        help="Directory for response dataset outputs.",
    )
    parser.add_argument(
        "--return-tolerance-ratio",
        type=float,
        default=0.05,
        help="Allowed return degradation versus default for robust flags.",
    )
    parser.add_argument(
        "--risk-tolerance-ratio",
        type=float,
        default=0.10,
        help="Allowed directional-risk degradation versus default for robust flags.",
    )
    parser.add_argument(
        "--cv-tolerance-ratio",
        type=float,
        default=0.10,
        help="Maximum accepted return CV for robust flags.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include incomplete runs if they have a manifest. Default keeps complete runs only.",
    )
    return parser.parse_args()


def read_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _to_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_values(values: list[float] | pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _nanmean(values: list[float] | pd.Series | np.ndarray) -> float:
    finite = _finite_values(values)
    return float(finite.mean()) if finite.size else math.nan


def _nanmax(values: list[float] | pd.Series | np.ndarray) -> float:
    finite = _finite_values(values)
    return float(finite.max()) if finite.size else math.nan


def _nanmin(values: list[float] | pd.Series | np.ndarray) -> float:
    finite = _finite_values(values)
    return float(finite.min()) if finite.size else math.nan


def _nanstd(values: list[float] | pd.Series | np.ndarray) -> float:
    finite = _finite_values(values)
    return float(finite.std(ddof=1)) if finite.size > 1 else math.nan


def _scalar_series_stats(values: list[float]) -> dict[str, float]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {stat: math.nan for stat in SCALAR_STATS}
    if finite.size == 1:
        trend = math.nan
    else:
        x_axis = np.linspace(0.0, 1.0, finite.size)
        trend = float(np.polyfit(x_axis, finite, 1)[0])
    return {
        "final": float(finite[-1]),
        "mean": float(finite.mean()),
        "trend": trend,
    }


def _read_scalar_values(accumulator: Any, *candidate_tags: str) -> list[float]:
    tags = set(accumulator.Tags().get("scalars", []))
    for tag in candidate_tags:
        if tag not in tags:
            continue
        try:
            return [float(event.value) for event in accumulator.Scalars(tag)]
        except Exception:
            return []
    return []


def extract_learning_dynamics(run_dir: Path) -> dict[str, float]:
    """Extract final/mean/trend learner signals from TensorBoard scalars."""
    records: dict[str, float] = {column: math.nan for column in LEARNING_DYNAMIC_METRICS}
    records["ld_tensorboard_available"] = 0.0
    records["ld_agent_metric_available_count"] = 0.0
    if EventAccumulator is None:
        return records

    serial_dir = run_dir / "log" / "serial"
    if not serial_dir.exists():
        return records

    try:
        accumulator = EventAccumulator(str(serial_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
    except Exception:
        return records

    records["ld_tensorboard_available"] = 1.0
    eval_stats = _scalar_series_stats(
        _read_scalar_values(
            accumulator,
            "evaluator_iter/eval_episode_return_mean",
            "evaluator_step/eval_episode_return_mean",
            "evaluator_iter/reward_mean",
            "evaluator_step/reward_mean",
        )
    )
    for stat_name, value in eval_stats.items():
        records[f"ld_eval_return_{stat_name}"] = value

    available_count = 0
    for metric_name, tag_suffix in LEARNER_AGENT_METRICS.items():
        per_stat_values: dict[str, list[float]] = {stat_name: [] for stat_name in SCALAR_STATS}
        for agent_idx in range(AGENT_COUNT):
            stats = _scalar_series_stats(
                _read_scalar_values(
                    accumulator,
                    f"learner_iter/agent{agent_idx}_{tag_suffix}",
                    f"learner_step/agent{agent_idx}_{tag_suffix}",
                )
            )
            if pd.notna(stats["mean"]):
                available_count += 1
            for stat_name, value in stats.items():
                per_stat_values[stat_name].append(value)
        for stat_name, values in per_stat_values.items():
            records[f"ld_{metric_name}_{stat_name}_agent_mean"] = _nanmean(values)
            records[f"ld_{metric_name}_{stat_name}_agent_max"] = _nanmax(values)
            records[f"ld_{metric_name}_{stat_name}_agent_std"] = _nanstd(values)
    records["ld_agent_metric_available_count"] = float(available_count)
    return records


def compute_execution_metrics(frames: list[pd.DataFrame]) -> dict[str, float]:
    """Compute action, violation, margin, and seasonal diagnostics from evaluator exports."""
    metrics: dict[str, float] = {metric: math.nan for metric in EXECUTION_METRICS}
    metrics["execution_metric_available"] = 0.0
    if not frames:
        return metrics

    action_corrections: list[float] = []
    action_correction_flags: list[float] = []
    actual_action_volatility: list[float] = []
    agent_action_volatility: list[float] = []
    ecology_violations: list[float] = []
    guarantee_violations: list[float] = []
    navigation_violations: list[float] = []
    any_violations: list[float] = []
    high_margins: list[float] = []
    low_margins: list[float] = []
    remaining_storage: list[float] = []

    dry_power = 0.0
    flood_power = 0.0
    recovery_power = 0.0
    dry_spill_flow = 0.0
    flood_spill_flow = 0.0
    recovery_spill_flow = 0.0

    for frame in frames:
        if frame.empty:
            continue

        months = _infer_months(frame)
        dry_mask = months.isin(DRY_SEASON_MONTHS)
        flood_mask = months.isin(FLOOD_SEASON_MONTHS)
        recovery_mask = months.isin(RECOVERY_SEASON_MONTHS)

        correction = _to_numeric_series(frame, EXCEL_COLUMNS["action_correction"]).abs()
        action_corrections.extend(correction.dropna().astype(float).tolist())
        action_correction_flags.extend((correction > 1e-9).dropna().astype(float).tolist())

        actual_action = _to_numeric_series(frame, EXCEL_COLUMNS["actual_action"])
        actual_action_volatility.extend(actual_action.diff().abs().dropna().astype(float).tolist())

        agent_action = _to_numeric_series(frame, EXCEL_COLUMNS["agent_action"])
        agent_action_volatility.extend(agent_action.diff().abs().dropna().astype(float).tolist())

        ecology = _to_numeric_series(frame, EXCEL_COLUMNS["ecology_violation"]).fillna(0.0)
        guarantee = _to_numeric_series(frame, EXCEL_COLUMNS["guarantee_violation"]).fillna(0.0)
        navigation = _to_numeric_series(frame, EXCEL_COLUMNS["navigation_violation"]).fillna(0.0)
        ecology_violations.extend((ecology > 0).astype(float).tolist())
        guarantee_violations.extend((guarantee > 0).astype(float).tolist())
        navigation_violations.extend((navigation > 0).astype(float).tolist())
        any_violations.extend(((ecology > 0) | (guarantee > 0) | (navigation > 0)).astype(float).tolist())

        high_margin = _to_numeric_series(frame, EXCEL_COLUMNS["high_margin"])
        low_margin = _to_numeric_series(frame, EXCEL_COLUMNS["low_margin"])
        remaining = _to_numeric_series(frame, EXCEL_COLUMNS["remaining_storage"])
        high_margins.extend(high_margin.dropna().astype(float).tolist())
        low_margins.extend(low_margin.dropna().astype(float).tolist())
        remaining_storage.extend(remaining.dropna().astype(float).tolist())

        power = _to_numeric_series(frame, EXCEL_COLUMNS["power"]).fillna(0.0)
        spill = _to_numeric_series(frame, EXCEL_COLUMNS["spill_flow"]).clip(lower=0.0).fillna(0.0)
        dry_power += float(power[dry_mask].sum())
        flood_power += float(power[flood_mask].sum())
        recovery_power += float(power[recovery_mask].sum())
        dry_spill_flow += float(spill[dry_mask].sum())
        flood_spill_flow += float(spill[flood_mask].sum())
        recovery_spill_flow += float(spill[recovery_mask].sum())

    metrics.update(
        {
            "execution_metric_available": 1.0,
            "mean_action_correction": _nanmean(action_corrections),
            "max_action_correction": _nanmax(action_corrections),
            "action_correction_rate": _nanmean(action_correction_flags),
            "mean_action_volatility": _nanmean(actual_action_volatility),
            "mean_agent_action_volatility": _nanmean(agent_action_volatility),
            "ecology_violation_rate": _nanmean(ecology_violations),
            "guarantee_violation_rate": _nanmean(guarantee_violations),
            "navigation_violation_rate": _nanmean(navigation_violations),
            "any_violation_rate": _nanmean(any_violations),
            "high_limit_margin_mean": _nanmean(high_margins),
            "high_limit_margin_min": _nanmin(high_margins),
            "low_limit_margin_mean": _nanmean(low_margins),
            "low_limit_margin_min": _nanmin(low_margins),
            "remaining_storage_mean": _nanmean(remaining_storage),
            "dry_season_power": dry_power,
            "flood_season_power": flood_power,
            "recovery_season_power": recovery_power,
            "dry_season_spill_neutral": _flow_sum_to_volume_yi_m3(dry_spill_flow),
            "flood_season_spill_neutral": _flow_sum_to_volume_yi_m3(flood_spill_flow),
            "recovery_season_spill_neutral": _flow_sum_to_volume_yi_m3(recovery_spill_flow),
        }
    )
    return metrics


def build_response_frame(
    run_frame: pd.DataFrame,
    stage: str,
    data_type: str,
    include_incomplete: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    export_key = f"latest_{data_type}_export"
    for _, row in run_frame.iterrows():
        run_dir = Path(str(row.get("run_dir", "")))
        manifest = read_manifest(run_dir)
        if not manifest:
            continue
        if str(manifest.get("stage", "")).upper() != stage.upper():
            continue
        has_ckpt_best = (run_dir / "ckpt" / "ckpt_best.pth.tar").exists()
        if not include_incomplete and (row.get("status") != "complete" or not has_ckpt_best):
            continue

        changed_params = manifest.get("changed_params", {})
        record: dict[str, Any] = {
            "run_name": row.get("run_name"),
            "config_id": manifest.get("config_id"),
            "stage": manifest.get("stage"),
            "seed": manifest.get("seed"),
            "round": manifest.get("round"),
            "run_dir": str(run_dir.resolve()),
            "result_root_name": row.get("result_root_name"),
            "status": row.get("status"),
            "latest_evaluator_episode": row.get("latest_evaluator_episode"),
            "latest_collector_episode": row.get("latest_collector_episode"),
            "has_manifest": True,
            "has_total_config": row.get("has_total_config"),
            "has_ckpt_best": has_ckpt_best,
            "has_evaluator_export": row.get("has_evaluator_export"),
        }
        for param in LEARNING_PARAMS:
            record[param] = _as_float(changed_params.get(param))
        for output_name, source_col in BASE_METRIC_COLUMNS.items():
            record[output_name] = _as_float(row.get(source_col))

        frames = _read_export_time_series_frames(row.get(export_key))
        record.update(compute_execution_metrics(frames))
        record.update(extract_learning_dynamics(run_dir))
        records.append(record)

    response = pd.DataFrame(records)
    if response.empty:
        return response

    numeric_columns = list(LEARNING_PARAMS) + list(BASE_METRIC_COLUMNS.keys()) + list(EXECUTION_METRICS)
    numeric_columns.extend(["execution_metric_available", "ld_tensorboard_available", "ld_agent_metric_available_count"])
    numeric_columns.extend(LEARNING_DYNAMIC_METRICS)
    for column in numeric_columns:
        if column in response.columns:
            response[column] = pd.to_numeric(response[column], errors="coerce")
    return response.sort_values(["config_id", "seed", "run_name"]).reset_index(drop=True)


def _cv(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.shape[0] < 2:
        return math.nan
    mean = float(values.mean())
    if mean == 0:
        return math.nan
    return float(values.std(ddof=1) / abs(mean))


def build_config_summary(
    response: pd.DataFrame,
    return_tolerance: float,
    risk_tolerance: float,
    cv_tolerance: float,
) -> pd.DataFrame:
    if response.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for config_id, group in response.groupby("config_id", dropna=False):
        record: dict[str, Any] = {
            "config_id": config_id,
            "run_count": int(len(group)),
            "seed_count": int(group["seed"].dropna().nunique()) if "seed" in group else 0,
        }
        for param in LEARNING_PARAMS:
            record[param] = group[param].dropna().iloc[0] if param in group and group[param].notna().any() else math.nan
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            record[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            record[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
            record[f"{metric}_min"] = float(values.min()) if values.notna().any() else math.nan
            record[f"{metric}_max"] = float(values.max()) if values.notna().any() else math.nan
            record[f"{metric}_cv"] = _cv(values)
        records.append(record)

    summary = pd.DataFrame(records)
    summary = add_screening_columns(summary, return_tolerance, risk_tolerance, cv_tolerance)
    return summary.sort_values(["screening_rank", "config_id"]).reset_index(drop=True)


def _minmax(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(1.0, index=series.index)
    normalized = (values - values.min()) / span
    return normalized if higher_is_better else 1.0 - normalized


def _threshold_from_default(default_value: float, tolerance_ratio: float, higher_is_better: bool) -> float:
    if pd.isna(default_value):
        return math.nan
    tolerance = max(abs(float(default_value)) * tolerance_ratio, 1e-9)
    return default_value - tolerance if higher_is_better else default_value + tolerance


def add_screening_columns(
    summary: pd.DataFrame,
    return_tolerance: float,
    risk_tolerance: float,
    cv_tolerance: float,
) -> pd.DataFrame:
    enriched = summary.copy()
    if enriched.empty:
        return enriched

    enriched["return_score"] = _minmax(enriched.get("return_mean"), higher_is_better=True)
    enriched["power_score"] = _minmax(enriched.get("power_mean"), higher_is_better=True)
    enriched["non_flood_spill_score"] = _minmax(enriched.get("non_flood_spill_mean"), higher_is_better=False)
    enriched["low_level_pressure_score"] = _minmax(
        enriched.get("low_level_pressure_days_mean"), higher_is_better=False
    )
    enriched["boundary_pressure_spill_score"] = _minmax(
        enriched.get("boundary_pressure_spill_mean"), higher_is_better=False
    )
    enriched["action_correction_score"] = _minmax(
        enriched.get("mean_action_correction_mean"), higher_is_better=False
    )
    enriched["violation_score"] = _minmax(enriched.get("any_violation_rate_mean"), higher_is_better=False)
    enriched["screening_score"] = (
        0.38 * enriched["return_score"]
        + 0.16 * enriched["power_score"]
        + 0.12 * enriched["non_flood_spill_score"]
        + 0.10 * enriched["low_level_pressure_score"]
        + 0.08 * enriched["boundary_pressure_spill_score"]
        + 0.08 * enriched["action_correction_score"]
        + 0.08 * enriched["violation_score"]
    )
    enriched["screening_rank"] = enriched["screening_score"].rank(ascending=False, method="min")
    enriched["return_rank"] = enriched["return_mean"].rank(ascending=False, method="min")

    default_rows = enriched[enriched["config_id"] == "default"]
    if default_rows.empty:
        enriched["robust_flag"] = False
        enriched["robust_reason"] = "default_missing"
        return enriched

    default_row = default_rows.iloc[0]
    return_threshold = _threshold_from_default(default_row.get("return_mean"), return_tolerance, True)
    enriched["robust_flag"] = enriched["return_mean"] >= return_threshold
    reasons: list[str] = []
    for idx, row in enriched.iterrows():
        reason_parts = []
        if not bool(enriched.at[idx, "robust_flag"]):
            reason_parts.append("return_below_default_tolerance")
        if pd.notna(row.get("return_cv")) and row.get("return_cv") > cv_tolerance:
            enriched.at[idx, "robust_flag"] = False
            reason_parts.append("return_cv_above_tolerance")
        for metric in RISK_METRICS:
            col = f"{metric}_mean"
            threshold = _threshold_from_default(default_row.get(col), risk_tolerance, False)
            if pd.notna(threshold) and pd.notna(row.get(col)) and row.get(col) > threshold:
                enriched.at[idx, "robust_flag"] = False
                reason_parts.append(f"{metric}_above_default_tolerance")
        reasons.append(";".join(reason_parts) if reason_parts else "within_default_tolerance")
    enriched["robust_reason"] = reasons
    return enriched


def build_robust_interval(config_summary: pd.DataFrame) -> pd.DataFrame:
    robust = config_summary[config_summary.get("robust_flag", False) == True]  # noqa: E712
    records: list[dict[str, Any]] = []
    for param in LEARNING_PARAMS:
        values = pd.to_numeric(robust.get(param, pd.Series(dtype=float)), errors="coerce").dropna()
        records.append(
            {
                "parameter": param,
                "robust_config_count": int(robust.shape[0]),
                "min_value": float(values.min()) if not values.empty else math.nan,
                "max_value": float(values.max()) if not values.empty else math.nan,
                "unique_values": ";".join(format(v, ".12g") for v in sorted(values.unique())),
            }
        )
    return pd.DataFrame(records)


def build_representatives(config_summary: pd.DataFrame) -> pd.DataFrame:
    if config_summary.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    def add(role: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            return
        row = frame.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        row["representative_role"] = role
        records.append(row)

    def add_by_id(role: str, config_id: str) -> bool:
        frame = config_summary[config_summary["config_id"] == config_id]
        if frame.empty:
            return False
        row = frame.iloc[0].to_dict()
        row["representative_role"] = role
        records.append(row)
        return True

    risk_frame = config_summary[config_summary.get("robust_flag", False) == False]  # noqa: E712
    high_cv_frame = config_summary[pd.to_numeric(config_summary.get("return_cv"), errors="coerce") > 0.10]

    add("default", config_summary[config_summary["config_id"] == "default"], ["config_id"], [True])
    add("best_return", config_summary, ["return_mean"], [False])
    add("top_screening", config_summary, ["screening_score"], [False])
    add(
        "robust_top",
        config_summary[config_summary.get("robust_flag", False) == True],  # noqa: E712
        ["screening_score"],
        [False],
    )
    add("risk_low_return", risk_frame, ["return_mean"], [True])
    add("risk_low_level_pressure", risk_frame, ["low_level_pressure_days_mean"], [False])
    add("risk_non_flood_spill", risk_frame, ["non_flood_spill_mean", "return_mean"], [False, True])
    add("risk_high_cv", high_cv_frame, ["return_cv"], [False])
    if not add_by_id("risk_long_horizon", "gamma0999_clip02"):
        add(
            "risk_long_horizon",
            risk_frame[pd.to_numeric(risk_frame.get("discount_factor"), errors="coerce") >= 0.999],
            ["screening_score"],
            [True],
        )
    if not add_by_id("risk_high_clip", "clip03_gamma995"):
        add(
            "risk_high_clip",
            risk_frame[pd.to_numeric(risk_frame.get("clip_ratio"), errors="coerce") >= 0.3],
            ["screening_score"],
            [True],
        )

    return pd.DataFrame(records)


def build_stage_c_candidates(config_summary: pd.DataFrame) -> pd.DataFrame:
    if config_summary.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    indexed = config_summary.set_index("config_id", drop=False)
    for config_id, role in STAGE_C_RECOMMENDED_CONFIGS.items():
        if config_id not in indexed.index:
            continue
        row = indexed.loc[config_id].to_dict()
        row["stage_c_role"] = role
        row["stage_c_recommended"] = True
        records.append(row)
    return pd.DataFrame(records)


def build_parameter_spearman(response: pd.DataFrame, config_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add_records(level: str, frame: pd.DataFrame, metric_names: tuple[str, ...], suffix: str = "") -> None:
        for param in LEARNING_PARAMS:
            for metric in metric_names:
                metric_col = f"{metric}{suffix}"
                if param not in frame or metric_col not in frame:
                    continue
                values = frame[[param, metric_col]].apply(pd.to_numeric, errors="coerce").dropna()
                corr = float(values.corr(method="spearman").iloc[0, 1]) if len(values) >= 2 else math.nan
                records.append(
                    {
                        "level": level,
                        "parameter": param,
                        "metric": metric,
                        "spearman": corr,
                        "n": int(len(values)),
                    }
                )

    add_records("run", response, SENSITIVITY_RESPONSE_METRICS)
    add_records("config", config_summary, SENSITIVITY_RESPONSE_METRICS, suffix="_mean")
    return pd.DataFrame(records)


def build_main_effects(response: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for param in LEARNING_PARAMS:
        if param not in response:
            continue
        for value, group in response.groupby(param, dropna=True):
            record: dict[str, Any] = {
                "parameter": param,
                "parameter_value": value,
                "n": int(len(group)),
            }
            for metric in SENSITIVITY_RESPONSE_METRICS:
                if metric not in group:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce")
                record[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else math.nan
                record[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
            records.append(record)
    return pd.DataFrame(records)


def build_interactions(response: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for param_a, param_b in INTERACTION_PAIRS:
        if param_a not in response or param_b not in response:
            continue
        for (value_a, value_b), group in response.groupby([param_a, param_b], dropna=True):
            record: dict[str, Any] = {
                "interaction": f"{param_a} x {param_b}",
                "parameter_a": param_a,
                "parameter_b": param_b,
                "value_a": value_a,
                "value_b": value_b,
                "n": int(len(group)),
            }
            for metric in SENSITIVITY_RESPONSE_METRICS:
                if metric not in group:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce")
                record[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else math.nan
                record[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
            records.append(record)
    return pd.DataFrame(records)


def build_acceptance_matrix(
    config_summary: pd.DataFrame,
    return_tolerance: float,
    risk_tolerance: float,
    cv_tolerance: float,
) -> pd.DataFrame:
    if config_summary.empty:
        return pd.DataFrame()
    default_rows = config_summary[config_summary["config_id"] == "default"]
    if default_rows.empty:
        return pd.DataFrame()
    default_row = default_rows.iloc[0]
    return_threshold = _threshold_from_default(default_row.get("return_mean"), return_tolerance, True)

    records: list[dict[str, Any]] = []
    for _, row in config_summary.iterrows():
        record = {
            "config_id": row.get("config_id"),
            "seed_count": row.get("seed_count"),
            "return_mean": row.get("return_mean"),
            "return_cv": row.get("return_cv"),
            "screening_rank": row.get("screening_rank"),
            "robust_flag": row.get("robust_flag"),
            "robust_reason": row.get("robust_reason"),
            "pass_seed_count": row.get("seed_count") >= 3,
            "pass_return": row.get("return_mean") >= return_threshold if pd.notna(return_threshold) else False,
            "pass_return_cv": row.get("return_cv") <= cv_tolerance if pd.notna(row.get("return_cv")) else False,
        }
        for metric in RISK_METRICS:
            col = f"{metric}_mean"
            threshold = _threshold_from_default(default_row.get(col), risk_tolerance, False)
            record[f"pass_{metric}"] = (
                row.get(col) <= threshold if pd.notna(threshold) and pd.notna(row.get(col)) else False
            )
        records.append(record)
    return pd.DataFrame(records)


def build_stage_inventory(response: pd.DataFrame, stage: str) -> pd.DataFrame:
    columns = [
        "run_name",
        "config_id",
        "stage",
        "seed",
        "round",
        "status",
        "has_manifest",
        "has_total_config",
        "has_ckpt_best",
        "has_evaluator_export",
        "latest_evaluator_episode",
        "execution_metric_available",
        "ld_tensorboard_available",
        "run_dir",
    ]
    available_columns = [column for column in columns if column in response.columns]
    inventory = response[available_columns].copy() if available_columns else pd.DataFrame()
    if not inventory.empty:
        inventory["stage_complete_gate"] = (
            (inventory.get("stage").astype(str).str.upper() == stage.upper())
            & (inventory.get("status") == "complete")
            & (inventory.get("has_manifest") == True)  # noqa: E712
            & (inventory.get("has_total_config") == True)  # noqa: E712
            & (inventory.get("has_ckpt_best") == True)  # noqa: E712
            & (inventory.get("has_evaluator_export") == True)  # noqa: E712
        )
    return inventory


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _format_float(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    output_dir: Path,
    stage: str,
    response: pd.DataFrame,
    config_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    robust_interval: pd.DataFrame,
    stage_c_candidates: pd.DataFrame,
) -> None:
    report_path = output_dir / f"stage_{stage.lower()}_response_report.md"
    seed_counts = (
        config_summary["seed_count"].value_counts().sort_index().to_dict() if "seed_count" in config_summary else {}
    )
    storage_conflict_max = (
        pd.to_numeric(response.get("storage_spill_conflict"), errors="coerce").max() if not response.empty else math.nan
    )
    action_available = (
        int(pd.to_numeric(response.get("execution_metric_available"), errors="coerce").fillna(0).sum())
        if "execution_metric_available" in response
        else 0
    )
    learner_available = (
        int(pd.to_numeric(response.get("ld_tensorboard_available"), errors="coerce").fillna(0).sum())
        if "ld_tensorboard_available" in response
        else 0
    )

    lines = [
        f"# PaperFour Stage {stage.upper()} Response Dataset Report",
        "",
        "## Coverage",
        "",
        f"- Run-level records: {len(response)}",
        f"- Config-level records: {len(config_summary)}",
        f"- Complete run records: {int((response.get('status') == 'complete').sum()) if not response.empty else 0}",
        f"- Seed-count distribution: {seed_counts}",
        f"- Execution metrics available: {action_available}/{len(response)}",
        f"- Learner TensorBoard metrics available: {learner_available}/{len(response)}",
        "",
        "## Notes",
        "",
        "- Total spillage and flood-season spillage are neutral diagnostics, not robust-risk constraints.",
        "- Acceptability flags use evaluator return, return CV, non-flood spillage, storage-spill conflict, boundary-pressure spillage, lower-bound water-level pressure days, action correction, and violation rate. Legacy output columns named robust_flag or robust_reason should be read as sampled acceptability fields.",
        "- Stage C should only start after representative roles are stable across seeds.",
    ]
    if pd.notna(storage_conflict_max) and float(storage_conflict_max) == 0.0:
        lines.append("- Storage-spill conflict is zero for all Stage B records; report it as not triggered rather than interpreting a main effect.")
    lines.append("")

    if not config_summary.empty:
        top_return = config_summary.sort_values("return_mean", ascending=False).iloc[0]
        top_screening = config_summary.sort_values("screening_score", ascending=False).iloc[0]
        default_rows = config_summary[config_summary["config_id"] == "default"]
        lines.extend(
            [
                "## Key Results",
                "",
                f"- Best return: {top_return.get('config_id')} (return_mean={_format_float(top_return.get('return_mean'))}).",
                f"- Top screening: {top_screening.get('config_id')} (screening_score={_format_float(top_screening.get('screening_score'))}).",
            ]
        )
        if not default_rows.empty:
            default = default_rows.iloc[0]
            lines.append(
                f"- Default: return_rank={_format_float(default.get('return_rank'), 0)}, "
                f"screening_rank={_format_float(default.get('screening_rank'), 0)}, "
                f"return_mean={_format_float(default.get('return_mean'))}."
            )
        lines.append("")

    if not robust_interval.empty:
        lines.extend(["## Sampled Acceptable Values", ""])
        for _, row in robust_interval.iterrows():
            lines.append(
                f"- {row.get('parameter')}: {row.get('unique_values')} "
                f"(n={int(row.get('robust_config_count', 0))})"
            )
        lines.append("")

    if not representatives.empty:
        lines.extend(["## Representatives", ""])
        for _, row in representatives.iterrows():
            lines.append(
                f"- {row.get('representative_role')}: {row.get('config_id')} "
                f"(return_mean={_format_float(row.get('return_mean'))}, "
                f"screening_score={_format_float(row.get('screening_score'))}, "
                f"robust={row.get('robust_flag')})"
            )
        lines.append("")

    if not stage_c_candidates.empty:
        lines.extend(["## Stage C Candidates", ""])
        for _, row in stage_c_candidates.iterrows():
            lines.append(
                f"- {row.get('config_id')}: {row.get('stage_c_role')} "
                f"(return_mean={_format_float(row.get('return_mean'))}, "
                f"return_cv={_format_float(row.get('return_cv'))})"
            )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_roots = discover_result_roots(args.result_roots, base_dir=args.base_dir)
    if not result_roots:
        raise SystemExit("No result roots found.")

    run_records, _ = collect_run_summary_records(
        result_roots,
        data_type=args.data_type,
        include_config=True,
    )
    run_frame = pd.DataFrame(run_records)
    run_frame = add_conditional_spill_metrics(run_frame, args.data_type)

    response = build_response_frame(run_frame, args.stage, args.data_type, args.include_incomplete)
    config_summary = build_config_summary(
        response,
        args.return_tolerance_ratio,
        args.risk_tolerance_ratio,
        args.cv_tolerance_ratio,
    )
    robust_interval = build_robust_interval(config_summary)
    representatives = build_representatives(config_summary)
    stage_c_candidates = build_stage_c_candidates(config_summary)
    parameter_spearman = build_parameter_spearman(response, config_summary)
    main_effects = build_main_effects(response)
    interactions = build_interactions(response)
    stage_inventory = build_stage_inventory(response, args.stage)
    acceptance_matrix = build_acceptance_matrix(
        config_summary,
        args.return_tolerance_ratio,
        args.risk_tolerance_ratio,
        args.cv_tolerance_ratio,
    )

    output_dir = args.output_dir
    stage_label = args.stage.lower()
    write_csv(response, output_dir / f"stage_{stage_label}_response_dataset.csv")
    write_csv(stage_inventory, output_dir / f"stage_{stage_label}_run_inventory.csv")
    write_csv(config_summary, output_dir / f"stage_{stage_label}_config_summary.csv")
    write_csv(robust_interval, output_dir / f"stage_{stage_label}_robust_interval.csv")
    write_csv(representatives, output_dir / f"stage_{stage_label}_representative_configs.csv")
    write_csv(stage_c_candidates, output_dir / f"stage_{stage_label}_stage_c_candidates.csv")
    write_csv(parameter_spearman, output_dir / f"stage_{stage_label}_parameter_spearman.csv")
    write_csv(main_effects, output_dir / f"stage_{stage_label}_main_effects.csv")
    write_csv(interactions, output_dir / f"stage_{stage_label}_interactions.csv")
    write_csv(acceptance_matrix, output_dir / f"stage_{stage_label}_acceptance_matrix.csv")
    write_report(
        output_dir,
        args.stage,
        response,
        config_summary,
        representatives,
        robust_interval,
        stage_c_candidates,
    )

    print(f"Stage {args.stage.upper()} response rows: {len(response)}")
    print(f"Stage {args.stage.upper()} config rows: {len(config_summary)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
