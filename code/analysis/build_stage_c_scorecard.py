#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build PaperFour Stage C scenario scorecards from offline evaluation outputs.

The script first scans ``result_paperFour_eval`` for ``eval_metadata.json`` files
written by ``PaperFour_run_experiments_eval.py``.  If hydrology analysis outputs
are available, it merges the evaluator return with conditional spill,
generation, pressure, and violation diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
AGENT_ROOT = MODULE_DIR.parents[1]
DEFAULT_EVAL_ROOT = AGENT_ROOT / "result_paperFour_eval"
DEFAULT_HYDROLOGY_ROOT = MODULE_DIR / "output_stage_c" / "hydrology"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output_stage_c"


LEARNING_PARAM_NAMES = [
    "learning_rate",
    "critic_learning_rate",
    "clip_ratio",
    "entropy_weight",
    "discount_factor",
    "gae_lambda",
]

HYDROLOGY_COLUMN_MAP = {
    "metric_总发电量": "generation",
    "metric_年均发电量": "annual_generation",
    "metric_总全局奖励": "global_reward",
    "metric_总航运惩罚": "navigation_penalty",
    "metric_总弃水体积": "total_spill_neutral",
    "metric_汛期弃水体积": "flood_spill_neutral",
    "metric_非汛期弃水体积": "non_flood_spill",
    "metric_弃水率": "spill_rate_neutral",
    "metric_非汛期弃水率": "non_flood_spill_rate",
    "metric_可蓄弃水冲突指数": "storage_spill_conflict",
    "metric_边界压力弃水体积": "boundary_pressure_spill",
    "metric_低水位压力天数": "low_level_pressure_days",
    "metric_调度天数": "decision_days",
}


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_evaluator_export(run_dir: Path) -> str:
    export_root = run_dir / "schedule_exports" / "evaluator"
    if not export_root.is_dir():
        return ""
    files = sorted(export_root.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else ""


def _metadata_rows(eval_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metadata_path in sorted(eval_root.rglob("eval_metadata.json")):
        metadata = _load_json(metadata_path)
        run_dir = metadata_path.parent
        row = {
            "run_name": metadata.get("run_name") or run_dir.name,
            "status": metadata.get("status", ""),
            "scenario_id": metadata.get("scenario_id", ""),
            "scenario_type": metadata.get("scenario_type", ""),
            "scenario_description": metadata.get("scenario_description", ""),
            "source_years": metadata.get("source_years", ""),
            "config_id": metadata.get("config_id", ""),
            "stage_c_role": metadata.get("stage_c_role", ""),
            "selection_policy": metadata.get("selection_policy", ""),
            "source_run_name": metadata.get("source_run_name", ""),
            "source_seed": metadata.get("source_seed", ""),
            "source_return": _to_float(metadata.get("source_return")),
            "eval_return_mean": _to_float(metadata.get("eval_episode_return_mean")),
            "eval_return_std": _to_float(metadata.get("eval_episode_return_std")),
            "eval_seed": metadata.get("eval_seed", ""),
            "eval_inflow_file": metadata.get("eval_inflow_file", ""),
            "eval_output_dir": metadata.get("eval_output_dir") or str(run_dir),
            "checkpoint_path": metadata.get("checkpoint_path", ""),
            "total_config_path": metadata.get("total_config_path", ""),
            "latest_evaluator_export": _latest_evaluator_export(run_dir),
            "error": metadata.get("error", ""),
        }
        changed_params = metadata.get("changed_params") or {}
        for name in LEARNING_PARAM_NAMES:
            row[name] = _to_float(changed_params.get(name))
        rows.append(row)
    return rows


def load_eval_metadata(eval_root: Path) -> pd.DataFrame:
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Stage C eval root not found: {eval_root}")
    rows = _metadata_rows(eval_root)
    if not rows:
        raise FileNotFoundError(f"No eval_metadata.json files found under {eval_root}")
    return pd.DataFrame(rows)


def _iter_hydrology_summary_files(hydrology_root: Path) -> Iterable[Path]:
    if not hydrology_root.is_dir():
        return []
    return sorted(hydrology_root.rglob("run_summary_evaluator.csv"))


def load_hydrology_summary(hydrology_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for summary_path in _iter_hydrology_summary_files(hydrology_root):
        frame = pd.read_csv(summary_path)
        scenario_id = summary_path.parent.name
        if "scenario_id" not in frame.columns:
            frame["scenario_id"] = scenario_id
        frame["hydrology_summary_file"] = str(summary_path)
        keep_cols = ["scenario_id", "run_name", "hydrology_summary_file"]
        rename_map: Dict[str, str] = {}
        for source_col, target_col in HYDROLOGY_COLUMN_MAP.items():
            if source_col in frame.columns:
                keep_cols.append(source_col)
                rename_map[source_col] = target_col
        if "result_eval_value" in frame.columns:
            keep_cols.append("result_eval_value")
            rename_map["result_eval_value"] = "hydrology_result_eval_value"
        frames.append(frame[keep_cols].rename(columns=rename_map))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    for col in set(HYDROLOGY_COLUMN_MAP.values()).intersection(merged.columns):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def _pct_delta(value: pd.Series, reference: pd.Series) -> pd.Series:
    denom = reference.abs().replace(0, np.nan)
    return (value - reference) / denom * 100.0


def build_scorecard(eval_df: pd.DataFrame, hydrology_df: pd.DataFrame) -> pd.DataFrame:
    scorecard = eval_df.copy()
    if not hydrology_df.empty:
        scorecard = scorecard.merge(hydrology_df, on=["scenario_id", "run_name"], how="left")

    scorecard["eval_return_rank"] = scorecard.groupby("scenario_id")["eval_return_mean"].rank(
        method="min", ascending=False
    )
    if "generation" in scorecard.columns:
        scorecard["generation_rank"] = scorecard.groupby("scenario_id")["generation"].rank(
            method="min", ascending=False
        )

    default_ref = (
        scorecard[scorecard["config_id"] == "default"]
        .groupby("scenario_id", dropna=False)
        .agg(default_eval_return=("eval_return_mean", "mean"))
        .reset_index()
    )
    scorecard = scorecard.merge(default_ref, on="scenario_id", how="left")
    scorecard["eval_return_delta_vs_default_pct"] = _pct_delta(
        scorecard["eval_return_mean"], scorecard["default_eval_return"]
    )

    for metric in [
        "generation",
        "non_flood_spill",
        "storage_spill_conflict",
        "boundary_pressure_spill",
        "low_level_pressure_days",
    ]:
        if metric not in scorecard.columns:
            continue
        ref = (
            scorecard[scorecard["config_id"] == "default"]
            .groupby("scenario_id", dropna=False)[metric]
            .mean()
            .rename(f"default_{metric}")
            .reset_index()
        )
        scorecard = scorecard.merge(ref, on="scenario_id", how="left")
        scorecard[f"{metric}_delta_vs_default_pct"] = _pct_delta(
            scorecard[metric], scorecard[f"default_{metric}"]
        )

    scorecard["return_not_below_default_5pct"] = scorecard["eval_return_delta_vs_default_pct"].fillna(-np.inf) >= -5.0
    if "low_level_pressure_days_delta_vs_default_pct" in scorecard.columns:
        scorecard["low_level_pressure_not_worse_10pct"] = (
            scorecard["low_level_pressure_days_delta_vs_default_pct"].fillna(np.inf) <= 10.0
        )
    else:
        scorecard["low_level_pressure_not_worse_10pct"] = np.nan
    if "boundary_pressure_spill_delta_vs_default_pct" in scorecard.columns:
        scorecard["boundary_spill_not_worse_10pct"] = (
            scorecard["boundary_pressure_spill_delta_vs_default_pct"].fillna(np.inf) <= 10.0
        )
    else:
        scorecard["boundary_spill_not_worse_10pct"] = np.nan

    available_gates = ["return_not_below_default_5pct"]
    for col in ["low_level_pressure_not_worse_10pct", "boundary_spill_not_worse_10pct"]:
        if scorecard[col].notna().any():
            available_gates.append(col)
    scorecard["stage_c_acceptability_flag"] = scorecard[available_gates].all(axis=1)
    return scorecard


def build_rank_stability(scorecard: pd.DataFrame) -> pd.DataFrame:
    grouped = scorecard.groupby(["config_id", "stage_c_role"], dropna=False)
    rows = grouped.agg(
        scenario_count=("scenario_id", "nunique"),
        eval_return_mean=("eval_return_mean", "mean"),
        eval_return_std=("eval_return_mean", "std"),
        eval_return_min=("eval_return_mean", "min"),
        eval_return_max=("eval_return_mean", "max"),
        rank_mean=("eval_return_rank", "mean"),
        rank_std=("eval_return_rank", "std"),
        rank_min=("eval_return_rank", "min"),
        rank_max=("eval_return_rank", "max"),
        top3_scenario_count=("eval_return_rank", lambda s: int((s <= 3).sum())),
        acceptable_scenario_count=("stage_c_acceptability_flag", lambda s: int(s.fillna(False).sum())),
        worse_than_default_count=("eval_return_delta_vs_default_pct", lambda s: int((s < 0).sum())),
    ).reset_index()
    rows["eval_return_cv"] = rows["eval_return_std"] / rows["eval_return_mean"].abs().replace(0, np.nan)
    rows["acceptable_scenario_rate"] = rows["acceptable_scenario_count"] / rows["scenario_count"].replace(0, np.nan)
    rows["top3_scenario_rate"] = rows["top3_scenario_count"] / rows["scenario_count"].replace(0, np.nan)
    return rows.sort_values(["rank_mean", "eval_return_mean"], ascending=[True, False])


def build_behavior_index(scorecard: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario_id",
        "scenario_type",
        "config_id",
        "stage_c_role",
        "run_name",
        "source_run_name",
        "source_seed",
        "eval_output_dir",
        "latest_evaluator_export",
        "eval_return_mean",
        "eval_return_rank",
    ]
    return scorecard[[col for col in columns if col in scorecard.columns]].copy()


def write_report(
    scorecard: pd.DataFrame,
    rank_stability: pd.DataFrame,
    hydrology_available: bool,
    output_dir: Path,
) -> None:
    lines = [
        "# PaperFour Stage C Scorecard Report",
        "",
        "## Coverage",
        "",
        f"- Evaluation rows: {len(scorecard)}",
        f"- Scenario count: {scorecard['scenario_id'].nunique()}",
        f"- Config count: {scorecard['config_id'].nunique()}",
        f"- Hydrology metrics merged: {hydrology_available}",
        "",
        "## Notes",
        "",
        "- Stage C is offline evaluation only; it must not be described as retraining.",
        "- Total spillage remains a neutral diagnostic and is not used as a one-way risk metric.",
        "- If hydrology metrics are missing, do not claim changes in lower-bound water-level pressure, boundary-pressure spillage, or conditional spillage.",
        "",
        "## Best By Scenario",
        "",
    ]
    for scenario_id, frame in scorecard.groupby("scenario_id"):
        best = frame.sort_values("eval_return_rank").iloc[0]
        lines.append(
            f"- {scenario_id}: {best['config_id']} "
            f"(eval_return_mean={best['eval_return_mean']:.3f}, rank={best['eval_return_rank']:.0f})"
        )
    lines.extend(["", "## Rank Stability", ""])
    for _, row in rank_stability.head(8).iterrows():
        lines.append(
            f"- {row['config_id']}: rank_mean={row['rank_mean']:.2f}, "
            f"top3_rate={row['top3_scenario_rate']:.2f}, "
            f"acceptable_rate={row['acceptable_scenario_rate']:.2f}"
        )
    (output_dir / "stage_c_scorecard_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PaperFour Stage C scenario scorecard and rank stability tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--hydrology-root", type=Path, default=DEFAULT_HYDROLOGY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    hydrology_root = args.hydrology_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = load_eval_metadata(eval_root)
    hydrology_df = load_hydrology_summary(hydrology_root)
    scorecard = build_scorecard(eval_df, hydrology_df)
    rank_stability = build_rank_stability(scorecard)
    behavior_index = build_behavior_index(scorecard)

    scorecard.to_csv(output_dir / "stage_c_scenario_scorecard.csv", index=False, encoding="utf-8-sig")
    rank_stability.to_csv(output_dir / "stage_c_rank_stability.csv", index=False, encoding="utf-8-sig")
    behavior_index.to_csv(output_dir / "stage_c_behavior_timeseries_index.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(output_dir / "stage_c_eval_inventory.csv", index=False, encoding="utf-8-sig")
    write_report(scorecard, rank_stability, not hydrology_df.empty, output_dir)

    print(f"Stage C scorecard rows: {len(scorecard)}")
    print(f"Stage C rank-stability rows: {len(rank_stability)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
