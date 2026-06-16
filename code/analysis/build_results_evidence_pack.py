from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEARNING_PARAMS = [
    "learning_rate",
    "critic_learning_rate",
    "clip_ratio",
    "entropy_weight",
    "discount_factor",
    "gae_lambda",
]

PARAM_LABELS = {
    "learning_rate": "actor learning rate",
    "critic_learning_rate": "critic learning rate",
    "clip_ratio": "clip ratio",
    "entropy_weight": "entropy weight",
    "discount_factor": "discount factor",
    "gae_lambda": "GAE lambda",
}

METRIC_LABELS = {
    "return": "Evaluator return",
    "power": "Generation",
    "low_level_pressure_days": "Lower-bound water-level pressure days",
    "mean_action_correction": "Action correction",
    "any_violation_rate": "Violation rate",
    "boundary_pressure_spill": "Boundary-pressure spillage",
    "non_flood_spill": "Non-flood-season spillage",
}

FORBIDDEN_PATTERNS = [
    "metric_总弃水体积_benefit_delta",
    "total_spillage_benefit_delta",
]


def _reservoir_root() -> Path:
    # .../reservoir_multi_agents/analysis/experiment_analysis_paperFour/this_file.py
    return Path(__file__).resolve().parents[2]


def _paperfour_dir(reservoir_root: Path) -> Path:
    markdown_dir = reservoir_root / "markdown"
    for child in markdown_dir.iterdir():
        candidate = child / "PaperFour"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Cannot locate markdown/*/PaperFour")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Any, digits: int = 2) -> str:
    value = _safe_float(value)
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def _pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return (numerator - denominator) / abs(denominator) * 100.0


def _row_by_config(frame: pd.DataFrame, config_id: str) -> pd.Series:
    sub = frame[frame["config_id"].astype(str) == config_id]
    if sub.empty:
        raise KeyError(config_id)
    return sub.iloc[0]


def _spearman_value(spearman: pd.DataFrame, parameter: str, metric: str) -> float:
    sub = spearman[
        (spearman["level"].astype(str) == "config")
        & (spearman["parameter"].astype(str) == parameter)
        & (spearman["metric"].astype(str) == metric)
    ]
    if sub.empty:
        return float("nan")
    return _safe_float(sub.iloc[0]["spearman"])


def _add_key(rows: list[dict[str, Any]], rq: str, figure: str, metric: str, config_or_group: str, value: Any, unit: str, source: str, claim: str) -> None:
    rows.append(
        {
            "evidence_id": f"E{len(rows) + 1:03d}",
            "rq": rq,
            "figure": figure,
            "metric": metric,
            "config_or_group": config_or_group,
            "value": value,
            "unit": unit,
            "source_file": source,
            "claim_relevance": claim,
        }
    )


def _figure_counts(figure_output_dir: Path) -> dict[str, int]:
    return {
        "png": len(list(figure_output_dir.glob("fig*.png"))),
        "pdf": len(list(figure_output_dir.glob("fig*.pdf"))),
        "md": len(list(figure_output_dir.glob("fig*.md"))),
    }


def build_key_numbers(data: dict[str, pd.DataFrame], figure_output_dir: Path) -> pd.DataFrame:
    config_summary = data["config_summary"]
    response = data["response"]
    spearman = data["spearman"]
    interactions = data["interactions"]
    robust_interval = data["robust_interval"]
    stage_c = data["stage_c"]
    fig8 = data["fig8"]

    rows: list[dict[str, Any]] = []
    default_b = _row_by_config(config_summary, "default")
    robust_b = _row_by_config(config_summary, "gamma099_clip02")
    risk_b = _row_by_config(config_summary, "risk_lhs009")
    long_b = _row_by_config(config_summary, "gamma0999_clip02")
    high_clip_b = _row_by_config(config_summary, "clip03_gamma995")

    _add_key(rows, "coverage", "Table S", "complete Stage B runs", "Stage B", len(response), "runs", "stage_b_response_dataset.csv", "Stage B has the full 19 configs x 3 seeds response dataset.")
    _add_key(rows, "coverage", "Table S", "config count", "Stage B", config_summary["config_id"].nunique(), "configs", "stage_b_config_summary.csv", "All Stage B configurations can be aggregated.")
    _add_key(rows, "RQ1", "Fig.4", "Spearman(return)", "discount_factor", _spearman_value(spearman, "discount_factor", "return"), "rho", "stage_b_parameter_spearman.csv", "Discount factor is the strongest return-sensitive axis.")
    _add_key(rows, "RQ1", "Fig.4", "Spearman(return)", "clip_ratio", _spearman_value(spearman, "clip_ratio", "return"), "rho", "stage_b_parameter_spearman.csv", "Clip ratio is the second dominant return-sensitive axis.")
    _add_key(rows, "RQ1", "Fig.4", "Spearman(LBP)", "discount_factor", _spearman_value(spearman, "discount_factor", "low_level_pressure_days"), "rho", "stage_b_parameter_spearman.csv", "Higher discount factors are associated with higher lower-bound water-level pressure.")
    _add_key(rows, "RQ1", "Fig.4", "Spearman(violation rate)", "clip_ratio", _spearman_value(spearman, "clip_ratio", "any_violation_rate"), "rho", "stage_b_parameter_spearman.csv", "Higher clip ratio is associated with higher violation pressure.")

    _add_key(rows, "RQ3", "Fig.9", "return_mean", "gamma099_clip02", robust_b["return_mean"], "reward", "stage_b_config_summary.csv", "gamma099_clip02 is the strongest Stage B robust-return candidate.")
    _add_key(rows, "RQ3", "Fig.9", "return_mean", "default", default_b["return_mean"], "reward", "stage_b_config_summary.csv", "Default is a mid-to-upper baseline but not the best configuration.")
    _add_key(rows, "RQ3", "Fig.9", "return delta vs default", "gamma099_clip02", _pct(_safe_float(robust_b["return_mean"]), _safe_float(default_b["return_mean"])), "%", "stage_b_config_summary.csv", "gamma099_clip02 improves return over default in Stage B.")
    _add_key(rows, "RQ3", "Fig.9", "return_mean", "risk_lhs009", risk_b["return_mean"], "reward", "stage_b_config_summary.csv", "risk_lhs009 is the primary failure counterexample.")
    _add_key(rows, "RQ3", "Fig.9", "return_cv", "risk_lhs009", risk_b["return_cv"], "CV", "stage_b_config_summary.csv", "risk_lhs009 has high seed uncertainty and should not define sampled acceptable values.")
    _add_key(rows, "RQ3", "Fig.9", "low_level_pressure_days_mean", "risk_lhs009", risk_b["low_level_pressure_days_mean"], "reservoir-days", "stage_b_config_summary.csv", "Failure behavior is reflected in water-level pressure, not only return.")
    _add_key(rows, "RQ3", "Fig.9", "return_mean", "gamma0999_clip02", long_b["return_mean"], "reward", "stage_b_config_summary.csv", "The long-horizon risk case supports the high-gamma instability interpretation.")
    _add_key(rows, "RQ3", "Fig.9", "return_mean", "clip03_gamma995", high_clip_b["return_mean"], "reward", "stage_b_config_summary.csv", "The high-clip case is a controlled high-update-risk comparison.")

    for parameter in ["clip_ratio", "discount_factor", "gae_lambda", "entropy_weight"]:
        interval = robust_interval[robust_interval["parameter"] == parameter]
        if not interval.empty:
            _add_key(rows, "RQ3", "Table 4", f"sampled acceptable values: {parameter}", parameter, interval.iloc[0]["unique_values"], "candidate values", "stage_b_robust_interval.csv", f"Sampled acceptable values for {parameter} based on Stage B screening constraints.")

    clip_gamma = interactions[interactions["interaction"].astype(str) == "clip_ratio x discount_factor"]
    if not clip_gamma.empty:
        best = clip_gamma.sort_values("return_mean", ascending=False).iloc[0]
        worst_pressure = clip_gamma.sort_values("low_level_pressure_days_mean", ascending=False).iloc[0]
        _add_key(rows, "RQ2", "Fig.5", "best clip-gamma group return", f"clip={best['value_a']}; gamma={best['value_b']}", best["return_mean"], "reward", "stage_b_interactions.csv", "Low clip with gamma=0.99 forms the strongest interaction region.")
        _add_key(rows, "RQ2", "Fig.5", "highest clip-gamma LBP", f"clip={worst_pressure['value_a']}; gamma={worst_pressure['value_b']}", worst_pressure["low_level_pressure_days_mean"], "reservoir-days", "stage_b_interactions.csv", "High discount factor regions intensify lower-bound water-level pressure.")

    stage_c_default = _row_by_config(stage_c, "default")
    stage_c_robust = _row_by_config(stage_c, "gamma099_clip02")
    stage_c_long = _row_by_config(stage_c, "gamma0999_clip02")
    stage_c_risk = _row_by_config(stage_c, "risk_lhs009")
    _add_key(rows, "RQ4", "Fig.10", "eval_return_mean", "gamma099_clip02", stage_c_robust["eval_return_mean"], "reward", "stage_c_scenario_scorecard.csv", "gamma099_clip02 ranks first in mixed-sequence validation.")
    _add_key(rows, "RQ4", "Fig.10", "eval_return_delta_vs_default_pct", "gamma099_clip02", stage_c_robust["eval_return_delta_vs_default_pct"], "%", "stage_c_scenario_scorecard.csv", "The operationally acceptable candidate remains above default in the held-out mixed sequence.")
    _add_key(rows, "RQ4", "Fig.10", "low_level_pressure_days", "gamma099_clip02", stage_c_robust["low_level_pressure_days"], "reservoir-days", "stage_c_scenario_scorecard.csv", "The operationally acceptable candidate reduces LBP relative to default.")
    _add_key(rows, "RQ4", "Fig.10", "low_level_pressure_days", "default", stage_c_default["low_level_pressure_days"], "reservoir-days", "stage_c_scenario_scorecard.csv", "Default is the validation baseline.")
    _add_key(rows, "RQ4", "Fig.10", "eval_return_delta_vs_default_pct", "gamma0999_clip02", stage_c_long["eval_return_delta_vs_default_pct"], "%", "stage_c_scenario_scorecard.csv", "The long-horizon risk case remains degraded in mixed-sequence validation.")
    _add_key(rows, "RQ4", "Fig.10", "eval_return_delta_vs_default_pct", "risk_lhs009", stage_c_risk["eval_return_delta_vs_default_pct"], "%", "stage_c_scenario_scorecard.csv", "The failure counterexample remains strongly degraded in mixed-sequence validation.")

    fig8_group = fig8.groupby("config_id", as_index=False).agg(
        action_correction=("action_correction_mean", "mean"),
        low_pressure_days=("low_pressure_days", "sum"),
        generation=("generation_100mkwh_sum", "sum"),
    )
    for config_id in ["default", "gamma099_clip02", "gamma0999_clip02", "risk_lhs009"]:
        row = _row_by_config(fig8_group, config_id)
        _add_key(rows, "RQ2", "Fig.8", "reservoir mean action correction", config_id, row["action_correction"], "normalized action", "fig8_reservoir_heterogeneity.csv", f"Reservoir-level action correction profile for {config_id}.")
        _add_key(rows, "RQ2", "Fig.8", "reservoir low-pressure days", config_id, row["low_pressure_days"], "reservoir-days", "fig8_reservoir_heterogeneity.csv", f"Reservoir-level pressure profile for {config_id}.")

    _add_key(rows, "diagnostic", "Fig.6/Fig.10", "storage_spill_conflict_mean", "all Stage B configs", config_summary["storage_spill_conflict_mean"].max(), "index", "stage_b_config_summary.csv", "Storage-spill conflict is not triggered and should be reported as absent rather than interpreted as a main effect.")
    counts = _figure_counts(figure_output_dir)
    _add_key(rows, "coverage", "Figure set", "figure PDF count", "Fig.1-Fig.10 and Fig.S1-Fig.S3", counts["pdf"], "files", "figures/output", "All main and supplementary figures have PDF outputs.")
    return pd.DataFrame(rows)


def build_claim_map() -> pd.DataFrame:
    rows = [
        {
            "rq": "Methods",
            "result_paragraph": "固定系统与实验边界",
            "figure": "Fig.1-Fig.2",
            "panel": "all",
            "source_file": "fig1_framework_nodes.csv; fig2_fixed_system.csv; fig2_stage_design.csv",
            "allowed_claim": "PaperFour 固定 5Res + HAPPO + fixed observation setting + 默认奖励，只研究 6 个学习动力学参数。",
            "forbidden_claim": "不要把 PaperOne 架构比较结论写成 PaperFour 结果。",
        },
        {
            "rq": "RQ1",
            "result_paragraph": "参数敏感性主轴",
            "figure": "Fig.4",
            "panel": "a-e",
            "source_file": "fig4_parameter_importance.csv; stage_b_parameter_spearman.csv",
            "allowed_claim": "`discount_factor` 与 `clip_ratio` 是最主要的学习动力学敏感轴，且同时影响算法回报和水文压力指标。",
            "forbidden_claim": "不要只按 return 排名断言全部水文稳健性。",
        },
        {
            "rq": "RQ2",
            "result_paragraph": "clip-gamma 交互机制",
            "figure": "Fig.5",
            "panel": "a-d",
            "source_file": "fig5_clip_gamma_3d.csv; stage_b_interactions.csv",
            "allowed_claim": "较低 `discount_factor` 与中低 `clip_ratio` 区域对应较高 return 与较低执行/水位压力。",
            "forbidden_claim": "不要将稀疏响应面解释为连续物理过程或精确最优控制曲面。",
        },
        {
            "rq": "RQ2",
            "result_paragraph": "跨季节机制",
            "figure": "Fig.6",
            "panel": "a-i",
            "source_file": "fig6_gamma_gae_seasonal_response.csv",
            "allowed_claim": "季节发电与季节弃水诊断揭示长视野参数对枯水期、汛期和蓄水恢复期调度权衡的影响。",
            "forbidden_claim": "Fig.6 的低水位压力行不能写成逐季节低水位压力过程；季节弃水不能写成越少越好。",
        },
        {
            "rq": "RQ2",
            "result_paragraph": "代表配置调度过程",
            "figure": "Fig.7",
            "panel": "a-d",
            "source_file": "fig7_mechanism_timeseries.csv",
            "allowed_claim": "代表配置在同一混合五年序列上表现出不同的水位、出流、动作修正和低水位事件轨迹。",
            "forbidden_claim": "不要把 normalized traces 写成原始物理单位值。",
        },
        {
            "rq": "RQ2",
            "result_paragraph": "水库级异质性",
            "figure": "Fig.8",
            "panel": "a-d",
            "source_file": "fig8_reservoir_heterogeneity.csv",
            "allowed_claim": "系统级收益和压力差异需要回溯到 5 个水库的发电、低水位压力、动作修正和剩余库容差异。",
            "forbidden_claim": "不要只用系统总量替代单库机制解释。",
        },
        {
            "rq": "RQ4",
            "result_paragraph": "训练外混合序列验证",
            "figure": "Fig.10",
            "panel": "a-d",
            "source_file": "stage_c_scenario_scorecard.csv; fig10_mixed_sequence_validation.csv",
            "allowed_claim": "`gamma099_clip02` 在 mixed-sequence validation 中保持高于 default 的 return，并降低低水位压力和边界压力弃水。",
            "forbidden_claim": "不要写成严格多情景泛化；当前只有一个 mixed_percentile_5yr 序列。",
        },
        {
            "rq": "RQ3",
            "result_paragraph": "多目标稳健性",
            "figure": "Fig.9",
            "panel": "a-d",
            "source_file": "stage_b_config_summary.csv; fig9_pareto_diagnostics.csv",
            "allowed_claim": "稳健配置应同时满足 return、低水位压力、动作修正、违规率和条件弃水诊断，而不是单一 return 或总弃水。",
            "forbidden_claim": "不要把总弃水作为单向风险排序轴。",
        },
    ]
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = frame[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        values = [str(row[column]).replace("\n", "<br>") for column in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_evidence_markdown(path: Path, key_numbers: pd.DataFrame, claim_map: pd.DataFrame) -> None:
    selected = key_numbers[
        key_numbers["figure"].isin(["Fig.4", "Fig.5", "Fig.7", "Fig.8", "Fig.9", "Fig.10", "Table 4"])
    ].copy()
    selected["value"] = selected["value"].map(lambda value: _fmt(value, 3) if isinstance(value, (int, float, np.floating)) else str(value))
    lines = [
        "# PaperFour Results 证据表",
        "",
        "本文档用于把 Stage B、Stage C 和 Fig.4-Fig.10 的图源证据映射到可写论文结论。它是 Results 写作前的证据包，不替代正式论文正文。",
        "",
        "## 1. 核心数字",
        "",
        _markdown_table(selected, ["evidence_id", "rq", "figure", "metric", "config_or_group", "value", "unit", "source_file"]),
        "",
        "## 2. 图表-结论映射",
        "",
        _markdown_table(claim_map, ["rq", "result_paragraph", "figure", "panel", "allowed_claim", "forbidden_claim"]),
        "",
        "## 3. 写作约束",
        "",
        "- Fig.1 和 Fig.2 进入 Methods，用于说明固定系统、参数边界和实验流程，不作为 Results 主结论。",
        "- Fig.6 的低水位压力行是 Stage B run-level grouped diagnostic，不能写成逐季节低水位压力过程。",
        "- Fig.10 只能写作 mixed-sequence validation，不能写成严格的多独立情景泛化。",
        "- 总弃水、汛期弃水和季节弃水只能作为 neutral diagnostic；风险解释依赖非汛期弃水、边界压力弃水、低水位压力、动作修正和违规率。",
        "- Results 每个段落必须绑定 Fig.4-Fig.10 或 Stage B/C 表格证据；Fig.3 仅作为实验对象说明。",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_results_draft(path: Path, key_numbers: pd.DataFrame) -> None:
    data = {row["evidence_id"]: row for _, row in key_numbers.iterrows()}
    lookup = {(row["metric"], row["config_or_group"]): row for _, row in key_numbers.iterrows()}

    def val(eid: str, digits: int = 2) -> str:
        return _fmt(data[eid]["value"], digits)

    def value_for(metric: str, config_or_group: str, digits: int = 2) -> str:
        return _fmt(lookup[(metric, config_or_group)]["value"], digits)

    def raw_for(metric: str, config_or_group: str) -> str:
        return str(lookup[(metric, config_or_group)]["value"])

    lines = [
        "# PaperFour Results 中文逻辑稿",
        "",
        "本文档是 Results 章节的中文逻辑稿。每一段均绑定主文图或结果表证据，后续英文写作时应保留“学习动力学参数 -> 学习更新过程 -> 水文调度行为 -> 训练外混合序列验证”的解释链。",
        "",
        "## RQ1：HAPPO 学习动力学参数的敏感性主轴",
        "",
        f"Stage B 的 `19 configs × 3 seeds` 共形成 {val('E001', 0)} 条完整 run-level 记录，可用于配置级敏感性分析。Fig.4 显示，`discount_factor` 和 `clip_ratio` 是最主要的敏感轴。其中，`discount_factor` 与 return 的 Spearman 相关为 {val('E003', 3)}，`clip_ratio` 与 return 的 Spearman 相关为 {val('E004', 3)}。这说明 PaperFour 的性能变化并不是随机训练波动，而是与 HAPPO 的时间折扣和 PPO 更新幅度控制存在明确关联。",
        "",
        f"从水文风险指标看，`discount_factor` 与低水位压力天数的相关为 {val('E005', 3)}，`clip_ratio` 与违规率的相关为 {val('E006', 3)}。因此，学习动力学参数不仅改变累计 return，还会通过动作更新强度和时间信用分配影响水位压力和约束可执行性。该结果支撑 Fig.4 的核心结论：敏感性分析必须同时报告算法指标和水文诊断指标。",
        "",
        "## RQ2：学习动力学参数如何转化为水文调度行为",
        "",
        f"Fig.5 进一步显示 `clip_ratio × discount_factor` 存在清晰交互。交互分组中，最优 return 区域对应 {data['E019']['config_or_group']}，平均 return 为 {val('E019', 2)}；而最高低水位压力区域对应 {data['E020']['config_or_group']}，低水位压力达到 {val('E020', 2)} reservoir-days。这表明高折扣因子会放大学习策略对跨期蓄泄行为的影响，而 clip ratio 控制的更新幅度会改变动作修正和违规压力。",
        "",
        "Fig.6 将上述交互放到枯水期、汛期和蓄水恢复期解释框架中。该图用于比较不同 `discount_factor × gae_lambda` 分组下的季节发电和季节弃水诊断，其中季节弃水只作为水文行为解释变量。Fig.7 和 Fig.8 则提供机制层证据：代表配置在同一混合五年序列上具有不同的水位状态、出流、动作修正、低水位事件和单库响应。尤其是 Fig.8 的水库级矩阵说明，系统级 return 差异不能只用总量解释，必须回溯到 5 个水库在发电、低水位压力、动作修正和剩余库容上的异质性。",
        "",
        "## RQ3：稳健区间与代表配置",
        "",
        f"Stage B 中 `gamma099_clip02` 是当前最强可接受候选，配置级 return_mean 为 {val('E007', 2)}，default 的 return_mean 为 {val('E008', 2)}，相对提升约 {val('E009', 2)}%。采样可接受值表明，`clip_ratio` 的可接受范围集中在 {raw_for('sampled acceptable values: clip_ratio', 'clip_ratio')}，`discount_factor` 的可接受范围集中在 {raw_for('sampled acceptable values: discount_factor', 'discount_factor')}，`gae_lambda` 的可接受范围集中在 {raw_for('sampled acceptable values: gae_lambda', 'gae_lambda')}。这些结果共同支持 Fig.9 的多目标可接受性判断。",
        "",
        f"相反，`risk_lhs009` 是清晰失稳反例，Stage B return_mean 仅为 {val('E010', 2)}，return CV 为 {val('E011', 3)}，低水位压力达到 {val('E012', 2)} reservoir-days。`gamma0999_clip02` 和 `clip03_gamma995` 分别对应长视野风险和高 clip 风险，对照说明过长时间折扣或过强更新幅度均可能导致 return 退化和水位压力上升。由此，稳健性不能仅依据最高 return 判断，而应同时约束低水位压力、动作修正、违规率和条件弃水诊断。",
        "",
        "## RQ4：训练指标、水文指标和训练外混合序列验证的一致性",
        "",
        f"Stage C 使用 `mixed_percentile_5yr` 训练外混合序列对 8 个代表 checkpoint 进行离线评估。Fig.10 显示，`gamma099_clip02` 在该序列中仍排名第一，eval_return_mean 为 {val('E021', 2)}，相对 default 提升 {val('E022', 2)}%。同时，其低水位压力为 {val('E023', 0)} reservoir-days，低于 default 的 {val('E024', 0)} reservoir-days。这说明 Stage B 中识别出的稳健候选在训练外混合序列上仍保持较好的算法性能和水文压力表现。",
        "",
        f"失稳反例在 Stage C 中也保持一致。`gamma0999_clip02` 相对 default 的 return 变化为 {val('E025', 2)}%，`risk_lhs009` 相对 default 的 return 变化为 {val('E026', 2)}%。这一结果强化了学习动力学参数敏感性具有可解释的水文后果，而不是单次 seed 或单一评价指标造成的偶然排序。需要注意的是，当前 Stage C 是单个混合五年序列验证，因此正文应表述为训练外混合序列验证，而非更强的泛化结论。",
        "",
        "## Results 写作边界",
        "",
        "- 不使用总弃水作为单调改进目标。",
        "- 不把季节弃水或总弃水作为单向风险结论。",
        "- 不把 Fig.6 的低水位压力行写成逐季节过程。",
        "- 不把 Fig.S1-S3 作为主文核心证据，只在补充材料或审稿回复中使用。",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_methods_caption_draft(path: Path) -> None:
    lines = [
        "# PaperFour Methods 图注落稿",
        "",
        "本文档用于提前固定 Fig.1 和 Fig.2 在 Methods 中的表述，避免 Results 章节重复解释实验边界。",
        "",
        "## Fig.1 Learning-hydrology interpretation framework",
        "",
        "Fig.1 展示 PaperFour 的解释链：6 个 HAPPO 学习动力学参数首先影响策略更新信号，包括 evaluator return、approximate KL、clip fraction、entropy loss 和 value loss；这些学习信号进一步影响动作修正、约束可执行性和水库水位状态；最终表现为发电收益、下限水位压力和条件弃水诊断。该图的作用是定义论文的分析框架，而不是报告某一个配置的最终性能。",
        "",
        "建议英文图注：",
        "",
        "> Fig. 1. Learning-to-hydrology interpretation framework for PaperFour. The six controlled HAPPO learning-dynamics parameters are linked to learner diagnostics, action feasibility, reservoir state pressure, conditional spillage diagnostics and water-resource outcomes. Total spillage is retained as a neutral diagnostic and is not used as a one-way risk indicator.",
        "",
        "## Fig.2 Experimental design",
        "",
        "Fig.2 展示 PaperFour 的固定系统和分阶段实验流程。固定部分为 `ResStr5.xlsx + HAPPO + fixed observation setting + default reward`，主文只改变 `learning_rate`、`critic_learning_rate`、`clip_ratio`、`entropy_weight`、`discount_factor` 和 `gae_lambda`。Stage 0 用于烟测，Stage A 用于低成本筛选，Stage B 用于主文敏感性实验，Stage C 用于训练外混合五年序列离线验证。",
        "",
        "建议英文图注：",
        "",
        "> Fig. 2. Experimental design of PaperFour. The reservoir system, HAPPO algorithm, fixed observation setting, and default reward function are fixed, while only six learning-dynamics parameters are varied. Stage 0 verifies the experiment chain, Stage A screens candidate regions, Stage B provides the main sensitivity evidence, and Stage C evaluates representative checkpoints under a held-out mixed five-year inflow sequence.",
        "",
        "## Methods 写作边界",
        "",
        "- Fig.1 和 Fig.2 不用于证明 `gamma099_clip02` 最优；该结论应放在 Results 并引用 Fig.9/Fig.10。",
        "- Methods 中只写实验边界、变量定义和证据链，不写 Stage B/C 排名结果。",
        "- PaperOne 只能作为流程范本，不作为 PaperFour 的结果来源。",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def validate_outputs(figure_data_dir: Path, key_numbers: pd.DataFrame, claim_map: pd.DataFrame, results_draft_path: Path) -> list[str]:
    errors: list[str] = []
    required_columns = {"config_id", "unit", "indicator_direction"}
    main_sources = [
        "fig4_homogeneous_lollipop.csv",
        "fig5_homogeneous_contour.csv",
        "fig6_homogeneous_seasonal.csv",
        "fig7_homogeneous_timeseries.csv",
        "fig8_homogeneous_matrix.csv",
        "fig9_homogeneous_scatter.csv",
        "fig10_homogeneous_lollipop.csv",
    ]
    for name in main_sources:
        frame = _read_csv(figure_data_dir / name)
        missing = required_columns - set(frame.columns)
        if missing:
            errors.append(f"{name} missing columns: {sorted(missing)}")
    for path in figure_data_dir.glob("fig*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in columns:
                errors.append(f"Forbidden column {pattern} in {path.name}")
    if key_numbers.empty:
        errors.append("key_numbers is empty")
    if claim_map["figure"].isna().any() or claim_map["source_file"].isna().any():
        errors.append("claim_map has empty figure/source fields")
    results_text = results_draft_path.read_text(encoding="utf-8")
    forbidden_text = ["多情景稳健性", "总弃水越少越好"]
    for text in forbidden_text:
        if text in results_text:
            errors.append(f"Forbidden wording in results draft: {text}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PaperFour Results evidence pack and writing drafts.")
    parser.add_argument("--analysis-output-dir", type=Path, default=None, help="Directory for CSV outputs. Defaults to analysis/experiment_analysis_paperFour/output_results_evidence.")
    parser.add_argument("--paper-dir", type=Path, default=None, help="PaperFour markdown directory. Auto-detected by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reservoir_root = _reservoir_root()
    analysis_dir = reservoir_root / "analysis" / "experiment_analysis_paperFour"
    paper_dir = args.paper_dir or _paperfour_dir(reservoir_root)
    stage_b_dir = analysis_dir / "output_stage_b"
    stage_c_dir = analysis_dir / "output_stage_c"
    figure_data_dir = paper_dir / "figures" / "data"
    figure_output_dir = paper_dir / "figures" / "output"
    output_dir = args.analysis_output_dir or (analysis_dir / "output_results_evidence")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "response": _read_csv(stage_b_dir / "stage_b_response_dataset.csv"),
        "config_summary": _read_csv(stage_b_dir / "stage_b_config_summary.csv"),
        "spearman": _read_csv(stage_b_dir / "stage_b_parameter_spearman.csv"),
        "interactions": _read_csv(stage_b_dir / "stage_b_interactions.csv"),
        "robust_interval": _read_csv(stage_b_dir / "stage_b_robust_interval.csv"),
        "stage_c": _read_csv(stage_c_dir / "stage_c_scenario_scorecard.csv"),
        "fig8": _read_csv(figure_data_dir / "fig8_reservoir_heterogeneity.csv"),
    }

    key_numbers = build_key_numbers(data, figure_output_dir)
    claim_map = build_claim_map()
    key_numbers_path = output_dir / "paperfour_results_key_numbers.csv"
    claim_map_path = output_dir / "paperfour_figure_claim_map.csv"
    key_numbers.to_csv(key_numbers_path, index=False, encoding="utf-8-sig")
    claim_map.to_csv(claim_map_path, index=False, encoding="utf-8-sig")

    evidence_path = paper_dir / "PaperFour_Results_证据表.md"
    results_draft_path = paper_dir / "PaperFour_Results_中文逻辑稿.md"
    methods_caption_path = paper_dir / "PaperFour_Methods_图注落稿.md"
    write_evidence_markdown(evidence_path, key_numbers, claim_map)
    write_results_draft(results_draft_path, key_numbers)
    write_methods_caption_draft(methods_caption_path)

    errors = validate_outputs(figure_data_dir, key_numbers, claim_map, results_draft_path)
    if errors:
        raise SystemExit("\n".join(errors))

    print("PaperFour Results evidence pack generated:")
    print(f" - {key_numbers_path}")
    print(f" - {claim_map_path}")
    print(f" - {evidence_path}")
    print(f" - {results_draft_path}")
    print(f" - {methods_caption_path}")


if __name__ == "__main__":
    main()
