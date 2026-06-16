from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _paperfour_plot_paths import FIGURE_DATA_DIR, FIGURE_OUTPUT_DIR
from _paperfour_plot_style import (
    ALGO_ORANGE,
    ALGO_RED,
    DEFAULT_GREY,
    HYDRO_BLUE,
    HYDRO_CYAN,
    NEUTRAL_GREY,
    RISK_RED,
    ROBUST_GREEN,
    STAGE_PURPLE,
    apply_style,
    config_color,
    panel_label,
    role_marker,
    save_figure,
    write_note,
)


PARAM_ORDER = [
    "learning_rate",
    "critic_learning_rate",
    "clip_ratio",
    "entropy_weight",
    "discount_factor",
    "gae_lambda",
]

PARAM_LABELS = {
    "learning_rate": "actor LR",
    "critic_learning_rate": "critic LR",
    "clip_ratio": "clip ratio",
    "entropy_weight": "entropy",
    "discount_factor": "discount",
    "gae_lambda": "GAE lambda",
}

CONFIG_LABELS = {
    "default": "Baseline",
    "gamma099_clip02": "Operationally\nacceptable",
    "gamma0999_clip02": "Long-horizon\nrisk",
    "risk_lhs009": "Instability\ncase",
    "clip03_gamma995": "High-clipping\nrisk",
    "return_lhs008": "High-return\ncandidate",
    "return_lhs012": "Conservative\ncandidate",
    "entropy004_robust": "Entropy\ncontrast",
}

PRIMARY_CONFIGS = ["default", "gamma099_clip02", "gamma0999_clip02", "risk_lhs009"]

POINT_LABELS = {
    "default": "Baseline",
    "gamma099_clip02": "Acceptable",
    "gamma0999_clip02": "long",
    "risk_lhs009": "Instability",
    "clip03_gamma995": "High clipping",
}

FIG3_METRICS = [
    ("return", "Return", "reward", "higher_better"),
    ("low_level_pressure_days", "Lower-bound pressure", "reservoir-days", "lower_better"),
    ("mean_action_correction", "Action correction", "normalized action", "lower_better"),
    ("any_violation_rate", "Violation rate", "fraction", "lower_better"),
    ("boundary_pressure_spill", "Boundary-pressure spill", "10$^{8}$ m$^{3}$", "lower_better"),
]

FIG4_METRICS = [
    ("return_mean", "Return", "reward", "higher_better", "viridis"),
    ("low_level_pressure_days_mean", "Lower-bound pressure", "reservoir-days", "lower_better", "YlOrRd"),
    ("mean_action_correction_mean", "Action correction", "normalized action", "lower_better", "YlOrBr"),
    ("any_violation_rate_mean", "Violation rate", "fraction", "lower_better", "Reds"),
]

SEASON_ORDER = ["dry", "flood", "recovery"]
SEASON_LABELS = {"dry": "Dry season", "flood": "Flood season", "recovery": "Recovery season"}


def _read(name: str, data_dir: Path = FIGURE_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / name)


def _short(text: str, width: int = 18) -> str:
    return "\n".join(wrap(str(text), width=width, break_long_words=False))


def _clean_config_label(config_id: str) -> str:
    return CONFIG_LABELS.get(str(config_id), str(config_id)).replace("\n", " ")


def _minmax(values: pd.Series, higher_better: bool = True) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    span = values.max() - values.min()
    if not np.isfinite(span) or abs(span) < 1e-12:
        score = pd.Series(0.5, index=values.index)
    else:
        score = (values - values.min()) / span
    return score if higher_better else 1.0 - score


def _scale(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    span = values.max() - values.min()
    if not np.isfinite(span) or abs(span) < 1e-12:
        return pd.Series(0.5, index=values.index)
    return (values - values.min()) / span


def _rolling(values: pd.Series, window: int = 14) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rolling(window, min_periods=1).mean()


def _shade_flood_seasons(ax: plt.Axes, dates: pd.Series) -> None:
    years = sorted(pd.to_datetime(dates).dt.year.dropna().unique())
    for year in years:
        ax.axvspan(
            pd.Timestamp(year=year, month=6, day=1),
            pd.Timestamp(year=year, month=9, day=30),
            color="#DBEAFE",
            alpha=0.25,
            linewidth=0,
        )


def _ordered_configs(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "gamma099_clip02",
        "return_lhs008",
        "return_lhs012",
        "default",
        "entropy004_robust",
        "clip03_gamma995",
        "gamma0999_clip02",
        "risk_lhs009",
    ]
    existing = [config_id for config_id in preferred if config_id in set(frame["config_id"].astype(str))]
    rest = sorted(set(frame["config_id"].astype(str)) - set(existing))
    return existing + rest


def _annotated_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    cmap: str,
    title: str,
    fmt: str = ".2f",
    center_zero: bool = False,
    cbar_label: str | None = None,
) -> None:
    values = pivot.to_numpy(dtype=float)
    norm = None
    if center_zero:
        finite = values[np.isfinite(values)]
        max_abs = max(abs(float(finite.min())), abs(float(finite.max())), 1e-9) if finite.size else 1.0
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    ax.set_title(title, loc="left", fontweight="bold")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=13, color="#111827")
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)


def _sparse_contour(ax: plt.Axes, frame: pd.DataFrame, z_col: str, title: str, cmap: str, cbar_label: str) -> None:
    sub = frame[["clip_ratio", "discount_factor", z_col, "config_id"]].dropna().copy()
    x = pd.to_numeric(sub["clip_ratio"], errors="coerce")
    y = pd.to_numeric(sub["discount_factor"], errors="coerce")
    z = pd.to_numeric(sub[z_col], errors="coerce")
    valid = x.notna() & y.notna() & z.notna()
    x, y, z, sub = x[valid], y[valid], z[valid], sub[valid]
    if len(sub) >= 3 and x.nunique() >= 2 and y.nunique() >= 2:
        try:
            triang = mtri.Triangulation(x.to_numpy(), y.to_numpy())
            filled = ax.tricontourf(triang, z.to_numpy(), levels=12, cmap=cmap, alpha=0.9)
            ax.tricontour(triang, z.to_numpy(), levels=6, colors="#FFFFFF", linewidths=0.55, alpha=0.72)
            plt.colorbar(filled, ax=ax, fraction=0.046, pad=0.02).set_label(cbar_label)
        except Exception:
            scatter = ax.scatter(x, y, c=z, cmap=cmap, s=105, edgecolor="white", linewidth=0.8)
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02).set_label(cbar_label)
    else:
        scatter = ax.scatter(x, y, c=z, cmap=cmap, s=105, edgecolor="white", linewidth=0.8)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02).set_label(cbar_label)
    ax.scatter(x, y, s=40, color="#0F172A", edgecolor="white", linewidth=0.7, zorder=3)
    for _, row in sub.iterrows():
        config_id = str(row["config_id"])
        if config_id in POINT_LABELS:
            x_text = float(row["clip_ratio"])
            ha = "right" if x_text >= 0.28 else "left"
            offset = -0.006 if ha == "right" else 0.006
            ax.text(x_text + offset, float(row["discount_factor"]), POINT_LABELS[config_id], fontsize=11.5, fontweight="bold", ha=ha, clip_on=True)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Clip ratio")
    ax.set_ylabel("Discount factor")
    ax.set_xlim(0.098, 0.302)
    ax.set_ylim(0.9896, 0.9994)
    ax.grid(alpha=0.16, linestyle="--")


def _lollipop_panel(
    ax: plt.Axes,
    labels: list[str],
    values: pd.Series,
    title: str,
    xlabel: str,
    higher_better: bool,
) -> None:
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    y = np.arange(len(labels))
    colors = []
    for value in values:
        if abs(value) < 1e-12:
            colors.append(NEUTRAL_GREY)
        elif higher_better:
            colors.append(ROBUST_GREEN if value >= 0 else RISK_RED)
        else:
            colors.append(RISK_RED if value >= 0 else ROBUST_GREEN)
    ax.axvline(0, color="#475569", linewidth=0.9)
    ax.hlines(y, 0, values, color="#CBD5E1", linewidth=2.5)
    ax.scatter(values, y, s=80, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.22, linestyle="--")


def plot_fig1(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    nodes = _read("fig1_framework_nodes.csv", data_dir)
    edges = _read("fig1_framework_edges.csv", data_dir)
    evidence = _read("fig1_dynamic_evidence.csv", data_dir)
    fig = plt.figure(figsize=(14.6, 6.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.25, 1.0], wspace=0.12, hspace=0.34)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_axis_off()

    layers = [
        ("params", "HAPPO learning\nparameters", ALGO_ORANGE, 0.08),
        ("learner", "Learning dynamics\nsignals", ALGO_RED, 0.31),
        ("execution", "Policy execution\nconstraints", STAGE_PURPLE, 0.54),
        ("hydrology", "Hydrologic\nbehavior", HYDRO_BLUE, 0.76),
        ("benefit", "Water-resource\noutcomes", ROBUST_GREEN, 0.92),
    ]
    y_positions = {
        "learning_rate": 0.78,
        "clip_ratio": 0.58,
        "discount_factor": 0.38,
        "approx_kl": 0.72,
        "clipfrac": 0.47,
        "action_correction": 0.60,
        "low_level_pressure": 0.72,
        "conditional_spill": 0.45,
        "generation_return": 0.58,
    }
    node_xy: dict[str, tuple[float, float]] = {}
    for layer, title, color, x in layers:
        ax.text(x, 0.97, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=color)
        for _, row in nodes[nodes["layer"] == layer].iterrows():
            node_id = row["node_id"]
            y = y_positions.get(node_id, 0.5)
            node_xy[node_id] = (x, y)
            box = FancyBboxPatch(
                (x - 0.075, y - 0.055),
                0.15,
                0.09,
                boxstyle="round,pad=0.013,rounding_size=0.018",
                linewidth=1.2,
                edgecolor=color,
                facecolor="#F8FAFC",
            )
            ax.add_patch(box)
            ax.text(x, y, _short(row["label"], 14), ha="center", va="center", fontsize=10)

    for _, row in edges.iterrows():
        if row["source"] not in node_xy or row["target"] not in node_xy:
            continue
        x1, y1 = node_xy[row["source"]]
        x2, y2 = node_xy[row["target"]]
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.078, y1),
                (x2 - 0.078, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.15,
                color="#64748B",
                alpha=0.85,
                connectionstyle="arc3,rad=0.05",
            )
        )
    ax.text(
        0.5,
        0.06,
        "Interpretation chain: learning dynamics -> feasible actions -> reservoir levels, conditional spillage and generation.",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#334155",
    )

    mini_specs = [
        ("eval_return", "(a) Training return", ROBUST_GREEN),
        ("clipfrac", "(b) Clip fraction / update clipping", ALGO_ORANGE),
        ("low_pressure_reservoir_count", "(c) Stage C lower-bound pressure", RISK_RED),
    ]
    for idx, (metric, title, _color) in enumerate(mini_specs):
        mini_ax = fig.add_subplot(gs[idx, 1])
        sub = evidence[evidence["metric"] == metric]
        for config_id in PRIMARY_CONFIGS:
            line = sub[sub["config_id"] == config_id].sort_values("relative_time")
            if line.empty:
                continue
            mini_ax.plot(line["relative_time"], line["value_scaled"], color=config_color(config_id), linewidth=1.6, alpha=0.9, label=_clean_config_label(config_id))
        if metric == "clipfrac":
            kl = evidence[evidence["metric"] == "approx_kl"]
            for config_id in ["gamma099_clip02", "risk_lhs009"]:
                line = kl[kl["config_id"] == config_id].sort_values("relative_time")
                if not line.empty:
                    mini_ax.plot(line["relative_time"], line["value_scaled"], color=config_color(config_id), linewidth=1.2, linestyle="--", alpha=0.8)
        mini_ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        mini_ax.set_xlim(0, 1)
        mini_ax.set_ylabel("scaled")
        mini_ax.grid(axis="y", alpha=0.18, linestyle="--")
        if idx == 2:
            mini_ax.set_xlabel("Relative training/evaluation time")
        else:
            mini_ax.set_xticklabels([])
    fig.legend(
        handles=[Line2D([0], [0], color=config_color(c), lw=2, label=_clean_config_label(c)) for c in PRIMARY_CONFIGS],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        frameon=False,
        ncol=2,
    )
    paths = save_figure(fig, output_dir, "fig1_learning_hydrology_framework")
    write_note(
        output_dir,
        "fig1_learning_hydrology_framework",
        [
            "# Fig.1 learning-hydrology framework",
            "",
            "Source data: `fig1_framework_nodes.csv`, `fig1_framework_edges.csv`, `fig1_dynamic_evidence.csv`.",
            "Dynamic insets use the same line-chart format to connect training return, update clipping and Stage C water-level pressure. Total spillage is a neutral spillage diagnostic.",
        ],
    )
    return paths


def plot_fig2(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    fixed = _read("fig2_fixed_system.csv", data_dir)
    stages = _read("fig2_stage_design.csv", data_dir)
    ranges = _read("fig2_parameter_ranges.csv", data_dir)
    chain = _read("fig2_output_chain.csv", data_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))

    ax = axes[0, 0]
    panel_label(ax, "a", "Fixed hydrological and MARL system")
    ax.set_axis_off()
    for idx, row in fixed.iterrows():
        y = 0.88 - idx * 0.17
        ax.add_patch(FancyBboxPatch((0.05, y - 0.055), 0.90, 0.095, boxstyle="round,pad=0.014", facecolor="#EFF6FF", edgecolor=HYDRO_BLUE))
        ax.text(0.10, y, row["component"], fontweight="bold", va="center")
        ax.text(0.42, y, _short(row["value"], 18), va="center", color="#0F172A")
        ax.text(0.74, y, _short(row["description"], 22), va="center", fontsize=9.8, color="#334155")

    ax = axes[0, 1]
    panel_label(ax, "b", "Stage flow and evidence scale")
    ax.set_axis_off()
    colors = [NEUTRAL_GREY, STAGE_PURPLE, ALGO_ORANGE, ROBUST_GREEN]
    x_positions = np.linspace(0.10, 0.88, len(stages))
    max_units = max(stages["data_units"].max(), 1)
    for idx, (_, row) in enumerate(stages.iterrows()):
        x = x_positions[idx]
        height = 0.12 + 0.34 * row["data_units"] / max_units
        ax.add_patch(FancyBboxPatch((x - 0.08, 0.44 - height / 2), 0.16, height, boxstyle="round,pad=0.014", facecolor=colors[idx], alpha=0.88, edgecolor="white"))
        ax.text(x, 0.66, row["stage"], ha="center", fontweight="bold", color="#0F172A")
        ax.text(x, 0.44, f"{int(row['config_count'])} x {int(row['seed_or_eval_count'])}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x, 0.20, _short(row["output_role"], 20), ha="center", va="center", fontsize=9.5)
        if idx < len(stages) - 1:
            ax.annotate("", xy=(x_positions[idx + 1] - 0.10, 0.44), xytext=(x + 0.10, 0.44), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#64748B"))

    ax = axes[1, 0]
    panel_label(ax, "c", "Six controlled learning-dynamics parameters")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(ranges) - 0.5)
    ax.set_yticks(range(len(ranges)))
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in ranges["parameter"]])
    ax.set_xticks([])
    for idx, row in ranges.iterrows():
        values = [value.strip() for value in str(row["candidate_values"]).split(",")]
        xs = np.linspace(0.08, 0.90, len(values))
        ax.scatter(xs, [idx] * len(values), s=60, color=ALGO_ORANGE, zorder=3)
        ax.plot(xs, [idx] * len(values), color="#FED7AA", linewidth=3, zorder=1)
        for x, value in zip(xs, values):
            ax.text(x, idx + 0.20, value, ha="center", fontsize=8.8)
    ax.invert_yaxis()

    ax = axes[1, 1]
    panel_label(ax, "d", "Data products for algorithm-water interpretation")
    ax.set_axis_off()
    y_positions = np.linspace(0.82, 0.18, len(chain))
    for y, (_, row) in zip(y_positions, chain.iterrows()):
        ax.add_patch(FancyBboxPatch((0.05, y - 0.055), 0.90, 0.10, boxstyle="round,pad=0.012", facecolor="#F0FDFA", edgecolor=HYDRO_CYAN))
        ax.text(0.09, y, _short(row["input_source"], 20), va="center", fontsize=10, fontweight="bold")
        ax.text(0.39, y, _short(row["analysis_layer"], 22), va="center", fontsize=9.8)
        ax.text(0.66, y, _short(row["output_metrics"], 30), va="center", fontsize=9.5, color="#334155")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10, wspace=0.34, hspace=0.45)
    paths = save_figure(fig, output_dir, "fig2_experiment_design")
    write_note(
        output_dir,
        "fig2_experiment_design",
        [
            "# Fig.2 experiment design",
            "",
            "Source data: `fig2_fixed_system.csv`, `fig2_stage_design.csv`, `fig2_parameter_ranges.csv`, `fig2_output_chain.csv`.",
            "Panel b uses a flow strip to show how Stage A/B/C produce the algorithm-water evidence chain.",
        ],
    )
    return paths


def plot_fig4(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig4_parameter_importance.csv", data_dir)
    labels = [PARAM_LABELS[param] for param in PARAM_ORDER]
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 5.4), sharey=True)
    for ax, (metric, label, unit, direction), letter in zip(axes, FIG3_METRICS, "abcde"):
        sub = frame[frame["metric"] == metric].set_index("parameter").reindex(PARAM_ORDER)
        values = sub["spearman"]
        _lollipop_panel(
            ax,
            labels,
            values,
            f"({letter}) {label}",
            f"Spearman rho ({unit})",
            higher_better=(direction == "higher_better"),
        )
        ax.set_xlim(-1.0, 1.0)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    paths = save_figure(fig, output_dir, "fig4_parameter_importance")
    write_note(
        output_dir,
        "fig4_parameter_importance",
        [
            "# Fig.4 parameter importance",
            "",
            "Source data: `fig4_parameter_importance.csv`, `fig4_homogeneous_lollipop.csv`.",
            "All main-text panels use the same lollipop-plot format: y-axis is the six HAPPO learning-dynamics parameters and x-axis is Spearman rho. Positive rho is interpreted by metric direction; total spillage is not used as a risk axis.",
        ],
    )
    return paths


def plot_fig5(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig5_clip_gamma_3d.csv", data_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex=True, sharey=True)
    for ax, (z_col, label, unit, direction, cmap), letter in zip(axes.ravel(), FIG4_METRICS, "abcd"):
        _sparse_contour(ax, frame, z_col, f"({letter}) {label}", cmap, unit)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    paths = save_figure(fig, output_dir, "fig5_clip_gamma_response")
    write_note(
        output_dir,
        "fig5_clip_gamma_response",
        [
            "# Fig.5 clip-gamma response",
            "",
            "Source data: `fig5_clip_gamma_3d.csv`, `fig5_homogeneous_contour.csv`.",
            "All panels use the same sparse contour/scatter-plot format with clip ratio on x and discount factor on y. The 3D view is moved to Supplementary Fig.S2.",
        ],
    )
    return paths


def plot_fig6(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig6_gamma_gae_seasonal_response.csv", data_dir)
    row_specs = [
        ("season_power", "Generation", "10$^{8}$ kWh", True),
        ("low_level_pressure_days", "Overall lower-bound pressure", "reservoir-days", False),
        ("season_spill_neutral", "Seasonal spill diagnostic", "10$^{8}$ m$^{3}$; neutral", False),
    ]
    gae_values = sorted(pd.to_numeric(frame["gae_lambda"], errors="coerce").dropna().unique())
    gae_colors = {gae: color for gae, color in zip(gae_values, [HYDRO_BLUE, ALGO_ORANGE, RISK_RED, ROBUST_GREEN])}
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.2), sharex=True)
    for row_idx, (metric, metric_label, unit, higher_better) in enumerate(row_specs):
        for col_idx, season in enumerate(SEASON_ORDER):
            ax = axes[row_idx, col_idx]
            sub = frame[frame["season"] == season].copy()
            grouped = (
                sub.groupby(["discount_factor", "gae_lambda"], dropna=False)
                .agg(mean_value=(metric, "mean"), sd_value=(metric, "std"))
                .reset_index()
                .sort_values(["gae_lambda", "discount_factor"])
            )
            for gae in gae_values:
                line = grouped[np.isclose(pd.to_numeric(grouped["gae_lambda"], errors="coerce"), gae)]
                if line.empty:
                    continue
                ax.errorbar(
                    line["discount_factor"],
                    line["mean_value"],
                    yerr=line["sd_value"].fillna(0),
                    marker="o",
                    linewidth=1.55,
                    capsize=2.5,
                    color=gae_colors.get(gae, DEFAULT_GREY),
                    label=f"GAE={gae:g}",
                )
            title = f"({chr(97 + row_idx * 3 + col_idx)}) {SEASON_LABELS[season]}" if row_idx == 0 else f"({chr(97 + row_idx * 3 + col_idx)})"
            ax.set_title(title, loc="left", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{metric_label}\n({unit})")
            if row_idx == 2:
                ax.set_xlabel("Discount factor")
            ax.grid(axis="y", alpha=0.22, linestyle="--")
            if not higher_better and metric == "season_spill_neutral":
                ax.text(0.02, 0.92, "neutral spillage diagnostic", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=NEUTRAL_GREY)
            if metric == "low_level_pressure_days":
                ax.text(0.02, 0.92, "run-level diagnostic", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=NEUTRAL_GREY)

    handles = [Line2D([0], [0], color=gae_colors.get(gae, DEFAULT_GREY), marker="o", lw=1.55, label=f"GAE={gae:g}") for gae in gae_values]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(gae_values), frameon=False, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = save_figure(fig, output_dir, "fig6_gamma_gae_seasonal_mechanism")
    write_note(
        output_dir,
        "fig6_gamma_gae_seasonal_mechanism",
        [
            "# Fig.6 seasonal mechanism",
            "",
            "Source data: `fig6_gamma_gae_seasonal_response.csv`, `fig6_homogeneous_seasonal.csv`.",
            "All panels use the same point-line-with-error-bar format. Seasonal spill is explicitly a neutral spillage diagnostic; water-level pressure carries risk interpretation.",
        ],
    )
    return paths


def plot_fig7(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig7_mechanism_timeseries.csv", data_dir)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["water_state_scaled"] = pd.to_numeric(frame["normalized_water_state_mean"], errors="coerce").clip(-0.1, 1.1)
    for col in ["outflow_m3s_sum", "spill_flow_m3s_sum", "action_correction_mean", "low_pressure_reservoir_count", "violation_reservoir_count"]:
        frame[f"{col}_scaled"] = _scale(frame[col])
    configs = [config_id for config_id in PRIMARY_CONFIGS if config_id in set(frame["config_id"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.8), sharex=True, sharey=True)
    legend_handles = [
        Line2D([0], [0], color=HYDRO_BLUE, lw=1.8, label="Water state"),
        Line2D([0], [0], color=HYDRO_CYAN, lw=1.4, label="Outflow"),
        Line2D([0], [0], color=NEUTRAL_GREY, lw=1.4, label="Spill diagnostic"),
        Line2D([0], [0], color=ALGO_ORANGE, lw=1.4, label="Action correction"),
        Line2D([0], [0], marker="|", color=RISK_RED, linestyle="None", markersize=10, label="Lower-bound event"),
        Line2D([0], [0], marker="x", color="#111827", linestyle="None", markersize=5, label="Violation event"),
    ]
    for idx, (ax, config_id) in enumerate(zip(axes.ravel(), configs)):
        sub = frame[frame["config_id"] == config_id].sort_values("date").copy()
        dates = sub["date"]
        _shade_flood_seasons(ax, dates)
        ax.plot(dates, _rolling(sub["water_state_scaled"]), color=HYDRO_BLUE, linewidth=1.8)
        ax.plot(dates, _rolling(sub["outflow_m3s_sum_scaled"]), color=HYDRO_CYAN, linewidth=1.25, alpha=0.82)
        ax.plot(dates, _rolling(sub["spill_flow_m3s_sum_scaled"]), color=NEUTRAL_GREY, linewidth=1.25, alpha=0.72)
        ax.plot(dates, _rolling(sub["action_correction_mean_scaled"]), color=ALGO_ORANGE, linewidth=1.25, alpha=0.9)
        low_event = pd.to_numeric(sub["low_pressure_reservoir_count"], errors="coerce").fillna(0) > 0
        violation_event = pd.to_numeric(sub["violation_reservoir_count"], errors="coerce").fillna(0) > 0
        ax.vlines(dates[low_event], 1.03, 1.10, color=RISK_RED, alpha=0.45, linewidth=0.65)
        ax.scatter(dates[violation_event], np.full(int(violation_event.sum()), 1.13), marker="x", s=12, color="#111827", linewidth=0.7)
        ax.axhline(0, color="#94A3B8", linewidth=0.6, linestyle=":")
        ax.axhline(1, color="#94A3B8", linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.05, 1.18)
        ax.set_title(f"({chr(97 + idx)}) {_clean_config_label(config_id)}", loc="left", fontweight="bold", color=config_color(config_id))
        ax.grid(axis="y", alpha=0.18, linestyle="--")
        if idx in {2, 3}:
            ax.set_xlabel("Date")
        if idx in {0, 2}:
            ax.set_ylabel("Normalized daily signal")
        ax.tick_params(axis="x", rotation=25)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=6, frameon=False, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_figure(fig, output_dir, "fig7_representative_operation_mechanisms")
    write_note(
        output_dir,
        "fig7_representative_operation_mechanisms",
        [
            "# Fig.7 representative operation mechanisms",
            "",
            "Source data: `fig7_mechanism_timeseries.csv`, `fig7_homogeneous_timeseries.csv`, `fig7_stage_c_reservoir_daily.csv`.",
            "Each panel uses the same normalized-trace format for a representative configuration; flood-season shading is June-September and spill is a neutral spillage diagnostic.",
        ],
    )
    return paths


def plot_fig8(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig8_reservoir_heterogeneity.csv", data_dir)
    configs = [config_id for config_id in PRIMARY_CONFIGS if config_id in set(frame["config_id"].astype(str))]
    reservoir_order = (
        frame[["reservoir", "reservoir_label", "reservoir_order"]]
        .drop_duplicates()
        .sort_values("reservoir_order")
    )
    reservoirs = reservoir_order["reservoir"].tolist()
    reservoir_labels = dict(zip(reservoir_order["reservoir"], reservoir_order["reservoir_label"]))
    metric_specs = [
        ("generation_100mkwh_sum", "Generation", "10$^{8}$ kWh", "viridis", ".0f"),
        ("low_pressure_days", "Lower-bound pressure days", "days", "YlOrRd", ".0f"),
        ("action_correction_mean", "Action correction", "normalized action", "YlOrBr", ".3f"),
        ("remaining_storage_ratio_mean", "Remaining storage", "fraction", "Blues", ".2f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
    for ax, (metric, label, unit, cmap, fmt), letter in zip(axes.ravel(), metric_specs, "abcd"):
        pivot = (
            frame.pivot_table(index="reservoir", columns="config_id", values=metric, aggfunc="mean")
            .reindex(reservoirs)
            .reindex(columns=configs)
        )
        pivot.index = [reservoir_labels.get(name, name) for name in pivot.index]
        pivot.columns = [_clean_config_label(col) for col in pivot.columns]
        _annotated_heatmap(ax, pivot, cmap, f"({letter}) {label}", fmt=fmt, cbar_label=f"{label} ({unit})")
        ax.set_xlabel("Representative configuration")
        ax.set_ylabel("Reservoir")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    paths = save_figure(fig, output_dir, "fig8_reservoir_heterogeneity")
    write_note(
        output_dir,
        "fig8_reservoir_heterogeneity",
        [
            "# Fig.8 reservoir-level heterogeneity",
            "",
            "Source data: `fig8_reservoir_heterogeneity.csv`, `fig8_homogeneous_matrix.csv`.",
            "All panels are 5-reservoir x 4-representative-configuration heatmaps, enabling direct upstream-downstream comparison.",
        ],
    )
    return paths


def plot_fig10(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig10_mixed_sequence_validation.csv", data_dir)
    frame = frame.set_index("config_id").reindex(_ordered_configs(frame)).reset_index()
    metric_specs = [
        ("eval_return_delta_vs_default_pct", "Return delta", "vs Baseline (%)", True),
        ("low_level_pressure_days_delta_vs_default_pct", "Lower-bound pressure delta", "vs Baseline (%)", False),
        ("boundary_pressure_spill_delta_vs_default_pct", "Boundary-pressure spillage delta", "vs Baseline (%)", False),
        ("non_flood_spill_delta_vs_default_pct", "Non-flood-season spillage delta", "vs Baseline (%)", False),
    ]
    labels = [_clean_config_label(config_id) for config_id in frame["config_id"]]
    y = np.arange(len(frame))
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), sharey=True)
    for ax, (metric, label, unit, higher_better), letter in zip(axes.ravel(), metric_specs, "abcd"):
        values = pd.to_numeric(frame[metric], errors="coerce").fillna(0)
        _lollipop_panel(ax, labels, values, f"({letter}) {label}", unit, higher_better=higher_better)
        for yi, value, config_id, role in zip(y, values, frame["config_id"], frame["stage_c_role"]):
            ax.scatter(value, yi, s=96, color=config_color(config_id), marker=role_marker(role), edgecolor="white", linewidth=0.8, zorder=4)
        if ax not in axes[:, 0]:
            ax.set_ylabel("")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    paths = save_figure(fig, output_dir, "fig10_mixed_sequence_validation")
    write_note(
        output_dir,
        "fig10_mixed_sequence_validation",
        [
            "# Fig.10 Stage C mixed-sequence checkpoint validation",
            "",
            "Source data: `fig10_mixed_sequence_validation.csv`, `fig10_homogeneous_lollipop.csv`.",
            "All panels use Baseline-relative lollipops. This is Stage C mixed-sequence checkpoint validation, not multi-scenario robustness; total spillage remains a neutral spillage diagnostic.",
        ],
    )
    return paths


def plot_fig9(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig9_pareto_diagnostics.csv", data_dir)
    metric_specs = [
        ("low_level_pressure_days_mean", "Lower-bound pressure", "reservoir-days"),
        ("mean_action_correction_mean", "Action correction", "normalized action"),
        ("any_violation_rate_mean", "Violation rate", "fraction"),
        ("boundary_pressure_spill_mean", "Boundary-pressure spill", "10$^{8}$ m$^{3}$"),
    ]
    marker_map = {0.1: "o", 0.2: "s", 0.3: "^"}
    vmin = frame["discount_factor"].min()
    vmax = frame["discount_factor"].max()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), sharey=True)
    last_scatter = None
    for ax, (metric, label, unit), letter in zip(axes.ravel(), metric_specs, "abcd"):
        for clip, marker in marker_map.items():
            sub = frame[np.isclose(pd.to_numeric(frame["clip_ratio"], errors="coerce"), clip)]
            if sub.empty:
                continue
            last_scatter = ax.scatter(
                sub[metric],
                sub["return_mean"],
                c=sub["discount_factor"],
                cmap="viridis",
                marker=marker,
                s=82,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax,
                label=f"clip={clip}",
            )
        for _, row in frame.iterrows():
            if row["config_id"] in {"default", "gamma099_clip02", "gamma0999_clip02", "risk_lhs009", "clip03_gamma995"}:
                ax.text(row[metric], row["return_mean"], f" {POINT_LABELS.get(str(row['config_id']), row['config_id'])}", fontsize=11.5, fontweight="bold", va="center")
        ax.set_title(f"({letter}) Return vs {label}", loc="left", fontweight="bold")
        ax.set_xlabel(f"{label} ({unit}; lower is better)")
        ax.set_ylabel("Evaluator return (higher is better)")
        ax.grid(alpha=0.22, linestyle="--")
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
        cbar.set_label("Discount factor")
    handles = [Line2D([0], [0], marker=marker, linestyle="None", markerfacecolor="#64748B", markeredgecolor="white", markersize=12, label=f"clip={clip}") for clip, marker in marker_map.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.92, bottom=0.08, wspace=0.18, hspace=0.34)
    paths = save_figure(fig, output_dir, "fig9_pareto_acceptability_diagnostics")
    write_note(
        output_dir,
        "fig9_pareto_acceptability_diagnostics",
        [
            "# Fig.9 multi-objective acceptability diagnostics",
            "",
            "Source data: `fig9_pareto_diagnostics.csv`, `fig9_homogeneous_scatter.csv`.",
            "All panels use the same scatter projection: y-axis is return, x-axis is one operational risk metric, color is discount factor, and marker shape is clip ratio. 3D and parallel-coordinate diagnostics are moved to Supplementary Fig.S3.",
        ],
    )
    return paths


def plot_figS1_parameter_diagnostics(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig4_parameter_importance.csv", data_dir)
    direction = _read("fig4_parameter_direction.csv", data_dir)
    top_params = ["discount_factor", "clip_ratio", "gae_lambda"]
    metric_order = [metric for metric, _label, _unit, _direction in FIG3_METRICS]
    metric_labels = [label for _metric, label, _unit, _direction in FIG3_METRICS]
    fig = plt.figure(figsize=(12.4, 5.8))
    ax_radar = fig.add_subplot(1, 2, 1, projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(metric_order), endpoint=False).tolist()
    angles += angles[:1]
    for param in top_params:
        sub = frame[frame["parameter"] == param].set_index("metric").reindex(metric_order)
        values = sub["abs_spearman"].fillna(0).tolist()
        values += values[:1]
        ax_radar.plot(angles, values, linewidth=2, label=PARAM_LABELS[param], color=ALGO_ORANGE if param in {"discount_factor", "clip_ratio"} else HYDRO_BLUE)
        ax_radar.fill(angles, values, alpha=0.08)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, fontsize=9.5)
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_title("(a) Sensitivity fingerprint", loc="left", fontweight="bold", pad=18)
    ax_radar.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.36, 1.16))

    ax = fig.add_subplot(1, 2, 2)
    key = direction[direction["parameter"].isin(["clip_ratio", "discount_factor"]) & direction["metric"].isin(["return", "low_level_pressure_days"])].copy()
    for param, offset in [("clip_ratio", 0), ("discount_factor", 4)]:
        sub = key[key["parameter"] == param].copy()
        for metric, color, marker in [("return", ROBUST_GREEN, "o"), ("low_level_pressure_days", RISK_RED, "s")]:
            line = sub[sub["metric"] == metric].sort_values("parameter_value")
            if line.empty:
                continue
            xs = np.arange(len(line)) + offset
            ys = _minmax(line["metric_value"], higher_better=(metric == "return"))
            ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8, label=metric if offset == 0 else None)
            for x, value in zip(xs, line["parameter_value"]):
                ax.text(x, -0.08, str(value), ha="center", va="top", fontsize=8.5, rotation=30)
        ax.text(offset + 1, 1.07, PARAM_LABELS[param], ha="center", fontweight="bold", color="#334155")
    ax.set_ylim(-0.18, 1.15)
    ax.set_xticks([])
    ax.set_ylabel("Normalized desirability")
    ax.set_title("(b) Directional response of key parameters", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "figS1_parameter_diagnostic_supplement")
    write_note(
        output_dir,
        "figS1_parameter_diagnostic_supplement",
        [
            "# Supplementary Fig.S1 parameter diagnostics",
            "",
            "Source data: `fig4_parameter_importance.csv`, `fig4_parameter_direction.csv`.",
            "Radar and directional-response views are retained as supplementary diagnostics; the main Fig.4 uses homogeneous lollipop panels.",
        ],
    )
    return paths


def plot_figS2_clip_gamma_3d(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig5_clip_gamma_3d.csv", data_dir)
    fig = plt.figure(figsize=(12.6, 5.6))
    specs = [
        ("return_mean", "Return", "viridis"),
        ("low_level_pressure_days_mean", "Lower-bound pressure", "YlOrRd"),
    ]
    for idx, (metric, label, cmap) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        sub = frame[["clip_ratio", "discount_factor", metric]].dropna()
        x = pd.to_numeric(sub["clip_ratio"], errors="coerce")
        y = pd.to_numeric(sub["discount_factor"], errors="coerce")
        z = pd.to_numeric(sub[metric], errors="coerce")
        valid = x.notna() & y.notna() & z.notna()
        x, y, z = x[valid], y[valid], z[valid]
        if len(x) >= 3 and x.nunique() >= 2 and y.nunique() >= 2:
            try:
                triang = mtri.Triangulation(x.to_numpy(), y.to_numpy())
                ax.plot_trisurf(triang, z.to_numpy(), cmap=cmap, alpha=0.82, edgecolor="#FFFFFF", linewidth=0.35)
            except Exception:
                ax.scatter(x, y, z, c=z, cmap=cmap, s=45, edgecolor="white", linewidth=0.6)
        else:
            ax.scatter(x, y, z, c=z, cmap=cmap, s=45, edgecolor="white", linewidth=0.6)
        ax.set_title(f"({chr(96 + idx)}) {label} 3D surface", loc="left", fontweight="bold")
        ax.set_xlabel("Clip ratio")
        ax.set_ylabel("Discount factor")
        ax.set_zlabel(label)
        ax.view_init(elev=24, azim=-56)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "figS2_clip_gamma_3d_supplement")
    write_note(
        output_dir,
        "figS2_clip_gamma_3d_supplement",
        [
            "# Supplementary Fig.S2 clip-gamma 3D diagnostics",
            "",
            "Source data: `fig5_clip_gamma_3d.csv`.",
            "The 3D view is supplementary because the main-text Fig.5 uses readable 2D contour panels for water-resource audiences.",
        ],
    )
    return paths


def plot_figS3_parallel_robustness(data_dir: Path = FIGURE_DATA_DIR, output_dir: Path = FIGURE_OUTPUT_DIR) -> list[Path]:
    apply_style()
    frame = _read("fig9_pareto_diagnostics.csv", data_dir).copy()
    metrics = [
        ("return_mean", "Return", True),
        ("mean_action_correction_mean", "Action\nfeasibility", False),
        ("low_level_pressure_days_mean", "Water-level\npressure", False),
        ("any_violation_rate_mean", "Constraint\ncompliance", False),
        ("acceptability_score", "Composite\nacceptability", True),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    xs = np.arange(len(metrics))
    for _, row in frame.iterrows():
        scores = [float(_minmax(frame[col], higher_better=higher).loc[row.name]) for col, _label, higher in metrics]
        highlight = row["config_id"] in {"default", "gamma099_clip02", "gamma0999_clip02", "risk_lhs009", "clip03_gamma995"}
        ax.plot(xs, scores, color=config_color(row["config_id"]), linewidth=2.6 if highlight else 1.0, alpha=0.94 if highlight else 0.32)
        if highlight:
            ax.text(xs[-1] + 0.05, scores[-1], _clean_config_label(row["config_id"]), fontsize=8.7, va="center", color=config_color(row["config_id"]))
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _col, label, _higher in metrics])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized acceptability (higher is better)")
    ax.set_title("Supplementary multi-objective acceptability profile", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "figS3_parallel_acceptability_supplement")
    write_note(
        output_dir,
        "figS3_parallel_acceptability_supplement",
        [
            "# Supplementary Fig.S3 parallel-coordinate acceptability diagnostics",
            "",
            "Source data: `fig9_pareto_diagnostics.csv`.",
            "Parallel-coordinate diagnostics are retained as supplementary evidence; main Fig.9 uses homogeneous scatter projections.",
        ],
    )
    return paths
