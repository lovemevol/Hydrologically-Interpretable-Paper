from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


FONT_FAMILY = ["Times New Roman", "DejaVu Serif"]
BASE_FONT_SIZE = 15
SAVE_DPI = 320

HYDRO_BLUE = "#2A7FB8"
HYDRO_CYAN = "#40B7AD"
ALGO_ORANGE = "#D97706"
ALGO_RED = "#B91C1C"
ROBUST_GREEN = "#166534"
RISK_RED = "#991B1B"
DEFAULT_GREY = "#6B7280"
NEUTRAL_GREY = "#9CA3AF"
STAGE_PURPLE = "#5B5F97"

CONFIG_COLORS = {
    "default": DEFAULT_GREY,
    "gamma099_clip02": ROBUST_GREEN,
    "return_lhs012": "#2F855A",
    "return_lhs008": "#38A169",
    "entropy004_robust": "#0F766E",
    "clip03_gamma995": "#C2410C",
    "gamma0999_clip02": "#B91C1C",
    "risk_lhs009": RISK_RED,
}

ROLE_MARKERS = {
    "default_reference": "o",
    "robust_top_and_best_return": "s",
    "high_return_low_clip_candidate": "^",
    "stage_a_return_leader_check": "D",
    "high_clip_risk_contrast": "v",
    "long_horizon_risk_contrast": "X",
    "failure_counterexample": "P",
    "high_screening_but_high_cv_uncertainty": "h",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 1,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "figure.titlesize": BASE_FONT_SIZE + 1,
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", loc="left", fontweight="bold")


def save_figure(fig: plt.Figure, output_dir: Path, base_name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_note(output_dir: Path, base_name: str, lines: Iterable[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / f"{base_name}.md"
    note_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return note_path


def config_color(config_id: str) -> str:
    return CONFIG_COLORS.get(str(config_id), "#475569")


def role_marker(role: str) -> str:
    return ROLE_MARKERS.get(str(role), "o")


def config_legend(config_ids: list[str]) -> list[Patch]:
    return [Patch(facecolor=config_color(config_id), edgecolor="none", label=config_id) for config_id in config_ids]
