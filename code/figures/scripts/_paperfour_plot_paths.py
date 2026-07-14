from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
FIGURE_DATA_DIR = FIGURE_DIR / "data"
FIGURE_OUTPUT_DIR = FIGURE_DIR / "output"
PAPERFOUR_DIR = FIGURE_DIR.parent
MARKDOWN_DIR = PAPERFOUR_DIR.parent.parent
RESERVOIR_DIR = MARKDOWN_DIR.parent
ANALYSIS_DIR = RESERVOIR_DIR / "analysis"
PAPERFOUR_ANALYSIS_DIR = ANALYSIS_DIR / "experiment_analysis_paperFour"
STAGE_B_OUTPUT_DIR = PAPERFOUR_ANALYSIS_DIR / "output_stage_b"
STAGE_C_OUTPUT_DIR = PAPERFOUR_ANALYSIS_DIR / "output_stage_c"
STAGE_C_HYDROLOGY_DIR = STAGE_C_OUTPUT_DIR / "hydrology" / "mixed_percentile_5yr"
RESULT_PAPERFOUR_EVAL_DIR = RESERVOIR_DIR / "result_paperFour_eval"


def ensure_figure_dirs() -> None:
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

