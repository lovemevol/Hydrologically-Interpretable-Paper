#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parameter utilities for PaperFour HAPPO learning-dynamics experiments.

PaperFour fixes the hydrological system and algorithm, then varies only six
HAPPO learning-dynamics parameters.  This module intentionally contains no
training code; it only defines deterministic stage designs used by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PAPERFOUR_RESULT_ROOT = "agent/reservoir_multi_agents/result_paperFour"

FIXED_RES_STR_FILE = "ResStr5.xlsx"
FIXED_OBSERVATION_PARADIGM = "fixed_observation_setting"
FIXED_ACTION_SPACE_TYPE = "discrete"
FIXED_ACTION_DIM = 100
FIXED_LOCAL_RATIO = 0.5
FIXED_REWARD_WEIGHTING_MODE = "none"
FIXED_MAX_CYCLES = 1825
FIXED_DATA_EXPORT_MODE = "excel"

DEFAULT_MAX_EPISODES = 50
DEFAULT_COLLECTOR_ENV_NUM = 10
STAGE0_MAX_EPISODES = 2
STAGE0_COLLECTOR_ENV_NUM = 1

LEARNING_PARAM_NAMES: Tuple[str, ...] = (
    "learning_rate",
    "critic_learning_rate",
    "clip_ratio",
    "entropy_weight",
    "discount_factor",
    "gae_lambda",
)

DEFAULT_LEARNING_PARAMS: Dict[str, float] = {
    "learning_rate": 5e-4,
    "critic_learning_rate": 5e-4,
    "clip_ratio": 0.2,
    "entropy_weight": 0.016,
    "discount_factor": 0.995,
    "gae_lambda": 0.95,
}

CANDIDATE_VALUES: Dict[str, Tuple[float, ...]] = {
    "learning_rate": (1e-4, 3e-4, 5e-4, 1e-3),
    "critic_learning_rate": (1e-4, 3e-4, 5e-4, 1e-3),
    "clip_ratio": (0.1, 0.2, 0.3),
    "entropy_weight": (0.004, 0.008, 0.016, 0.032),
    "discount_factor": (0.99, 0.995, 0.999),
    "gae_lambda": (0.90, 0.95, 0.98),
}

STAGE_LABELS: Dict[str, str] = {
    "S0": "Stage 0 smoke test",
    "A": "Stage A screening design",
    "B": "Stage B main experiment design",
}

STAGE_BUDGETS: Dict[str, Dict[str, int]] = {
    "S0": {
        "MAX_EPISODES": STAGE0_MAX_EPISODES,
        "COLLECTOR_ENV_NUM": STAGE0_COLLECTOR_ENV_NUM,
        "DATA_EXPORT_COUNT": 2,
    },
    "A": {
        "MAX_EPISODES": DEFAULT_MAX_EPISODES,
        "COLLECTOR_ENV_NUM": DEFAULT_COLLECTOR_ENV_NUM,
        "DATA_EXPORT_COUNT": 10,
    },
    "B": {
        "MAX_EPISODES": DEFAULT_MAX_EPISODES,
        "COLLECTOR_ENV_NUM": DEFAULT_COLLECTOR_ENV_NUM,
        "DATA_EXPORT_COUNT": 10,
    },
}


@dataclass(frozen=True)
class PaperFourParamSet:
    """One learning-dynamics parameter setting in a PaperFour stage."""

    stage: str
    config_id: str
    description: str
    params: Mapping[str, float]

    def ordered_params(self) -> Dict[str, float]:
        """Return parameters in the canonical PaperFour order."""
        return {name: float(self.params[name]) for name in LEARNING_PARAM_NAMES}


def _ordered_params(params: Mapping[str, float]) -> Dict[str, float]:
    merged = dict(DEFAULT_LEARNING_PARAMS)
    merged.update(params)
    return {name: float(merged[name]) for name in LEARNING_PARAM_NAMES}


def build_stage0_param_sets() -> List[PaperFourParamSet]:
    """Return the three smoke-test configurations required before Stage A."""
    return [
        PaperFourParamSet(
            stage="S0",
            config_id="default",
            description="Default HAPPO learning dynamics",
            params=_ordered_params({}),
        ),
        PaperFourParamSet(
            stage="S0",
            config_id="high_lr_high_clip",
            description="Aggressive update stress test: high actor/critic LR and high PPO clip",
            params=_ordered_params(
                {
                    "learning_rate": 1e-3,
                    "critic_learning_rate": 1e-3,
                    "clip_ratio": 0.3,
                }
            ),
        ),
        PaperFourParamSet(
            stage="S0",
            config_id="high_entropy_low_discount",
            description="Exploration and short-horizon stress test: high entropy and low discount",
            params=_ordered_params(
                {
                    "entropy_weight": 0.032,
                    "discount_factor": 0.99,
                }
            ),
        ),
    ]


def _candidate_grid() -> List[Dict[str, float]]:
    values = [CANDIDATE_VALUES[name] for name in LEARNING_PARAM_NAMES]
    return [
        {name: float(value) for name, value in zip(LEARNING_PARAM_NAMES, combo)}
        for combo in product(*values)
    ]


def _sample_evenly(rows: Sequence[Dict[str, float]], count: int) -> List[Dict[str, float]]:
    if count >= len(rows):
        return list(rows)
    if count <= 1:
        return [rows[0]]

    selected: List[Dict[str, float]] = []
    used_indices = set()
    for i in range(count):
        idx = round(i * (len(rows) - 1) / (count - 1))
        used_indices.add(idx)
        selected.append(rows[idx])

    # Rounding can theoretically duplicate indices. Fill any gap deterministically.
    idx = 0
    while len(selected) < count and idx < len(rows):
        if idx not in used_indices:
            selected.append(rows[idx])
            used_indices.add(idx)
        idx += 1
    return selected[:count]


def build_stage_a_param_sets(count: int = 24) -> List[PaperFourParamSet]:
    """Build a deterministic coarse screening design.

    The design is a reproducible spread sample from the full candidate grid plus
    an explicit default row, keeping the total size in the guide's 18-30 range.
    """
    if count < 3:
        raise ValueError("Stage A needs at least three configurations.")

    default = DEFAULT_LEARNING_PARAMS
    rows: List[PaperFourParamSet] = [
        PaperFourParamSet(
            stage="A",
            config_id="default",
            description="Default HAPPO learning dynamics repeated inside Stage A",
            params=_ordered_params(default),
        )
    ]

    grid = _candidate_grid()
    sampled = _sample_evenly(grid, count - 1)
    for params in sampled:
        if _ordered_params(params) == _ordered_params(default):
            continue
        rows.append(
            PaperFourParamSet(
                stage="A",
                config_id=f"lhs{len(rows):03d}",
                description="Deterministic coarse screening sample from the six-parameter grid",
                params=_ordered_params(params),
            )
        )
        if len(rows) >= count:
            break

    for params in grid:
        if len(rows) >= count:
            break
        ordered = _ordered_params(params)
        if any(existing.ordered_params() == ordered for existing in rows):
            continue
        rows.append(
            PaperFourParamSet(
                stage="A",
                config_id=f"lhs{len(rows):03d}",
                description="Deterministic coarse screening sample from the six-parameter grid",
                params=ordered,
            )
        )

    return rows


def build_stage_b_param_sets() -> List[PaperFourParamSet]:
    """Build the confirmatory Stage B design from Stage A screening evidence.

    Stage A identified clip_ratio and discount_factor as the strongest
    return-sensitive axes.  Stage B keeps the design focused on those learning
    dynamics while preserving robust, default, and stress-test representatives.
    """
    rows = [
        ("default", "Default reference", {}),
        ("robust_lhs014", "Stage A robust screening candidate lhs014", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.008,
            "discount_factor": 0.995,
            "gae_lambda": 0.90,
        }),
        ("return_lhs012", "Stage A highest-return candidate lhs012", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 1e-4,
            "clip_ratio": 0.1,
            "entropy_weight": 0.004,
            "discount_factor": 0.99,
            "gae_lambda": 0.90,
        }),
        ("return_lhs008", "Stage A high-return operationally acceptable candidate lhs008", {
            "learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.1,
            "entropy_weight": 0.008,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
        }),
        ("fast_actor_lhs019", "Stage A high actor-learning-rate candidate lhs019", {
            "learning_rate": 1e-3,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.1,
            "entropy_weight": 0.008,
            "discount_factor": 0.99,
            "gae_lambda": 0.90,
        }),
        ("screen_lhs007", "Stage A top screening-score candidate lhs007", {
            "learning_rate": 3e-4,
            "critic_learning_rate": 1e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.004,
            "discount_factor": 0.995,
            "gae_lambda": 0.90,
        }),
        ("conservative_lhs001", "Stage A conservative low-learning-rate candidate lhs001", {
            "learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "clip_ratio": 0.1,
            "entropy_weight": 0.004,
            "discount_factor": 0.99,
            "gae_lambda": 0.90,
        }),
        ("clip01_gamma995", "Clip-ratio main-effect candidate at gamma 0.995", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.1,
            "entropy_weight": 0.008,
            "discount_factor": 0.995,
            "gae_lambda": 0.90,
        }),
        ("clip03_gamma995", "High-clip main-effect contrast at gamma 0.995", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.3,
            "entropy_weight": 0.008,
            "discount_factor": 0.995,
            "gae_lambda": 0.90,
        }),
        ("gamma099_clip02", "Discount-factor main-effect candidate at clip 0.2", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.008,
            "discount_factor": 0.99,
            "gae_lambda": 0.90,
        }),
        ("gamma0999_clip02", "Long-horizon risk contrast at clip 0.2", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.008,
            "discount_factor": 0.999,
            "gae_lambda": 0.90,
        }),
        ("gae095_robust", "GAE main-effect candidate around the robust region", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.008,
            "discount_factor": 0.995,
            "gae_lambda": 0.95,
        }),
        ("gae098_robust", "High-GAE contrast around the robust region", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.008,
            "discount_factor": 0.995,
            "gae_lambda": 0.98,
        }),
        ("entropy004_robust", "Low-entropy exploration contrast around the robust region", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.004,
            "discount_factor": 0.995,
            "gae_lambda": 0.90,
        }),
        ("entropy032_robust", "High-entropy exploration contrast around the robust region", {
            "learning_rate": 5e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.032,
            "discount_factor": 0.995,
            "gae_lambda": 0.90,
        }),
        ("risk_lhs009", "Stage A weakest-return and non-flood spill stress case lhs009", {
            "learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.3,
            "entropy_weight": 0.008,
            "discount_factor": 0.999,
            "gae_lambda": 0.95,
        }),
        ("risk_lhs005", "Stage A low-return lower-bound-pressure stress case lhs005", {
            "learning_rate": 1e-4,
            "critic_learning_rate": 5e-4,
            "clip_ratio": 0.3,
            "entropy_weight": 0.016,
            "discount_factor": 0.999,
            "gae_lambda": 0.98,
        }),
        ("risk_lhs023", "Stage A aggressive unstable stress case lhs023", {
            "learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "clip_ratio": 0.3,
            "entropy_weight": 0.032,
            "discount_factor": 0.999,
            "gae_lambda": 0.98,
        }),
        ("risk_lhs003", "Stage A weak middle-clip stress case lhs003", {
            "learning_rate": 1e-4,
            "critic_learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "entropy_weight": 0.008,
            "discount_factor": 0.995,
            "gae_lambda": 0.95,
        }),
    ]
    return [
        PaperFourParamSet(stage="B", config_id=config_id, description=description, params=_ordered_params(params))
        for config_id, description, params in rows
    ]


def build_learning_param_sets(stage: str | None = None) -> List[PaperFourParamSet]:
    """Return PaperFour learning-parameter settings for one or all stages."""
    stage_builders = {
        "S0": build_stage0_param_sets,
        "A": build_stage_a_param_sets,
        "B": build_stage_b_param_sets,
    }
    if stage is not None:
        normalized = stage.upper()
        if normalized not in stage_builders:
            raise ValueError(f"Unknown PaperFour stage: {stage}")
        return stage_builders[normalized]()

    rows: List[PaperFourParamSet] = []
    for builder in stage_builders.values():
        rows.extend(builder())
    return rows


def format_learning_params(params: Mapping[str, float]) -> str:
    """Format the six learning parameters for CLI listing."""
    ordered = _ordered_params(params)
    return ", ".join(f"{name}={ordered[name]:g}" for name in LEARNING_PARAM_NAMES)


def validate_learning_params(params: Mapping[str, float]) -> None:
    """Fail fast if a parameter set varies anything outside the PaperFour scope."""
    missing = [name for name in LEARNING_PARAM_NAMES if name not in params]
    extra = [name for name in params if name not in LEARNING_PARAM_NAMES]
    if missing:
        raise ValueError(f"Missing PaperFour learning parameter(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected non-PaperFour parameter(s): {', '.join(extra)}")


__all__ = [
    "CANDIDATE_VALUES",
    "DEFAULT_COLLECTOR_ENV_NUM",
    "DEFAULT_LEARNING_PARAMS",
    "DEFAULT_MAX_EPISODES",
    "FIXED_ACTION_DIM",
    "FIXED_ACTION_SPACE_TYPE",
    "FIXED_DATA_EXPORT_MODE",
    "FIXED_LOCAL_RATIO",
    "FIXED_MAX_CYCLES",
    "FIXED_OBSERVATION_PARADIGM",
    "FIXED_RES_STR_FILE",
    "FIXED_REWARD_WEIGHTING_MODE",
    "LEARNING_PARAM_NAMES",
    "PAPERFOUR_RESULT_ROOT",
    "PaperFourParamSet",
    "STAGE_BUDGETS",
    "STAGE_LABELS",
    "build_learning_param_sets",
    "build_stage0_param_sets",
    "build_stage_a_param_sets",
    "build_stage_b_param_sets",
    "format_learning_params",
    "validate_learning_params",
]
