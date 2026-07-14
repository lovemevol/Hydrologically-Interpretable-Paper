#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch launcher for PaperFour HAPPO learning-dynamics experiments.

PaperFour fixes the hydrological system to:
    ResStr5.xlsx + HAPPO + fixed observation setting + default reward.

Only six HAPPO learning-dynamics parameters are allowed to vary:
learning_rate, critic_learning_rate, clip_ratio, entropy_weight,
discount_factor, and gae_lambda.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .paperfour_param_utils import (
        DEFAULT_LEARNING_PARAMS,
        FIXED_ACTION_DIM,
        FIXED_ACTION_SPACE_TYPE,
        FIXED_DATA_EXPORT_MODE,
        FIXED_LOCAL_RATIO,
        FIXED_MAX_CYCLES,
        FIXED_OBSERVATION_PARADIGM,
        FIXED_RES_STR_FILE,
        FIXED_REWARD_WEIGHTING_MODE,
        LEARNING_PARAM_NAMES,
        PAPERFOUR_RESULT_ROOT,
        STAGE_BUDGETS,
        STAGE_LABELS,
        build_learning_param_sets,
        format_learning_params,
        validate_learning_params,
    )
except ImportError:
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from paperfour_param_utils import (
        DEFAULT_LEARNING_PARAMS,
        FIXED_ACTION_DIM,
        FIXED_ACTION_SPACE_TYPE,
        FIXED_DATA_EXPORT_MODE,
        FIXED_LOCAL_RATIO,
        FIXED_MAX_CYCLES,
        FIXED_OBSERVATION_PARADIGM,
        FIXED_RES_STR_FILE,
        FIXED_REWARD_WEIGHTING_MODE,
        LEARNING_PARAM_NAMES,
        PAPERFOUR_RESULT_ROOT,
        STAGE_BUDGETS,
        STAGE_LABELS,
        build_learning_param_sets,
        format_learning_params,
        validate_learning_params,
    )


BEIJING_TZ = timezone(timedelta(hours=8))
BASE_CONFIG_FILE = "reservoir_happo.py"
DEFAULT_STAGE = "S0"

ExperimentConfig = Dict[str, Any]


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _replace_required(config_content: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, config_content, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Could not replace {label}; pattern not found: {pattern}")
    return updated


def _replace_param(config_content: str, param_name: str, param_value: Any) -> str:
    """Apply one PaperFour override to a HAPPO config source string."""
    scalar = _format_scalar(param_value)

    top_level_patterns = {
        "RES_STR_FILE": r"^RES_STR_FILE\s*=\s*['\"]ResStr\d+\.xlsx['\"]",
        "OBSERVATION_PARADIGM": r"^OBSERVATION_PARADIGM\s*=\s*['\"][a-z_]+['\"]",
        "ACTION_SPACE_TYPE": r"^ACTION_SPACE_TYPE\s*=\s*['\"][a-z_]+['\"]",
        "ACTION_DIM": r"^ACTION_DIM\s*=\s*\d+(?:\s+if\s+ACTION_SPACE_TYPE\s*==\s*['\"]discrete['\"]\s+else\s+\d+)?",
        "MAX_CYCLES": r"^MAX_CYCLES\s*=\s*\d+",
        "MAX_EPISODES": r"^MAX_EPISODES\s*=\s*\d+",
        "COLLECTOR_ENV_NUM": r"^COLLECTOR_ENV_NUM\s*=\s*\d+",
        "DATA_EXPORT_COUNT": r"^DATA_EXPORT_COUNT\s*=\s*\d+",
        "DATA_EXPORT_MODE": r"^DATA_EXPORT_MODE\s*=\s*['\"][a-z]+['\"]",
        "LOCAL_RATIO": r"^LOCAL_RATIO\s*=\s*[\d.eE+-]+",
        "LOCAL_REWARD_WEIGHTING_MODE": r"^LOCAL_REWARD_WEIGHTING_MODE\s*=\s*['\"][a-z_]+['\"]",
        "EVALUATOR_LOCAL_REWARD_WEIGHTING_MODE": (
            r"^EVALUATOR_LOCAL_REWARD_WEIGHTING_MODE\s*=\s*['\"][a-z_]+['\"]"
        ),
    }
    if param_name in top_level_patterns:
        return _replace_required(
            config_content,
            top_level_patterns[param_name],
            f"{param_name} = {scalar}",
            param_name,
        )

    learning_patterns = {
        "learning_rate": r"(?<!critic_)learning_rate\s*=\s*[\d.eE+-]+",
        "critic_learning_rate": r"critic_learning_rate\s*=\s*[\d.eE+-]+",
        "entropy_weight": r"entropy_weight\s*=\s*[\d.eE+-]+",
        "clip_ratio": r"clip_ratio\s*=\s*[\d.eE+-]+",
        "discount_factor": r"discount_factor\s*=\s*[\d.eE+-]+",
        "gae_lambda": r"gae_lambda\s*=\s*[\d.eE+-]+",
    }
    if param_name in learning_patterns:
        return _replace_required(
            config_content,
            learning_patterns[param_name],
            f"{param_name}={scalar}",
            param_name,
        )

    raise KeyError(f"Unsupported PaperFour config override: {param_name}")


def _build_fixed_params(stage: str) -> Dict[str, Any]:
    budget = STAGE_BUDGETS[stage]
    return {
        "RES_STR_FILE": FIXED_RES_STR_FILE,
        "OBSERVATION_PARADIGM": FIXED_OBSERVATION_PARADIGM,
        "ACTION_SPACE_TYPE": FIXED_ACTION_SPACE_TYPE,
        "ACTION_DIM": FIXED_ACTION_DIM,
        "MAX_CYCLES": FIXED_MAX_CYCLES,
        "MAX_EPISODES": budget["MAX_EPISODES"],
        "COLLECTOR_ENV_NUM": budget["COLLECTOR_ENV_NUM"],
        "DATA_EXPORT_COUNT": budget["DATA_EXPORT_COUNT"],
        "DATA_EXPORT_MODE": FIXED_DATA_EXPORT_MODE,
        "LOCAL_RATIO": FIXED_LOCAL_RATIO,
        "LOCAL_REWARD_WEIGHTING_MODE": FIXED_REWARD_WEIGHTING_MODE,
        "EVALUATOR_LOCAL_REWARD_WEIGHTING_MODE": FIXED_REWARD_WEIGHTING_MODE,
    }


def build_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Build PaperFour stage experiment definitions."""
    configs: Dict[str, ExperimentConfig] = {}
    for param_set in build_learning_param_sets():
        stage = param_set.stage
        validate_learning_params(param_set.ordered_params())
        exp_name = f"P4_{stage}_{param_set.config_id}"
        configs[exp_name] = {
            "paper_id": "PaperFour",
            "stage": stage,
            "config_id": param_set.config_id,
            "description": param_set.description,
            "base_config": BASE_CONFIG_FILE,
            "result_root": PAPERFOUR_RESULT_ROOT,
            "fixed_params": _build_fixed_params(stage),
            "changed_params": param_set.ordered_params(),
        }
    return configs


EXPERIMENT_CONFIGS = build_experiment_configs()


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    def run_git(args: List[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status and status != "unknown"),
    }


def _assert_generated_config(config_content: str, experiment: ExperimentConfig, result_dir: Path) -> None:
    expected_literals = {
        "RES_STR_FILE": FIXED_RES_STR_FILE,
        "OBSERVATION_PARADIGM": FIXED_OBSERVATION_PARADIGM,
        "ACTION_SPACE_TYPE": FIXED_ACTION_SPACE_TYPE,
        "DATA_EXPORT_MODE": FIXED_DATA_EXPORT_MODE,
        "LOCAL_REWARD_WEIGHTING_MODE": FIXED_REWARD_WEIGHTING_MODE,
        "EVALUATOR_LOCAL_REWARD_WEIGHTING_MODE": FIXED_REWARD_WEIGHTING_MODE,
    }
    for name, value in expected_literals.items():
        if f"{name} = {repr(value)}" not in config_content and f'{name} = "{value}"' not in config_content:
            raise ValueError(f"Generated config failed fixed literal check: {name}={value}")

    expected_numbers = {
        "ACTION_DIM": FIXED_ACTION_DIM,
        "MAX_CYCLES": FIXED_MAX_CYCLES,
        "MAX_EPISODES": experiment["fixed_params"]["MAX_EPISODES"],
        "COLLECTOR_ENV_NUM": experiment["fixed_params"]["COLLECTOR_ENV_NUM"],
        "DATA_EXPORT_COUNT": experiment["fixed_params"]["DATA_EXPORT_COUNT"],
        "LOCAL_RATIO": FIXED_LOCAL_RATIO,
    }
    for name, value in expected_numbers.items():
        if f"{name} = {_format_scalar(value)}" not in config_content:
            raise ValueError(f"Generated config failed fixed numeric check: {name}={value}")

    for name, value in experiment["changed_params"].items():
        pattern = rf"(?<!critic_){name}\s*=\s*{re.escape(_format_scalar(value))}"
        if name == "critic_learning_rate":
            pattern = rf"{name}\s*=\s*{re.escape(_format_scalar(value))}"
        if not re.search(pattern, config_content):
            raise ValueError(f"Generated config failed learning-param check: {name}={value}")

    if f"EXP_NAME = '{result_dir.as_posix()}'" not in config_content:
        raise ValueError("Generated config failed EXP_NAME replacement check.")


class ExperimentRunner:
    """Generate PaperFour config files and launch selected experiments."""

    def __init__(
        self,
        config_dir: Path,
        config_output_override: Optional[Path] = None,
        validate_only: bool = False,
    ) -> None:
        self.config_dir = config_dir.resolve()
        self.repo_root = self.config_dir.parents[2]
        self.config_output_override = config_output_override.resolve() if config_output_override else None
        self.validate_only = validate_only

    def _resolve_config_output_dir(self) -> Path:
        if self.config_output_override is not None:
            self.config_output_override.mkdir(parents=True, exist_ok=True)
            return self.config_output_override

        output_dir = self.repo_root / PAPERFOUR_RESULT_ROOT / "config"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_manifest(
        self,
        exp_name: str,
        experiment: ExperimentConfig,
        round_num: int,
        seed: int,
        run_name: str,
        result_dir: Path,
        config_path: Path,
    ) -> Dict[str, Any]:
        metadata = _git_metadata(self.repo_root)
        return {
            "paper_id": "PaperFour",
            "stage": experiment["stage"],
            "config_id": experiment["config_id"],
            "experiment_name": exp_name,
            "run_name": run_name,
            "round": round_num,
            "seed": seed,
            "fixed_system": "ResStr5.xlsx + HAPPO + fixed_observation_setting + default_reward",
            "base_config": BASE_CONFIG_FILE,
            "changed_params": deepcopy(experiment["changed_params"]),
            "fixed_params": deepcopy(experiment["fixed_params"]),
            "default_learning_params": deepcopy(DEFAULT_LEARNING_PARAMS),
            "input_files": {
                "res_inflow_file": "ResInflowEnhan.xlsx",
                "res_limit_file": "ResLimit.xlsx",
                "res_str_file": FIXED_RES_STR_FILE,
            },
            "result_root": PAPERFOUR_RESULT_ROOT,
            "output_dir": result_dir.as_posix(),
            "config_path": config_path.as_posix(),
            "git_commit": metadata["commit"],
            "git_dirty": metadata["dirty"],
            "created_at": datetime.now(BEIJING_TZ).isoformat(),
            "notes": "Total spillage is a neutral diagnostic for PaperFour, not a monotonic risk metric.",
        }

    def _write_manifest(self, result_dir: Path, manifest: Mapping[str, Any]) -> Path:
        result_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = result_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def generate_config_file(
        self,
        exp_name: str,
        experiment: ExperimentConfig,
        round_num: int,
        seed: int,
    ) -> tuple[Path, Path, str, Dict[str, Any]]:
        base_config_path = self.config_dir / BASE_CONFIG_FILE
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base config file not found: {base_config_path}")

        config_content = base_config_path.read_text(encoding="utf-8")
        params = dict(experiment["fixed_params"])
        params.update(experiment["changed_params"])
        for param_name, param_value in params.items():
            config_content = _replace_param(config_content, param_name, param_value)

        timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
        run_name = f"{exp_name}_R{round_num}_{timestamp}"
        result_dir = self.repo_root / PAPERFOUR_RESULT_ROOT / run_name
        result_dir_posix = (Path(PAPERFOUR_RESULT_ROOT) / run_name).as_posix()
        config_content = _replace_required(
            config_content,
            r"^EXP_NAME\s*=\s*f['\"][^'\"]*\{_timestamp\}['\"]",
            f"EXP_NAME = '{result_dir_posix}'",
            "EXP_NAME",
        )

        sys_path_insert = """import sys
import os

def _paperfour_find_agent_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.basename(current) == "reservoir_multi_agents":
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(f"Could not locate reservoir_multi_agents root from: {start_dir}")

_config_dir = os.path.dirname(os.path.abspath(__file__))
_agent_root_hint = _paperfour_find_agent_root(_config_dir)
_project_root = os.path.abspath(os.path.join(_agent_root_hint, "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _agent_root_hint not in sys.path:
    sys.path.insert(0, _agent_root_hint)

"""
        config_content = sys_path_insert + config_content
        config_content = _replace_required(
            config_content,
            r"_agent_root\s*=\s*os\.path\.dirname\(_current_dir\)",
            "_agent_root = _paperfour_find_agent_root(_current_dir)",
            "_agent_root",
        )
        _assert_generated_config(config_content, experiment, Path(result_dir_posix))

        output_dir = self._resolve_config_output_dir()
        config_path = output_dir / f"{run_name}_config.py"
        config_path.write_text(config_content, encoding="utf-8")
        manifest = self._build_manifest(
            exp_name=exp_name,
            experiment=experiment,
            round_num=round_num,
            seed=seed,
            run_name=run_name,
            result_dir=result_dir,
            config_path=config_path,
        )
        return config_path, result_dir, run_name, manifest

    def run_single_experiment(self, exp_name: str, experiment: ExperimentConfig, round_num: int, seed: int) -> bool:
        print(f"\n{'=' * 80}")
        print(f"Starting PaperFour experiment: {exp_name} - round {round_num}")
        print(f"Stage: {experiment['stage']} ({STAGE_LABELS.get(experiment['stage'], 'unknown')})")
        print(f"Config id: {experiment['config_id']}")
        print(f"Description: {experiment['description']}")
        print("Fixed system: ResStr5.xlsx + HAPPO + fixed_observation_setting + default_reward")
        print(f"Result root: {experiment['result_root']}")
        print(f"Seed: {seed}")
        print(f"Learning params: {format_learning_params(experiment['changed_params'])}")
        print(f"{'=' * 80}\n")

        try:
            config_path, result_dir, run_name, manifest = self.generate_config_file(
                exp_name,
                experiment,
                round_num,
                seed,
            )
            print(f"Config generated: {config_path}")

            cmd = [sys.executable, "-u", str(config_path), "--seed", str(seed)]
            if self.validate_only:
                cmd.append("--validate")
            print(f"Run name: {run_name}")
            print(f"Command: {' '.join(cmd)}\n")

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env.setdefault("PYTHONUTF8", "1")
            result = subprocess.run(cmd, cwd=self.repo_root, capture_output=False, text=True, env=env)
            if result.returncode == 0:
                if not self.validate_only:
                    manifest_path = self._write_manifest(result_dir, manifest)
                    print(f"Manifest generated: {manifest_path}")
                print(f"\nExperiment {exp_name} round {round_num} completed.\n")
                return True

            print(f"\nExperiment {exp_name} round {round_num} failed with code {result.returncode}.\n")
            return False
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"\nExperiment {exp_name} round {round_num} crashed: {exc}\n")
            import traceback

            traceback.print_exc()
            return False

    def run_all_experiments(self, experiments: List[str], rounds: int, seed_base: int) -> Dict[str, List[bool]]:
        results: Dict[str, List[bool]] = {}
        valid_experiments: List[str] = []
        for exp_name in experiments:
            if exp_name not in EXPERIMENT_CONFIGS:
                print(f"Warning: unknown experiment config {exp_name}, skipped.")
                continue
            valid_experiments.append(exp_name)
            results[exp_name] = []

        total_tasks = len(valid_experiments) * rounds
        current_task = 0
        print(f"\n{'=' * 80}")
        print("PaperFour batch execution plan")
        print(f"Experiment configs: {len(valid_experiments)}")
        print(f"Rounds: {rounds}")
        print(f"Total tasks: {total_tasks}")
        print(f"Seed rule: seed = round * {seed_base}")
        print(f"Execution mode: {'validate_only' if self.validate_only else 'train'}")
        print(f"{'=' * 80}\n")

        for round_num in range(1, rounds + 1):
            for exp_name in valid_experiments:
                current_task += 1
                seed = round_num * seed_base
                print(f"\nProgress: {current_task}/{total_tasks}")
                success = self.run_single_experiment(exp_name, EXPERIMENT_CONFIGS[exp_name], round_num, seed)
                results[exp_name].append(success)
        return results

    @staticmethod
    def print_summary(results: Mapping[str, List[bool]]) -> None:
        print(f"\n\n{'=' * 80}")
        print("PaperFour experiment summary")
        print(f"{'=' * 80}\n")

        total_success = 0
        total_fail = 0
        for exp_name, round_results in results.items():
            success_count = sum(round_results)
            fail_count = len(round_results) - success_count
            total_success += success_count
            total_fail += fail_count
            if all(round_results):
                status = "OK"
            elif any(round_results):
                status = "PARTIAL"
            else:
                status = "FAIL"
            print(f"{status:7} {exp_name}: {success_count}/{len(round_results)} rounds succeeded")

        total_runs = total_success + total_fail
        success_rate = (total_success / total_runs * 100.0) if total_runs else 0.0
        print(f"\n{'=' * 80}")
        print(f"Total success: {total_success}")
        print(f"Total fail: {total_fail}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"{'=' * 80}\n")


def _select_experiments(args: argparse.Namespace) -> List[str]:
    if args.experiments:
        return args.experiments

    stage = args.stage.upper()
    return [name for name, cfg in EXPERIMENT_CONFIGS.items() if cfg["stage"] == stage]


def _print_available_experiments(stage: Optional[str] = None) -> None:
    selected_stage = stage.upper() if stage else None
    print("\nAvailable PaperFour experiment configs")
    print("=" * 100)
    for stage_key, label in STAGE_LABELS.items():
        if selected_stage and selected_stage != stage_key:
            continue
        print(f"{stage_key}: {label}")
        for exp_name, cfg in EXPERIMENT_CONFIGS.items():
            if cfg["stage"] != stage_key:
                continue
            print(f"  {exp_name}: {cfg['description']}")
            print(f"    params: {format_learning_params(cfg['changed_params'])}")
            budget = cfg["fixed_params"]
            print(
                "    budget: "
                f"MAX_EPISODES={budget['MAX_EPISODES']}, "
                f"COLLECTOR_ENV_NUM={budget['COLLECTOR_ENV_NUM']}, "
                f"DATA_EXPORT_COUNT={budget['DATA_EXPORT_COUNT']}"
            )
        print()
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaperFour HAPPO learning-dynamics experiment batch launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent/reservoir_multi_agents/config/PaperFour_run_experiments.py --list\n"
            "  python agent/reservoir_multi_agents/config/PaperFour_run_experiments.py --stage S0 --validate_only\n"
            "  python agent/reservoir_multi_agents/config/PaperFour_run_experiments.py --stage S0 --rounds 1\n"
            "  python agent/reservoir_multi_agents/config/PaperFour_run_experiments.py "
            "--experiments P4_B_default P4_B_high_clip --rounds 3\n"
        ),
    )
    parser.add_argument("--experiments", nargs="+", default=None, help="Explicit PaperFour experiment names to run")
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_LABELS.keys()),
        default=DEFAULT_STAGE,
        help=f"PaperFour stage to run when --experiments is omitted (default: {DEFAULT_STAGE})",
    )
    parser.add_argument("--rounds", type=int, default=1, help="Training rounds for each experiment")
    parser.add_argument("--seed_base", type=int, default=100, help="Seed base. Actual seed = round * seed_base")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for generated config files. Relative paths are resolved from the repo root.",
    )
    parser.add_argument("--list", action="store_true", help="List available PaperFour experiment configs")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Generate configs and run each config with --validate instead of starting training",
    )
    args = parser.parse_args()

    if args.list:
        _print_available_experiments(args.stage if args.stage else None)
        return

    experiments = _select_experiments(args)
    invalid = [name for name in experiments if name not in EXPERIMENT_CONFIGS]
    if invalid:
        print(f"Error: unknown PaperFour experiment config(s): {', '.join(invalid)}")
        print("Use --list to inspect available configs.")
        return

    config_dir = Path(__file__).resolve().parent
    base_config = config_dir / BASE_CONFIG_FILE
    if not base_config.exists():
        print(f"Error: missing base config file: {base_config}")
        return

    if not args.experiments:
        print(f"\nNo --experiments provided. Running PaperFour stage: {args.stage}\n")

    config_output_override = None
    if args.output_dir:
        candidate = Path(args.output_dir)
        config_output_override = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
        config_output_override.mkdir(parents=True, exist_ok=True)
        print(f"Generated configs will be saved to: {config_output_override.resolve()}\n")
    else:
        print(f"Generated configs will be saved to: {(PROJECT_ROOT / PAPERFOUR_RESULT_ROOT / 'config').resolve()}\n")

    runner = ExperimentRunner(
        config_dir,
        config_output_override=config_output_override,
        validate_only=args.validate_only,
    )
    results = runner.run_all_experiments(experiments, args.rounds, args.seed_base)
    runner.print_summary(results)
    if any(not all(round_results) for round_results in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
