#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline evaluator for PaperFour Stage C hydrological scenarios.

Stage C does not retrain policies.  It restores representative Stage B
checkpoints from ``result_paperFour`` and evaluates them on one or more held-out
inflow workbooks.  Candidate configurations are read from the formal Stage B
response dataset produced by ``build_response_dataset.py``.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import runpy
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


warnings.filterwarnings("ignore", category=FutureWarning, module="treevalue")
warnings.filterwarnings("ignore", category=UserWarning, module="gym.envs.registration")


CURRENT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = AGENT_ROOT.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


DEFAULT_RESULT_DIR = AGENT_ROOT / "result_paperFour"
DEFAULT_OUTPUT_ROOT = AGENT_ROOT / "result_paperFour_eval"
DEFAULT_STAGE_B_RESPONSE = (
    AGENT_ROOT
    / "analysis"
    / "experiment_analysis_paperFour"
    / "output_stage_b"
    / "stage_b_response_dataset.csv"
)
DEFAULT_STAGE_C_CANDIDATES = (
    AGENT_ROOT
    / "analysis"
    / "experiment_analysis_paperFour"
    / "output_stage_b"
    / "stage_b_stage_c_candidates.csv"
)
DEFAULT_EVAL_INFLOW = AGENT_ROOT / "input" / "ResInflowEva.xlsx"
DEFAULT_CKPT_NAME = "ckpt_best.pth.tar"
DEFAULT_EVAL_SEED = 0
DEFAULT_SCENARIO_ID = "mixed_percentile_5yr"
FIXED_SYSTEM = "ResStr5.xlsx + HAPPO + fixed_observation_setting + default_reward"


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    scenario_type: str
    inflow_path: Path
    description: str
    source_years: str = ""


@dataclass(frozen=True)
class EvalSpec:
    config_id: str
    stage_c_role: str
    source_run_name: str
    source_run_dir: Path
    total_config_path: Path
    checkpoint_path: Path
    source_seed: int | None
    source_round: int | None
    source_return: float | None
    selection_policy: str
    changed_params: Mapping[str, Any]


def _normalize_path(path_str: str | Path) -> Path:
    return Path(path_str).expanduser().resolve()


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_scenario_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        if isinstance(payload, dict):
            rows = payload.get("scenarios", [])
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError(f"JSON scenario file must contain a list or a scenarios list: {path}")
        return [dict(row) for row in rows]
    return [dict(row) for row in _read_csv_rows(path)]


def _safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned.strip("_") or "scenario"


def _resolve_input_file(value: str | Path, base_dir: Path | None = None) -> Path:
    raw = Path(value).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if base_dir is not None:
            candidates.append(base_dir / raw)
        candidates.extend([PROJECT_ROOT / raw, AGENT_ROOT / "input" / raw, AGENT_ROOT / raw])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else raw.resolve()


def _default_scenario(eval_inflow: Path, scenario_id: str, scenario_type: str) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=_safe_id(scenario_id),
        scenario_type=scenario_type,
        inflow_path=eval_inflow,
        description=(
            "Five-year held-out evaluation inflow sequence composed of annual "
            "inflow percentiles 5%, 25%, 50%, 75%, and 95%."
        ),
        source_years="5%,25%,50%,75%,95% annual inflow percentiles",
    )


def load_scenarios(args: argparse.Namespace) -> List[ScenarioSpec]:
    if args.scenario_file:
        scenario_file = _normalize_path(args.scenario_file)
        if not scenario_file.is_file():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")
        rows = _read_scenario_rows(scenario_file)
        scenarios: List[ScenarioSpec] = []
        for row in rows:
            raw_scenario_id = row.get("scenario_id") or row.get("id") or ""
            if not raw_scenario_id:
                raise ValueError(f"Scenario row is missing scenario_id: {row}")
            scenario_id = _safe_id(str(raw_scenario_id))
            inflow_value = row.get("eval_inflow") or row.get("inflow_file") or row.get("inflow_path")
            if not inflow_value:
                raise ValueError(f"Scenario row is missing eval_inflow: {row}")
            scenarios.append(
                ScenarioSpec(
                    scenario_id=scenario_id,
                    scenario_type=row.get("scenario_type") or row.get("type") or "custom",
                    inflow_path=_resolve_input_file(inflow_value, scenario_file.parent),
                    description=row.get("description") or "",
                    source_years=row.get("source_years") or "",
                )
            )
    else:
        eval_inflow = _normalize_path(args.eval_inflow) if args.eval_inflow else DEFAULT_EVAL_INFLOW.resolve()
        scenarios = [_default_scenario(eval_inflow, args.scenario_id, args.scenario_type)]

    selected = set(args.scenarios or [])
    if selected:
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in selected]
    if not scenarios:
        raise ValueError("No Stage C scenarios selected.")
    return scenarios


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_run_dirs(result_dir: Path) -> Iterable[Path]:
    for child in sorted(result_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name in {"config", "launcher_logs", "__pycache__"}:
            continue
        yield child


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        return _load_json(manifest_path)
    return {}


def _load_candidate_rows(
    candidates_file: Path,
    config_ids: Sequence[str] | None,
    roles: Sequence[str] | None,
) -> List[Dict[str, str]]:
    if not candidates_file.is_file():
        raise FileNotFoundError(
            f"Stage C candidates file not found: {candidates_file}. "
            "Run build_response_dataset.py for Stage B first."
        )

    rows = _read_csv_rows(candidates_file)
    rows = [row for row in rows if _to_bool(row.get("stage_c_recommended", "true"))]

    selected_configs = set(config_ids or [])
    if selected_configs:
        rows = [row for row in rows if row.get("config_id") in selected_configs]

    selected_roles = set(roles or [])
    if selected_roles:
        rows = [row for row in rows if row.get("stage_c_role") in selected_roles]

    if not rows:
        raise ValueError("No Stage C candidate rows matched the requested filters.")
    return rows


def _select_response_rows(rows: List[Dict[str, str]], seed_policy: str) -> List[Dict[str, str]]:
    if seed_policy == "all_seeds":
        return sorted(rows, key=lambda row: (_to_int(row.get("seed"), 0) or 0, row.get("run_name") or ""))

    scored = [(row, _to_float(row.get("return"), float("nan"))) for row in rows]
    scored = [(row, value) for row, value in scored if value == value]
    if not scored:
        return [sorted(rows, key=lambda row: row.get("run_name") or "")[0]]

    if seed_policy == "best_return":
        selected = max(scored, key=lambda item: (item[1], -(_to_int(item[0].get("seed"), 10**9) or 10**9)))
        return [selected[0]]
    if seed_policy == "worst_return":
        selected = min(scored, key=lambda item: (item[1], _to_int(item[0].get("seed"), 10**9) or 10**9))
        return [selected[0]]
    if seed_policy == "median_return":
        values = sorted(value for _, value in scored)
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
        selected = min(
            scored,
            key=lambda item: (abs(item[1] - median), _to_int(item[0].get("seed"), 10**9) or 10**9),
        )
        return [selected[0]]
    raise ValueError(f"Unsupported seed policy: {seed_policy}")


def _build_spec_from_run_dir(run_dir: Path, ckpt_name: str, selection_policy: str) -> EvalSpec:
    manifest = _load_manifest(run_dir)
    checkpoint_path = run_dir / "ckpt" / ckpt_name
    total_config_path = run_dir / "total_config.py"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not total_config_path.is_file():
        raise FileNotFoundError(f"total_config.py not found: {total_config_path}")

    return EvalSpec(
        config_id=str(manifest.get("config_id") or run_dir.name),
        stage_c_role=str(manifest.get("stage_c_role") or "manual_run"),
        source_run_name=run_dir.name,
        source_run_dir=run_dir.resolve(),
        total_config_path=total_config_path.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        source_seed=_to_int(manifest.get("seed")),
        source_round=_to_int(manifest.get("round")),
        source_return=None,
        selection_policy=selection_policy,
        changed_params=manifest.get("changed_params") or {},
    )


def discover_eval_specs(
    result_dir: Path,
    response_file: Path,
    candidates_file: Path,
    ckpt_name: str,
    seed_policy: str,
    config_ids: Sequence[str] | None,
    roles: Sequence[str] | None,
    run_names: Sequence[str] | None,
    pattern: str,
) -> List[EvalSpec]:
    if run_names:
        selected = set(run_names)
        specs = []
        for run_dir in _iter_run_dirs(result_dir):
            if run_dir.name not in selected:
                continue
            specs.append(_build_spec_from_run_dir(run_dir, ckpt_name, "manual_run"))
        missing = sorted(selected.difference({spec.source_run_name for spec in specs}))
        if missing:
            raise FileNotFoundError(f"Selected run(s) not found: {missing}")
        return specs

    candidate_rows = _load_candidate_rows(candidates_file, config_ids, roles)
    candidate_roles = {row["config_id"]: row.get("stage_c_role") or "" for row in candidate_rows}
    candidate_ids = set(candidate_roles)

    if not response_file.is_file():
        raise FileNotFoundError(
            f"Stage B response dataset not found: {response_file}. "
            "Run build_response_dataset.py for Stage B first."
        )
    response_rows = _read_csv_rows(response_file)
    response_by_config: Dict[str, List[Dict[str, str]]] = {config_id: [] for config_id in candidate_ids}
    for row in response_rows:
        if row.get("stage") != "B":
            continue
        config_id = row.get("config_id")
        if config_id not in candidate_ids:
            continue
        run_name = row.get("run_name") or ""
        if pattern and not fnmatch.fnmatch(run_name, pattern):
            continue
        response_by_config[config_id].append(row)

    specs: List[EvalSpec] = []
    missing_configs = []
    for config_id in sorted(candidate_ids):
        rows = response_by_config.get(config_id) or []
        if not rows:
            missing_configs.append(config_id)
            continue
        for row in _select_response_rows(rows, seed_policy):
            run_dir_value = row.get("run_dir") or ""
            run_name = row.get("run_name") or ""
            run_dir = Path(run_dir_value) if run_dir_value else result_dir / run_name
            if not run_dir.is_absolute():
                run_dir = (PROJECT_ROOT / run_dir).resolve()
            checkpoint_path = run_dir / "ckpt" / ckpt_name
            total_config_path = run_dir / "total_config.py"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if not total_config_path.is_file():
                raise FileNotFoundError(f"total_config.py not found: {total_config_path}")
            manifest = _load_manifest(run_dir)
            specs.append(
                EvalSpec(
                    config_id=config_id,
                    stage_c_role=candidate_roles[config_id],
                    source_run_name=run_name or run_dir.name,
                    source_run_dir=run_dir.resolve(),
                    total_config_path=total_config_path.resolve(),
                    checkpoint_path=checkpoint_path.resolve(),
                    source_seed=_to_int(row.get("seed")),
                    source_round=_to_int(row.get("round")),
                    source_return=_to_float(row.get("return")),
                    selection_policy=seed_policy,
                    changed_params=manifest.get("changed_params") or {},
                )
            )
    if missing_configs:
        raise ValueError(f"No Stage B response rows found for config(s): {missing_configs}")
    return specs


def _load_frozen_config(total_config_path: Path) -> Dict[str, Any]:
    namespace = runpy.run_path(str(total_config_path), init_globals={"inf": float("inf")})
    exp_config = namespace.get("exp_config")
    if not isinstance(exp_config, dict):
        raise ValueError(f"`exp_config` not found in {total_config_path}")
    return deepcopy(exp_config)


def _to_easydict(obj: Any) -> Any:
    from easydict import EasyDict

    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_easydict(v) for v in obj)
    return obj


def _patch_cfg_for_eval(
    cfg: Any,
    run_output_dir: Path,
    eval_inflow_path: Path,
    show_progress: bool,
    force_cpu: bool,
) -> Any:
    cfg = deepcopy(cfg)
    cfg.exp_name = str(run_output_dir)

    excel_files = deepcopy(dict(cfg.env.excel_files))
    excel_files["res_inflow_file"] = str(eval_inflow_path)
    cfg.env.excel_files = _to_easydict(excel_files)

    cfg.env.show_progress = bool(show_progress)
    cfg.env.n_evaluator_episode = int(cfg.env.get("n_evaluator_episode", 1))
    cfg.env.evaluator_env_num = int(cfg.env.get("evaluator_env_num", 1))
    cfg.policy.eval.evaluator.n_episode = int(cfg.policy.eval.evaluator.get("n_episode", 1))
    cfg.policy.eval.env_num = int(cfg.policy.eval.get("env_num", cfg.env.evaluator_env_num))
    cfg.policy.eval.evaluator.stop_value = float("inf")

    if force_cpu:
        cfg.policy.cuda = False
    return cfg


def _create_env_with_config(
    env_fn,
    cfg: Any,
    idx: int,
    env_type: str,
    export_dir: str,
    tb_log_dir: str | None,
):
    cfg_with_idx = deepcopy(cfg)
    cfg_with_idx["env_idx"] = idx
    cfg_with_idx["env_type"] = env_type
    cfg_with_idx["should_export_schedule"] = idx == 0
    cfg_with_idx["export_dir"] = export_dir
    cfg_with_idx["export_interval"] = 1
    cfg_with_idx["max_episodes"] = 1
    if idx == 0 and tb_log_dir:
        cfg_with_idx["tb_log_dir"] = tb_log_dir
        cfg_with_idx["tb_log_interval"] = 1
    return env_fn(cfg_with_idx)


def _import_runtime_dependencies():
    try:
        import numpy as np
        import torch
        from ding.envs import create_env_manager, get_vec_env_setting
        from ding.policy import create_policy
        from ding.utils import set_pkg_seed
        from ding.worker import InteractionSerialEvaluator
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "Missing evaluation runtime dependency. Activate the project virtual "
            f"environment or install the missing dependency in it: {missing_name}"
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "create_env_manager": create_env_manager,
        "get_vec_env_setting": get_vec_env_setting,
        "create_policy": create_policy,
        "set_pkg_seed": set_pkg_seed,
        "InteractionSerialEvaluator": InteractionSerialEvaluator,
    }


def _eval_run_name(spec: EvalSpec, scenario: ScenarioSpec) -> str:
    seed_tag = f"seed{spec.source_seed}" if spec.source_seed is not None else "seedNA"
    return f"P4_C_{scenario.scenario_id}_{spec.config_id}_{seed_tag}"


def _metadata_payload(
    spec: EvalSpec,
    scenario: ScenarioSpec,
    run_output_dir: Path,
    eval_seed: int,
    force_cpu: bool,
    status: str,
    eval_returns: Sequence[float] | None = None,
    error: str = "",
) -> Dict[str, Any]:
    eval_returns = list(eval_returns or [])
    return {
        "paper_id": "PaperFour",
        "stage": "C",
        "fixed_system": FIXED_SYSTEM,
        "run_name": run_output_dir.name,
        "config_id": spec.config_id,
        "stage_c_role": spec.stage_c_role,
        "selection_policy": spec.selection_policy,
        "source_run_name": spec.source_run_name,
        "source_run_dir": str(spec.source_run_dir),
        "source_seed": spec.source_seed,
        "source_round": spec.source_round,
        "source_return": spec.source_return,
        "checkpoint_path": str(spec.checkpoint_path),
        "total_config_path": str(spec.total_config_path),
        "changed_params": dict(spec.changed_params),
        "scenario_id": scenario.scenario_id,
        "scenario_type": scenario.scenario_type,
        "scenario_description": scenario.description,
        "source_years": scenario.source_years,
        "eval_inflow_file": str(scenario.inflow_path),
        "eval_output_dir": str(run_output_dir),
        "eval_seed": eval_seed,
        "force_cpu": bool(force_cpu),
        "status": status,
        "eval_episode_return": eval_returns,
        "eval_episode_return_mean": None,
        "eval_episode_return_std": None,
        "error": error,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Stage C is offline evaluation only; no environment, reward, or policy retraining changes.",
    }


def evaluate_spec_on_scenario(
    spec: EvalSpec,
    scenario: ScenarioSpec,
    output_root: Path,
    eval_seed: int,
    show_progress: bool,
    force_cpu: bool,
    skip_existing: bool,
) -> Dict[str, Any]:
    run_output_dir = output_root / scenario.scenario_id / _eval_run_name(spec, scenario)
    metadata_path = run_output_dir / "eval_metadata.json"
    if skip_existing and metadata_path.is_file():
        metadata = _load_json(metadata_path)
        if metadata.get("status") == "ok":
            return _summary_row(metadata)

    runtime = _import_runtime_dependencies()
    np = runtime["np"]
    torch = runtime["torch"]
    create_env_manager = runtime["create_env_manager"]
    get_vec_env_setting = runtime["get_vec_env_setting"]
    create_policy = runtime["create_policy"]
    set_pkg_seed = runtime["set_pkg_seed"]
    InteractionSerialEvaluator = runtime["InteractionSerialEvaluator"]

    frozen_cfg = _load_frozen_config(spec.total_config_path)
    cfg = _to_easydict(frozen_cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _patch_cfg_for_eval(cfg, run_output_dir, scenario.inflow_path, show_progress, force_cpu)

    if bool(cfg.policy.get("cuda", False)) and not torch.cuda.is_available():
        cfg.policy.cuda = False

    export_dir = run_output_dir / "schedule_exports"
    tb_log_dir = run_output_dir / "log" / "serial"

    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    evaluator_env = create_env_manager(
        cfg.env.manager,
        [
            partial(
                _create_env_with_config,
                env_fn,
                env_cfg,
                idx,
                "evaluator",
                str(export_dir),
                str(tb_log_dir),
            )
            for idx, env_cfg in enumerate(evaluator_env_cfg)
        ],
    )

    try:
        evaluator_env.seed(eval_seed, dynamic_seed=False)
        set_pkg_seed(eval_seed, use_cuda=bool(cfg.policy.get("cuda", False)))
        policy = create_policy(cfg.policy, enable_field=["eval"])
        state_dict = torch.load(spec.checkpoint_path, map_location="cpu")
        policy.eval_mode.load_state_dict(state_dict)
        evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode)
        _, episode_info = evaluator.eval()

        eval_returns = [float(value) for value in episode_info.get("eval_episode_return", [])]
        eval_mean = float(np.mean(eval_returns)) if eval_returns else float("nan")
        eval_std = float(np.std(eval_returns)) if eval_returns else float("nan")
        metadata = _metadata_payload(
            spec=spec,
            scenario=scenario,
            run_output_dir=run_output_dir,
            eval_seed=eval_seed,
            force_cpu=force_cpu,
            status="ok",
            eval_returns=eval_returns,
        )
        metadata["cuda"] = bool(cfg.policy.get("cuda", False))
        metadata["eval_episode_return_mean"] = eval_mean
        metadata["eval_episode_return_std"] = eval_std
        _write_metadata(run_output_dir, metadata)
        return _summary_row(metadata)
    finally:
        try:
            evaluator_env.close()
        except Exception:
            pass


def _write_metadata(run_output_dir: Path, metadata: Mapping[str, Any]) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("eval_metadata.json", "manifest.json"):
        with (run_output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def _summary_row(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_name": metadata.get("run_name", ""),
        "status": metadata.get("status", ""),
        "scenario_id": metadata.get("scenario_id", ""),
        "scenario_type": metadata.get("scenario_type", ""),
        "config_id": metadata.get("config_id", ""),
        "stage_c_role": metadata.get("stage_c_role", ""),
        "source_run_name": metadata.get("source_run_name", ""),
        "source_seed": metadata.get("source_seed", ""),
        "source_return": metadata.get("source_return", ""),
        "selection_policy": metadata.get("selection_policy", ""),
        "eval_return_mean": metadata.get("eval_episode_return_mean", ""),
        "eval_return_std": metadata.get("eval_episode_return_std", ""),
        "eval_seed": metadata.get("eval_seed", ""),
        "eval_inflow_file": metadata.get("eval_inflow_file", ""),
        "checkpoint_path": metadata.get("checkpoint_path", ""),
        "total_config_path": metadata.get("total_config_path", ""),
        "eval_output_dir": metadata.get("eval_output_dir", ""),
        "error": metadata.get("error", ""),
    }


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "status",
        "scenario_id",
        "scenario_type",
        "config_id",
        "stage_c_role",
        "source_run_name",
        "source_seed",
        "source_return",
        "selection_policy",
        "eval_return_mean",
        "eval_return_std",
        "eval_seed",
        "eval_inflow_file",
        "checkpoint_path",
        "total_config_path",
        "eval_output_dir",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_batch_metadata(
    output_root: Path,
    args: argparse.Namespace,
    scenarios: Sequence[ScenarioSpec],
    specs: Sequence[EvalSpec],
) -> None:
    payload = {
        "paper_id": "PaperFour",
        "stage": "C",
        "fixed_system": FIXED_SYSTEM,
        "output_root": str(output_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_dir": str(_normalize_path(args.result_dir)),
        "stage_b_response": str(_normalize_path(args.stage_b_response)),
        "candidates_file": str(_normalize_path(args.candidates_file)),
        "seed_policy": args.seed_policy,
        "eval_seed": args.seed,
        "force_cpu": args.cpu,
        "show_progress": args.show_progress,
        "ckpt_name": args.ckpt_name,
        "scenario_count": len(scenarios),
        "scenarios": [scenario.__dict__ | {"inflow_path": str(scenario.inflow_path)} for scenario in scenarios],
        "candidate_count": len(specs),
        "candidates": [
            {
                "config_id": spec.config_id,
                "stage_c_role": spec.stage_c_role,
                "source_run_name": spec.source_run_name,
                "source_seed": spec.source_seed,
                "source_return": spec.source_return,
                "checkpoint_path": str(spec.checkpoint_path),
            }
            for spec in specs
        ],
    }
    with (output_root / "batch_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _print_plan(scenarios: Sequence[ScenarioSpec], specs: Sequence[EvalSpec], output_root: Path) -> None:
    print("\nPaperFour Stage C offline evaluation plan")
    print("=" * 100)
    print(f"Scenarios: {len(scenarios)}")
    for scenario in scenarios:
        status = "OK" if scenario.inflow_path.is_file() else "MISSING"
        print(f"  {status:7} {scenario.scenario_id}: {scenario.inflow_path}")
    print(f"Candidate checkpoints: {len(specs)}")
    for spec in specs:
        print(
            f"  {spec.config_id:22} seed={spec.source_seed!s:>4} "
            f"return={spec.source_return if spec.source_return is not None else 'NA'} "
            f"role={spec.stage_c_role} run={spec.source_run_name}"
        )
    print(f"Planned evaluations: {len(scenarios) * len(specs)}")
    print(f"Output root: {output_root}")
    print("=" * 100)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PaperFour Stage B representative checkpoints on Stage C hydrological scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--stage-b-response", type=str, default=str(DEFAULT_STAGE_B_RESPONSE))
    parser.add_argument("--candidates-file", type=str, default=str(DEFAULT_STAGE_C_CANDIDATES))
    parser.add_argument("--config-ids", nargs="*", help="Optional Stage C candidate config_id filter.")
    parser.add_argument("--roles", nargs="*", help="Optional Stage C role filter.")
    parser.add_argument("--run-names", nargs="*", help="Manual exact Stage B run directories to evaluate.")
    parser.add_argument("--pattern", type=str, default="*", help="Optional fnmatch filter on Stage B run names.")
    parser.add_argument(
        "--seed-policy",
        choices=["median_return", "best_return", "worst_return", "all_seeds"],
        default="median_return",
        help="How to choose one checkpoint from the three Stage B seeds for each config.",
    )
    parser.add_argument("--scenario-file", type=str, default=None, help="CSV scenario table.")
    parser.add_argument("--scenarios", nargs="*", help="Scenario IDs to run when --scenario-file is used.")
    parser.add_argument("--eval-inflow", type=str, default=str(DEFAULT_EVAL_INFLOW))
    parser.add_argument("--scenario-id", type=str, default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--scenario-type", type=str, default="mixed_five_year_percentile_sequence")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED, help="Evaluator random seed.")
    parser.add_argument("--ckpt-name", type=str, default=DEFAULT_CKPT_NAME)
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on candidate checkpoints.")
    parser.add_argument("--cpu", action="store_true", help="Force policy evaluation on CPU.")
    parser.add_argument("--show-progress", action="store_true", help="Enable environment-side progress bars.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip run directories with ok eval_metadata.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without evaluating.")
    parser.add_argument("--validate_only", action="store_true", help="Alias for --dry-run with strict file checks.")
    parser.add_argument("--list", action="store_true", help="List selected scenarios and checkpoints, then exit.")
    parser.add_argument("--strict", action="store_true", help="Stop immediately when one evaluation fails.")
    return parser


def main(args: argparse.Namespace) -> int:
    result_dir = _normalize_path(args.result_dir)
    output_root = _normalize_path(args.output_root)
    response_file = _normalize_path(args.stage_b_response)
    candidates_file = _normalize_path(args.candidates_file)

    if not result_dir.is_dir():
        print(f"Result directory not found: {result_dir}")
        return 1

    try:
        scenarios = load_scenarios(args)
        specs = discover_eval_specs(
            result_dir=result_dir,
            response_file=response_file,
            candidates_file=candidates_file,
            ckpt_name=args.ckpt_name,
            seed_policy=args.seed_policy,
            config_ids=args.config_ids,
            roles=args.roles,
            run_names=args.run_names,
            pattern=args.pattern,
        )
    except Exception as exc:
        print(f"Failed to build Stage C evaluation plan: {type(exc).__name__}: {exc}")
        return 1

    if args.max_runs is not None:
        specs = specs[: max(args.max_runs, 0)]

    _print_plan(scenarios, specs, output_root)

    missing_scenarios = [scenario for scenario in scenarios if not scenario.inflow_path.is_file()]
    if missing_scenarios:
        print(f"Missing scenario inflow file(s): {[str(s.inflow_path) for s in missing_scenarios]}")
        return 1
    if not specs:
        print("No candidate checkpoints selected.")
        return 1

    if args.list or args.dry_run or args.validate_only:
        print("Plan validation completed. No evaluator was launched.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    _write_batch_metadata(output_root, args, scenarios, specs)
    all_rows: List[Dict[str, Any]] = []
    summary_csv = output_root / "stage_c_eval_summary.csv"

    total_tasks = len(scenarios) * len(specs)
    current_task = 0
    for scenario in scenarios:
        scenario_rows: List[Dict[str, Any]] = []
        scenario_summary_csv = output_root / scenario.scenario_id / "summary.csv"
        for spec in specs:
            current_task += 1
            print(
                f"[{current_task}/{total_tasks}] Evaluating scenario={scenario.scenario_id} "
                f"config={spec.config_id} source_seed={spec.source_seed}"
            )
            try:
                row = evaluate_spec_on_scenario(
                    spec=spec,
                    scenario=scenario,
                    output_root=output_root,
                    eval_seed=args.seed,
                    show_progress=args.show_progress,
                    force_cpu=args.cpu,
                    skip_existing=args.skip_existing,
                )
                print(f"  OK: return_mean={row.get('eval_return_mean')}")
            except Exception as exc:
                run_output_dir = output_root / scenario.scenario_id / _eval_run_name(spec, scenario)
                metadata = _metadata_payload(
                    spec=spec,
                    scenario=scenario,
                    run_output_dir=run_output_dir,
                    eval_seed=args.seed,
                    force_cpu=args.cpu,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                _write_metadata(run_output_dir, metadata)
                row = _summary_row(metadata)
                print(f"  FAILED: {row['error']}")
                traceback.print_exc()
                if args.strict:
                    scenario_rows.append(row)
                    all_rows.append(row)
                    _write_summary_csv(scenario_rows, scenario_summary_csv)
                    _write_summary_csv(all_rows, summary_csv)
                    return 1

            scenario_rows.append(row)
            all_rows.append(row)
            _write_summary_csv(scenario_rows, scenario_summary_csv)
            _write_summary_csv(all_rows, summary_csv)

    ok_count = sum(1 for row in all_rows if row.get("status") == "ok")
    fail_count = len(all_rows) - ok_count
    print(f"Finished Stage C evaluation batch: ok={ok_count}, failed={fail_count}")
    print(f"Output directory: {output_root}")
    print(f"Summary CSV: {summary_csv}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    parser = build_arg_parser()
    raise SystemExit(main(parser.parse_args()))
