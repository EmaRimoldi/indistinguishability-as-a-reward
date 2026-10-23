from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent batched Optuna orchestration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--study-dir", required=True)
    prepare.add_argument("--search-config", required=True)
    prepare.add_argument("--template-config", required=True)

    propose = subparsers.add_parser("propose-batch")
    propose.add_argument("--study-dir", required=True)
    propose.add_argument("--count", type=int, required=True)
    propose.add_argument("--batch-id")

    enqueue = subparsers.add_parser("enqueue-refinement")
    enqueue.add_argument("--study-dir", required=True)
    enqueue.add_argument("--config", required=True)

    run_trial = subparsers.add_parser("run-trial")
    run_trial.add_argument("--spec", required=True)

    resume_evaluation = subparsers.add_parser("resume-evaluation")
    resume_evaluation.add_argument("--spec", required=True)

    collect = subparsers.add_parser("collect-batch")
    collect.add_argument("--batch-dir", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--study-dir", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_study(Path(args.study_dir), Path(args.search_config), Path(args.template_config))
    elif args.command == "propose-batch":
        batch = propose_batch(Path(args.study_dir), args.count, args.batch_id)
        print(batch)
    elif args.command == "enqueue-refinement":
        print(enqueue_refinement(Path(args.study_dir), Path(args.config)))
    elif args.command == "run-trial":
        run_one_trial(Path(args.spec))
    elif args.command == "resume-evaluation":
        resume_trial_evaluation(Path(args.spec))
    elif args.command == "collect-batch":
        collect_batch(Path(args.batch_dir))
    elif args.command == "summarize":
        print(json.dumps(study_summary(Path(args.study_dir)), indent=2, ensure_ascii=False))


def prepare_study(study_dir: Path, search_config_path: Path, template_config_path: Path) -> None:
    import optuna

    study_dir = study_dir.expanduser().resolve()
    search_config_path = search_config_path.expanduser().resolve()
    template_config_path = template_config_path.expanduser().resolve()
    search = read_yaml(search_config_path)
    template = read_yaml(template_config_path)
    project_root = Path(template["project"]["root"]).expanduser().resolve()
    if not study_dir.is_relative_to(project_root):
        raise ValueError("The HPO study directory must be inside the configured project root")

    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "batches").mkdir(exist_ok=True)
    (study_dir / "configs").mkdir(exist_ok=True)
    snapshot(search_config_path, study_dir / "search_config.yaml")
    snapshot(template_config_path, study_dir / "trial_template.yaml")
    ensure_training_manifest(template, search)

    study = create_or_load_study(study_dir, search)
    initial_trials = list(search["initial_trials"])
    if not study.trials:
        for values in initial_trials:
            params = {key: value for key, value in values.items() if key != "name"}
            study.enqueue_trial(params, user_attrs={"initialization": str(values["name"])})

    manifest = {
        "schema_version": 1,
        "study_name": study.study_name,
        "created_or_verified_at": now(),
        "storage": str((study_dir / "optuna.sqlite3").resolve()),
        "search_config_sha256": sha256(study_dir / "search_config.yaml"),
        "trial_template_sha256": sha256(study_dir / "trial_template.yaml"),
        "reward": search["reward_construction"]["value"],
        "training_prompts": int(search["short_trial"]["training_prompts"]),
        "validation_prompts": int(search["short_trial"]["validation_prompts"]),
        "evaluation_steps": list(search["short_trial"]["evaluation_steps"]),
        "initial_trials": len(initial_trials),
        "full_training_launched": False,
    }
    write_json(study_dir / "study_manifest.json", manifest)
    write_json(study_dir / "study_summary.json", study_summary(study_dir))


def create_or_load_study(study_dir: Path, search: dict[str, Any]):
    import optuna

    sampler_config = search["search"]["sampler"]
    pruner_config = search["search"]["pruner"]
    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_config["seed"]),
        multivariate=bool(sampler_config["multivariate"]),
        n_startup_trials=int(sampler_config["startup_trials"]),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_config["startup_trials"]),
        n_warmup_steps=int(pruner_config["warmup_steps"]),
        interval_steps=int(pruner_config["interval_steps"]),
    )
    storage = f"sqlite:///{(study_dir / 'optuna.sqlite3').resolve()}"
    return optuna.create_study(
        study_name=str(search["hpo_study_name"]),
        storage=storage,
        direction=str(search["selection"]["direction"]),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def propose_batch(study_dir: Path, count: int, batch_id: str | None) -> Path:
    if count <= 0:
        raise ValueError("Batch count must be positive")
    study_dir = study_dir.expanduser().resolve()
    search = read_yaml(study_dir / "search_config.yaml")
    template = read_yaml(study_dir / "trial_template.yaml")
    study = create_or_load_study(study_dir, search)
    batch_id = batch_id or datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%SZ")
    batch_dir = study_dir / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    specs_dir = batch_dir / "specs"
    results_dir = batch_dir / "results"
    specs_dir.mkdir()
    results_dir.mkdir()

    spec_paths: list[Path] = []
    for _ in range(count):
        trial = study.ask()
        params = suggest_parameters(trial, search["search"]["space"])
        initialization = trial.user_attrs.get("initialization")
        refinement = trial.user_attrs.get("refinement")
        source = (
            f"initial:{initialization}"
            if initialization
            else f"refinement:{refinement}"
            if refinement
            else "tpe"
        )
        spec = materialize_trial(
            study_dir=study_dir,
            batch_dir=batch_dir,
            template=template,
            search=search,
            trial_number=trial.number,
            params=params,
            seed=int(search["short_trial"]["fixed_seed"]),
            source=source,
        )
        spec_path = specs_dir / f"trial_{trial.number:04d}.json"
        write_json(spec_path, spec)
        spec_paths.append(spec_path)

    (batch_dir / "trial_specs.txt").write_text(
        "".join(f"{path.resolve()}\n" for path in spec_paths), encoding="utf-8"
    )
    write_json(
        batch_dir / "batch_manifest.json",
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "created_at": now(),
            "count": len(spec_paths),
            "trial_numbers": [int(path.stem.split("_")[-1]) for path in spec_paths],
            "status": "proposed",
        },
    )
    return batch_dir.resolve()


def enqueue_refinement(study_dir: Path, config_path: Path) -> Path:
    """Add an auditable, idempotent set of local ablations to an Optuna study."""
    study_dir = study_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    search = read_yaml(study_dir / "search_config.yaml")
    payload = read_yaml(config_path)
    entries = list(payload.get("configurations") or [])
    if not entries:
        raise ValueError("Refinement config must contain at least one configuration")

    refinement_dir = study_dir / "refinements" / config_path.stem
    refinement_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = refinement_dir / "requested.yaml"
    manifest_path = refinement_dir / "manifest.json"
    if manifest_path.exists():
        if not snapshot_path.exists() or snapshot_path.read_bytes() != config_path.read_bytes():
            raise RuntimeError("Existing refinement snapshot differs from the requested config")
        return manifest_path
    snapshot(config_path, snapshot_path)

    study = create_or_load_study(study_dir, search)
    queued: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry["name"])
        rationale = str(entry["rationale"])
        params = dict(entry["parameters"])
        validate_refinement_params(params, search["search"]["space"])
        study.enqueue_trial(
            params,
            user_attrs={"refinement": name, "rationale": rationale},
            skip_if_exists=True,
        )
        queued.append({"name": name, "rationale": rationale, "parameters": params})
    manifest = {
        "schema_version": 1,
        "study_name": study.study_name,
        "created_at": now(),
        "source_config": str(snapshot_path),
        "configurations": queued,
        "full_training_launched": False,
    }
    write_json(manifest_path, manifest)
    write_json(study_dir / "study_summary.json", study_summary(study_dir))
    return manifest_path


def validate_refinement_params(params: dict[str, Any], space: dict[str, list[Any]]) -> None:
    if set(params) != set(space):
        missing = sorted(set(space) - set(params))
        extra = sorted(set(params) - set(space))
        raise ValueError(f"Refinement parameter mismatch; missing={missing}, extra={extra}")
    for name, value in params.items():
        if value not in list(space[name]):
            raise ValueError(f"Refinement value {name}={value!r} is outside the study space")


def suggest_parameters(trial, space: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        name: trial.suggest_categorical(name, list(choices))
        for name, choices in space.items()
    }


def materialize_trial(
    *,
    study_dir: Path,
    batch_dir: Path,
    template: dict[str, Any],
    search: dict[str, Any],
    trial_number: int,
    params: dict[str, Any],
    seed: int,
    source: str,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    study_name = str(search["hpo_study_name"])
    run_id = f"hpo_logp_trial_{trial_number:04d}_seed{seed}"
    config["project"]["seed"] = int(seed)
    config["trial"]["run_name"] = run_id
    config["training"]["learning_rate"] = float(params["learning_rate"])
    config["training"]["beta"] = float(params["beta"])
    config["training"]["temperature"] = float(params["temperature"])
    config["training"]["scale_rewards"] = str(params["scale_rewards"])
    config["actor"]["lora"]["r"] = int(params["lora_r"])
    config["actor"]["lora"]["dropout"] = float(params["lora_dropout"])
    config["paths"]["experiments"] = f"experiments/{study_name}/trials"
    config["paths"]["checkpoints"] = f"models/checkpoints/{study_name}"
    config["hpo"] = {
        "study_name": study_name,
        "trial_number": int(trial_number),
        "source": source,
        "batch_id": batch_dir.name,
        "parameters": params,
    }
    config_path = study_dir / "configs" / f"trial_{trial_number:04d}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "schema_version": 1,
        "study_name": study_name,
        "trial_number": int(trial_number),
        "seed": int(seed),
        "source": source,
        "parameters": params,
        "run_id": run_id,
        "config_path": str(config_path.resolve()),
        "result_path": str((batch_dir / "results" / f"trial_{trial_number:04d}.json").resolve()),
        "experiment_dir": str(
            (
                Path(config["project"]["root"])
                / config["paths"]["experiments"]
                / run_id
            ).resolve()
        ),
    }


def run_one_trial(spec_path: Path) -> None:
    spec_path = spec_path.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result_path = Path(spec["result_path"])
    result = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "trial_number": spec["trial_number"],
        "run_id": spec["run_id"],
        "seed": spec["seed"],
        "parameters": spec["parameters"],
        "started_at": now(),
    }
    commands = [
        [sys.executable, "-m", "iar_grpo.preflight", "--config", spec["config_path"], "--require-training-deps"],
        [sys.executable, "-m", "iar_grpo.train", "--config", spec["config_path"], "--run-id", spec["run_id"]],
        [sys.executable, "-m", "iar_grpo.evaluate_hpo", "--config", spec["config_path"], "--run-id", spec["run_id"]],
    ]
    try:
        for command in commands:
            subprocess.run(command, check=True)
        objective_path = Path(spec["experiment_dir"]) / "objective.json"
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        result.update(
            {
                "status": "complete",
                "completed_at": now(),
                "objective": objective,
                "objective_path": str(objective_path),
            }
        )
        write_json(result_path, result)
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "completed_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(result_path, result)
        raise


def resume_trial_evaluation(spec_path: Path) -> None:
    """Resume only evaluation after a completed training and preserve failure provenance."""
    spec_path = spec_path.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result_path = Path(spec["result_path"])
    experiment_dir = Path(spec["experiment_dir"])
    training_summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    if training_summary.get("status") != "complete":
        raise RuntimeError("Cannot resume evaluation because training did not complete")
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    )
    objective_path = experiment_dir / "objective.json"
    archived_evaluations: str | None = None
    if not objective_path.exists():
        validation_dir = experiment_dir / "evaluations" / "validation"
        if validation_dir.exists():
            recovery_root = experiment_dir / "recovery_incomplete"
            recovery_root.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive = recovery_root / f"validation_{stamp}"
            shutil.move(str(validation_dir), str(archive))
            archived_evaluations = str(archive)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "iar_grpo.evaluate_hpo",
                "--config",
                spec["config_path"],
                "--run-id",
                spec["run_id"],
            ],
            check=True,
        )
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    recovered = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "trial_number": spec["trial_number"],
        "run_id": spec["run_id"],
        "seed": spec["seed"],
        "parameters": spec["parameters"],
        "started_at": previous_result.get("started_at") if previous_result else None,
        "status": "complete",
        "completed_at": now(),
        "objective": objective,
        "objective_path": str(objective_path),
        "recovery": {
            "resumed_at": now(),
            "reason": "evaluation_only_recovery_after_completed_training",
            "archived_incomplete_evaluations": archived_evaluations,
            "previous_result": previous_result,
        },
    }
    write_json(result_path, recovered)


def collect_batch(batch_dir: Path) -> None:
    import optuna

    batch_dir = batch_dir.expanduser().resolve()
    study_dir = batch_dir.parent.parent
    search = read_yaml(study_dir / "search_config.yaml")
    study = create_or_load_study(study_dir, search)
    specs = [Path(line) for line in (batch_dir / "trial_specs.txt").read_text().splitlines() if line]
    missing: list[int] = []
    collected: list[dict[str, Any]] = []
    state_by_number = {trial.number: trial.state for trial in study.trials}
    for spec_path in specs:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        number = int(spec["trial_number"])
        result_path = Path(spec["result_path"])
        if not result_path.exists():
            missing.append(number)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        current_state = state_by_number[number]
        if current_state.is_finished():
            collected.append({"trial_number": number, "status": f"already_{current_state.name.lower()}"})
            continue
        if result["status"] == "complete":
            objective = float(result["objective"]["objective_value"])
            study.tell(number, objective)
            collected.append(
                {
                    "trial_number": number,
                    "status": "complete",
                    "objective_value": objective,
                    "selected_step": result["objective"].get("selected_step"),
                    "quality_feasible": result["objective"].get("quality_feasible"),
                }
            )
        else:
            study.tell(number, state=optuna.trial.TrialState.FAIL)
            collected.append({"trial_number": number, "status": "failed"})

    collection = {
        "schema_version": 1,
        "collected_at": now(),
        "missing": missing,
        "trials": collected,
    }
    write_json(batch_dir / "collection.json", collection)
    manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "collected_at": now(),
            "status": "incomplete" if missing else "collected",
            "missing": missing,
        }
    )
    write_json(batch_dir / "batch_manifest.json", manifest)
    write_json(study_dir / "study_summary.json", study_summary(study_dir))
    print(json.dumps(collection, indent=2, ensure_ascii=False))
    if missing:
        raise RuntimeError(f"Trial results are missing: {missing}")


def study_summary(study_dir: Path) -> dict[str, Any]:
    import optuna

    study_dir = study_dir.expanduser().resolve()
    search = read_yaml(study_dir / "search_config.yaml")
    study = create_or_load_study(study_dir, search)
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "initialization": trial.user_attrs.get("initialization"),
            }
        )
    completed = [row for row in rows if row["state"] == "COMPLETE"]
    completed.sort(key=lambda row: float(row["value"]), reverse=True)
    return {
        "schema_version": 1,
        "study_name": study.study_name,
        "updated_at": now(),
        "trial_count": len(rows),
        "state_counts": {
            state.name: sum(trial.state == state for trial in study.trials)
            for state in optuna.trial.TrialState
        },
        "best_trial": completed[0] if completed else None,
        "ranking": completed,
        "trials": rows,
        "full_training_launched": False,
    }


def ensure_training_manifest(template: dict[str, Any], search: dict[str, Any]) -> Path:
    from iar_phase0.config import load_config
    from iar_phase0.data import load_mbpp, materialize_split

    project_root = Path(template["project"]["root"]).expanduser().resolve()
    manifest_path = project_root / str(template["dataset"]["index_manifest"])
    split = str(template["dataset"]["split"])
    count = int(search["short_trial"]["training_prompts"])
    seed = int(search["short_trial"]["fixed_seed"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest.get("dataset_indices", [])) != count or int(manifest.get("seed")) != seed:
            raise ValueError(f"Existing training manifest is inconsistent: {manifest_path}")
        return manifest_path

    phase0 = load_config(project_root / str(template["dataset"]["phase0_config"]))
    dataset = load_mbpp(phase0)
    examples = materialize_split(dataset, split)
    if count % 5:
        raise ValueError("The fixed five-stratum selector requires a prompt count divisible by five")
    rng = random.Random(seed)
    per_stratum = count // 5
    selected: list[int] = []
    strata: list[dict[str, Any]] = []
    for stratum in range(5):
        start = round(stratum * len(examples) / 5)
        end = round((stratum + 1) * len(examples) / 5)
        indices = sorted(rng.sample(range(start, end), per_stratum))
        selected.extend(indices)
        strata.append({"start": start, "end": end, "selected": indices})
    payload = {
        "schema_version": 1,
        "dataset": str(template["dataset"]["name"]),
        "split": split,
        "seed": seed,
        "strategy": "five_equal_contiguous_source_strata_without_replacement",
        "dataset_indices": selected,
        "task_ids": [str(examples[index].task_id) for index in selected],
        "strata": strata,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, payload)
    return manifest_path


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def snapshot(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Study snapshot differs from requested input: {destination}")
        return
    shutil.copy2(source, destination)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
