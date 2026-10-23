from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .configure import CONDITIONS, prepare
from .paths import ROOT, DOMAINS, SEEDS, defaults


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper reproduction: verify, configure, train, evaluate, analyze, and plot."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "verify", help="Offline integrity/configuration checks; no model calls"
    )
    check.add_argument(
        "--external",
        action="store_true",
        help="Also require cached actor weights and Spider assets",
    )
    check.add_argument("--output", type=Path)
    for name in ("configure", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--domain", choices=DOMAINS, required=True)
        command.add_argument("--condition", choices=CONDITIONS, required=True)
        command.add_argument("--seed", type=int, choices=SEEDS, default=23)
        command.add_argument("--run-id", default="attempt001")
        if name == "train":
            command.add_argument("--resume", type=Path)
        if name == "evaluate":
            command.add_argument("--step", type=int)
            command.add_argument("--workers", type=int, default=8)
    figures = sub.add_parser(
        "figures", help="Regenerate the manuscript's empirical figures offline"
    )
    figures.add_argument(
        "--output-dir", type=Path, default=defaults()["runs"] / "figures"
    )
    analysis = sub.add_parser(
        "analyze",
        help="Recompute analyses from recorded rollouts, labels, and judge scores",
    )
    analysis.add_argument(
        "--output-dir", type=Path, default=defaults()["runs"] / "reanalysis"
    )
    analysis.add_argument(
        "--extended",
        action="store_true",
        help="Also rerun bootstrap calibration and logistic controls",
    )
    causal = sub.add_parser(
        "causal",
        help="Reanalyze recorded counterfactual intervention judgments offline",
    )
    causal.add_argument("--bootstrap", type=int, default=5000)
    probes = sub.add_parser(
        "probes",
        help="Recompute recorded reward pilot, pairwise order effects, and K10 training windows",
    )
    probes.add_argument(
        "--output", type=Path, default=defaults()["runs"] / "probes.json"
    )
    pairwise = sub.add_parser(
        "pairwise", help="Configure, regenerate, or rejudge the fixed pairwise probe"
    )
    pairwise.add_argument(
        "action", choices=("configure", "generate", "judge", "collect")
    )
    pairwise.add_argument("--domain", choices=DOMAINS, required=True)
    pairwise.add_argument("--run-id", default="attempt001")
    screen = sub.add_parser(
        "teacher-screen",
        help="Regenerate teacher selection using frozen 100-task inputs",
    )
    screen.add_argument(
        "action",
        choices=(
            "configure",
            "actors",
            "teacher",
            "judge",
            "collect-teacher",
            "collect-judge",
        ),
    )
    screen.add_argument("--domain", choices=DOMAINS, required=True)
    screen.add_argument("--run-id", default="attempt001")
    screen.add_argument("--shard-index", type=int, choices=(0, 1, 2), default=0)
    pilot = sub.add_parser(
        "pilot", help="Configure/train/validate a short reward-form pilot"
    )
    pilot.add_argument("action", choices=("configure", "train", "evaluate"))
    pilot.add_argument("--domain", choices=("gsm8k", "kodcode"), required=True)
    pilot.add_argument(
        "--form",
        choices=("log_p_own", "p_own", "logit_p_own", "binary_own"),
        required=True,
    )
    pilot.add_argument("--run-id", default="attempt001")
    pilot.add_argument("--step", type=int, choices=(0, 187, 374), default=374)
    teacher = sub.add_parser(
        "teacher-corpus", help="Configure or regenerate the SFT teacher corpus (paid)"
    )
    teacher.add_argument("--domain", choices=DOMAINS, required=True)
    teacher.add_argument("--run-id", default="attempt001")
    teacher.add_argument("--execute", action="store_true")
    tables = sub.add_parser(
        "tables", help="Recompute paired held-out statistics from recorded outcomes"
    )
    tables.add_argument(
        "--output", type=Path, default=defaults()["runs"] / "heldout_stats.json"
    )
    smoke = sub.add_parser(
        "smoke",
        help="API-free synthetic training/evaluation checks, never a paper result",
    )
    smoke.add_argument(
        "--output-dir", type=Path, default=defaults()["runs"] / "synthetic_smoke"
    )
    args = parser.parse_args()
    if args.command in {"tables", "probes", "analyze", "causal", "figures"}:
        from .records import manifest

        try:
            manifest()
        except FileNotFoundError as exc:
            parser.exit(2, str(exc) + "\n")
    if args.command == "verify":
        from .verify import verify

        result = verify(external=args.external)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "configurations"}, indent=2
            )
        )
    elif args.command == "configure":
        print(prepare(args.domain, args.condition, args.seed, run_id=args.run_id).path)
    elif args.command == "train":
        from .training import train, train_sft

        if args.condition == "a0":
            parser.error("A0 is evaluated without training")
        if args.condition == "sft":
            if args.resume:
                parser.error("SFT resume is not exposed by the portable command")
            result = train_sft(args.domain, args.seed, args.run_id)
        else:
            result = train(
                args.domain, args.condition, args.seed, args.run_id, resume=args.resume
            )
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        config = prepare(args.domain, args.condition, args.seed, run_id=args.run_id)
        step = (
            args.step
            if args.step is not None
            else (
                0
                if args.condition == "a0"
                else 500
                if args.condition == "sft"
                else 2000
            )
        )
        test_name = {
            "commonsenseqa": "test.jsonl",
            "gsm8k": "test.jsonl",
            "kodcode": "test1000.jsonl",
            "spider": "test1034.jsonl",
        }[args.domain]
        output = (
            Path(config.raw["paths"]["experiments"])
            / args.run_id
            / "evaluations/test"
            / f"step_{step:04d}"
        )
        command = [
            sys.executable,
            str(ROOT / "scripts/tools/heldout_eval.py"),
            "--config",
            str(config.path),
            "--run-id",
            args.run_id,
            "--step",
            str(step),
            "--test-file",
            str(ROOT / "data" / args.domain / test_name),
            "--output-dir",
            str(output),
            "--workers",
            str(args.workers),
            "--reference-gate",
            str(ROOT / "data/reference_gate_valid_ids.json"),
        ]
        if args.condition == "sft" and step == 500:
            command += [
                "--adapter",
                str(
                    Path(config.raw["paths"]["checkpoints"])
                    / args.run_id
                    / "final_adapter"
                ),
            ]
        subprocess.run(command, check=True)
    elif args.command == "tables":
        from .records import evaluation_files
        from .tables import recompute

        args.output.parent.mkdir(parents=True, exist_ok=True)
        inputs = args.output.with_suffix(".inputs.json")
        inputs.write_text(json.dumps(evaluation_files(), indent=2))
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analysis/heldout_stats.py"),
                "--files",
                str(inputs),
                "--output",
                str(args.output),
            ],
            check=True,
        )
        recompute(args.output.with_name("main_table.json"))
    elif args.command == "figures":
        from .figures import build

        print(json.dumps(build(args.output_dir), indent=2))
    elif args.command == "analyze":
        from .analysis import run

        print(json.dumps(run(args.output_dir, extended=args.extended), indent=2))
    elif args.command == "causal":
        from .causal import run

        print(json.dumps(run(bootstrap=args.bootstrap), indent=2))
    elif args.command == "probes":
        from .probes import run

        print(json.dumps(run(args.output), indent=2))
    elif args.command == "pairwise":
        from .pairwise import run

        print(json.dumps(run(args.domain, args.action, args.run_id), indent=2))
    elif args.command == "teacher-screen":
        from .screen import run

        print(
            json.dumps(
                run(args.domain, args.action, args.run_id, args.shard_index), indent=2
            )
        )
    elif args.command == "teacher-corpus":
        from .teacher import corpus

        print(corpus(args.domain, args.run_id, execute=args.execute))
    elif args.command == "pilot":
        from .pilot import prepare as prepare_pilot

        config = prepare_pilot(args.domain, args.form, args.run_id)
        if args.action == "configure":
            print(config.path)
        elif args.action == "train":
            from .training import train

            print(
                json.dumps(
                    train(
                        args.domain,
                        "reward_form_pilot",
                        23,
                        args.run_id,
                        reward_form=args.form,
                    ),
                    indent=2,
                )
            )
        else:
            name = (
                "validation.jsonl" if args.domain == "gsm8k" else "validation300.jsonl"
            )
            output = (
                Path(config.raw["paths"]["experiments"])
                / args.run_id
                / "evaluations/validation"
                / f"step_{args.step:04d}"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/tools/heldout_eval.py"),
                    "--config",
                    str(config.path),
                    "--run-id",
                    args.run_id,
                    "--step",
                    str(args.step),
                    "--test-file",
                    str(ROOT / "data" / args.domain / name),
                    "--validation",
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
    elif args.command == "smoke":
        from .smoke import run

        print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
