from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest

from iar_repro.configure import prepare
from iar_repro.paths import ROOT, DOMAINS, SEEDS, sha256
from iar_repro.pilot import FORMS, prepare as pilot_config
from iar_repro.teacher import corpus


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("IAR_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts/analysis"))


@pytest.mark.parametrize("domain", DOMAINS)
def test_all_main_configs_preserve_frozen_templates(domain):
    before = {str(p): sha256(p) for p in (ROOT / "configs").rglob("*") if p.is_file()}
    for condition in ("indist", "quality", "oracle", "sft", "closed_weight_k10", "a0"):
        for seed in (23,) if condition in ("closed_weight_k10", "a0") else SEEDS:
            c = prepare(domain, condition, seed)
            assert c.raw["project"]["seed"] == seed
            assert Path(c.raw["dataset"]["split_file"]).is_file()
            assert c.raw["training"]["max_steps"] == (
                500 if condition == "sft" else 2000
            )
            p = json.loads(c.path.with_suffix(".provenance.json").read_text())
            assert p["resolved_config_sha256"] == sha256(c.path)
            if condition == "sft":
                assert c.raw["evaluation"]["checkpoints"] == [0, 125, 250, 375, 500]
                assert c.raw["training"]["completion_only_loss"] is True
                assert c.raw["training"]["shuffle_dataset"] is False
    assert all(sha256(Path(p)) == digest for p, digest in before.items())


@pytest.mark.parametrize("domain", ("gsm8k", "kodcode"))
def test_pilot_budget_and_validation_are_distinct(domain):
    for form in FORMS:
        c = pilot_config(domain, form)
        assert c.expected_reward_calls == 1496
        assert c.raw["reward"]["max_http_requests"] == 7480
        assert "test" not in Path(c.raw["evaluation"]["validation_file"]).name
        assert c.raw["training"]["max_steps"] == 374
        if domain == "kodcode":
            gate = json.loads(
                Path(c.raw["evaluation"]["kodcode_scoring_protocol"]).read_text()
            )
            assert gate["decision"] == "GO"
            assert len(set(gate["scored_ids"])) == 299
            assert gate["validation_file_sha256"] == sha256(
                Path(c.raw["evaluation"]["validation_file"])
            )


@pytest.mark.parametrize("domain", DOMAINS)
def test_teacher_generator_validates_correct_frozen_cohort(domain):
    path = corpus(domain, "attempt001")
    c = json.loads(path.read_text())
    assert c["expected_count"] == 1000
    assert c["protocol"] == (
        "campaign-step15-sft-teacher-v1"
        if domain in ("kodcode", "spider")
        else "campaign-step7-sft-teacher-v1"
    )
    assert c["training_split_sha256"] == sha256(Path(c["training_split"]))


@pytest.mark.parametrize("run_id", ("../escape", "/tmp/escape", "a/b", ""))
def test_run_id_cannot_escape(run_id):
    with pytest.raises(ValueError):
        prepare("gsm8k", "indist", 23, run_id=run_id)


@pytest.mark.parametrize("domain,n", (("kodcode", 992), ("spider", 1034)))
def test_reference_gate_requires_original_gate_and_test_hash(tmp_path, domain, n):
    evaluator = module("test_heldout_repro", "scripts/tools/heldout_eval.py")
    path = ROOT / "data/reference_gate_valid_ids.json"
    name = "test1000.jsonl" if domain == "kodcode" else "test1034.jsonl"
    kwargs = dict(
        domain=domain, gate_path=path, test_sha256=sha256(ROOT / "data" / domain / name)
    )
    assert len(evaluator._valid_ids(**kwargs)) == n
    kwargs["test_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="drifted"):
        evaluator._valid_ids(**kwargs)
    changed = json.loads(path.read_text())
    changed[domain] = changed[domain][1:]
    tampered = tmp_path / "gate.json"
    tampered.write_text(json.dumps(changed))
    kwargs.update(gate_path=tampered, test_sha256=sha256(ROOT / "data" / domain / name))
    with pytest.raises(ValueError, match="drifted"):
        evaluator._valid_ids(**kwargs)


def test_average_ranks_and_auc_match_tied_reference():
    q = module("test_quality_analysis", "scripts/analysis/q_vs_pown.py")
    s = np.array([2.0, 1.0, 2.0, 4.0, 1.0, 4.0])
    y = np.array([1, 0, 0, 1, 1, 0])
    assert q.average_ranks(s).tolist() == [3.5, 1.5, 3.5, 5.5, 1.5, 5.5]
    pairs = [float(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]]
    assert q.auc(y, s) == pytest.approx(np.mean(pairs))


def receipt(label):
    return {
        "id": "synthetic-receipt",
        "model": "openai/gpt-5-mini",
        "provider": "OpenAI",
        "service_tier": "flex",
        "choices": [{"message": {"content": json.dumps({"label": label})}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "cost": 0.00001,
        },
        "openrouter_metadata": {
            "requested": "openai/gpt-5-mini",
            "attempt": 1,
            "endpoints": {"available": [{"provider": "OpenAI", "selected": True}]},
        },
    }


@pytest.mark.parametrize("invalid", (False, True))
def test_k10_retries_transients_but_not_invalid_labels(tmp_path, monkeypatch, invalid):
    import iar_campaign.kodcode_gpt5mini_k10 as k10
    import iar_closed_weight.gemini_pilot as client

    monkeypatch.setattr(
        k10, "PROTOCOL", ROOT / "configs/closed_weight_k10/kodcode_protocol.json"
    )
    assert k10.protocol()["reward"]["maximum_http_attempts_per_physical_call"] == 8
    settings = replace(
        k10._settings(log_dir=tmp_path, canary=False), max_retry_wait_seconds=0.001
    )
    calls = []

    def handle(request):
        calls.append(request)
        if not invalid and len(calls) == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(200, json=receipt(3 if invalid else 1))

    runner = client.GeminiClosedWeightPilot(
        settings=settings,
        api_key="synthetic-not-a-key",
        log_path=tmp_path / "judgments.jsonl",
    )

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ) as transport:
            return await runner.judge_one(
                transport,
                {"call_id": "fixture", "prompt": "fixture", "temperature": 1.0},
                phase="full",
            )

    if invalid:
        with pytest.raises(RuntimeError, match="Judgment failed"):
            asyncio.run(check())
        assert len(calls) == 1
    else:
        assert asyncio.run(check()) == 1
        assert len(calls) == 2
    assert all(
        json.loads(request.content)["model"] == "openai/gpt-5-mini" for request in calls
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_pairwise_frozen_probe_is_nested_and_replay_configured(domain):
    from iar_repro.pairwise import configure
    from iar_campaign.generate import prompts_for_domain
    import iar_campaign.pairwise_experiment as original
    import yaml

    path, raw, examples = configure(domain)
    assert len(examples) == 100
    assert raw["generation_batch_size"] == (4 if domain == "kodcode" else 8)
    assert raw["gate_thresholds"]["complete_judge_call_coverage"] == 800
    assert json.loads(path.read_text()) == raw
    if domain != "spider":
        from iar_grpo.config import GrpoTrialConfig

        c = GrpoTrialConfig(
            yaml.safe_load(Path(raw["grpo_config"]).read_text()),
            Path(raw["grpo_config"]),
        )
        rows = [
            json.loads(line)
            for line in Path(c.raw["dataset"]["split_file"]).read_text().splitlines()
        ]
        prompts = prompts_for_domain(domain, rows, tables_json=None)["a0"]
        assert len(original._teacher_cache(c, rows, prompts)) == 1000


@pytest.mark.parametrize("domain", DOMAINS)
def test_teacher_screen_uses_tasks_without_bundled_responses(domain):
    from iar_repro.screen import configure
    import iar_campaign.judge_sanity as original
    import yaml

    path = configure(domain)
    raw = yaml.safe_load(path.read_text())
    assert raw["enabled_domains"] == [domain]
    assert not Path(raw["domains"][domain]["candidate_file"]).exists()
    original.validate_config(raw, require_candidates=False)
    with pytest.raises(FileNotFoundError):
        original.validate_config(raw)
    if domain != "spider":
        tasks = original.build_tasks(raw, "teacher")
        assert len(tasks) == 100
        assert len({row["logical_key"] for row in tasks}) == 100


@pytest.mark.parametrize("name", ("..", "../escape", "/tmp/escape", "a/b"))
@pytest.mark.parametrize("kind", ("pairwise", "screen"))
def test_probe_attempt_paths_are_safe(name, kind):
    from importlib import import_module

    configure = import_module("iar_repro." + kind).configure
    with pytest.raises(ValueError, match="run-id"):
        configure("gsm8k", name)


def test_release_archive_excludes_results_and_generated_state(tmp_path, monkeypatch):
    builder = module(
        "test_archive_builder", "scripts/tools/build_reproduction_package.py"
    )
    root = tmp_path / "source"
    root.mkdir()
    for relative in (
        "runs/private-output.txt",
        "src/example.egg-info/PKG-INFO",
        "reproduction/provenance/metadata.json",
        "src/example.py",
        "results/table.json",
        "reproduction/provenance/results/summary.json",
    ):
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n")
    monkeypatch.setattr(builder, "ROOT", root)
    names = {str(relative) for relative, _ in builder.files()}
    assert "reproduction/provenance/metadata.json" in names
    assert "src/example.py" in names
    assert not any(
        "egg-info" in name or "results" in Path(name).parts or name.startswith("runs/")
        for name in names
    )
    result = builder.build(tmp_path / "package.zip")
    assert result["status"] == "passed" and result["files"] == 2


@pytest.mark.parametrize("seed", SEEDS)
def test_sft_seeds_before_model_and_adapter_initialization(seed, monkeypatch):
    import torch
    import iar_campaign.sft_baseline as original
    from iar_repro.training import train_sft

    state = torch.random.get_rng_state()
    torch.manual_seed(999)

    def before_weights(_):
        assert torch.initial_seed() == seed
        raise RuntimeError("stop-before-loading-weights")

    monkeypatch.setattr(original, "_actor_source", before_weights)
    try:
        with pytest.raises(RuntimeError, match="stop-before-loading-weights"):
            train_sft("gsm8k", seed, "seed-regression")
    finally:
        torch.random.set_rng_state(state)


def test_external_records_are_optional_and_cannot_escape(tmp_path, monkeypatch):
    from iar_repro.records import manifest, result_path, materialize

    monkeypatch.setenv("IAR_RESULTS_ROOT", str(tmp_path / "external"))
    with pytest.raises(FileNotFoundError, match="Set IAR_RESULTS_ROOT"):
        manifest()
    with pytest.raises(ValueError, match="relative"):
        result_path("results/../../secret")
    import gzip

    directory = tmp_path / "external"
    directory.mkdir()
    data = gzip.compress(b'{"fixture": true}\n')
    (directory / "fixture.json.gz").write_bytes(data)
    import hashlib

    (directory / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "local": "results/fixture.json.gz",
                    "package_sha256": hashlib.sha256(data).hexdigest(),
                }
            ]
        )
    )
    assert json.loads(materialize("results/fixture.json.gz").read_text()) == {
        "fixture": True
    }
    (directory / "fixture.json.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="drifted"):
        materialize("results/fixture.json.gz")


def test_pairwise_judge_requires_fresh_generated_candidates():
    from iar_repro.pairwise import run

    with pytest.raises(FileNotFoundError, match="generate first"):
        run("gsm8k", "judge", "no-results")


def test_settings_extracts_keep_original_identity_checks():
    from iar_repro.paths import input_index, relocate, defaults

    record = input_index()[0]
    original = {
        "source": record["reference"],
        "source_sha256": record["original_sha256"],
    }
    resolved = relocate(original, **defaults())
    assert sha256(Path(resolved["source"])) == resolved["source_sha256"]
    with pytest.raises(ValueError, match="Historical input identity changed"):
        relocate({**original, "source_sha256": "0" * 64}, **defaults())


def test_source_archive_contains_no_experimental_outputs():
    builder = module(
        "test_source_only_builder", "scripts/tools/build_reproduction_package.py"
    )
    paths = [relative for relative, _ in builder.files()]
    assert paths
    assert not any(
        set(p.parts) & {"results", "runs", "audit", "checkpoints"} for p in paths
    )
    assert not any(str(p).endswith((".tar.gz", ".pdf", ".png")) for p in paths)
    for domain in DOMAINS:
        rows = [
            json.loads(line)
            for line in (ROOT / f"data/teacher_corpus/{domain}.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(rows) == 1000
        assert all(
            "teacher_correct" not in row and "teacher_valid" not in row for row in rows
        )


def test_screen_actor_generation_uses_the_portable_run_root(monkeypatch):
    from iar_repro.screen import run
    from iar_repro.paths import defaults
    import iar_campaign.generate as generation

    calls = []
    monkeypatch.setattr(
        generation, "run_generation", lambda **kwargs: calls.append(kwargs) or {}
    )
    run("gsm8k", "actors", "fresh-actors")
    assert calls[0]["campaign_root"] == defaults()["runs"]
    assert calls[0]["variants"] == ("a0", "prompt_imitation")
    assert calls[0]["split_file"].is_file()
