"""Offline queue tests: no GPU training and no W&B writes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_v3_exploration_queue as queue


@pytest.fixture
def baseline(tmp_path):
    jobs, entries = [], {}
    values = {.05: [300, 20, 20, 20], .1: [100]*4, .25: [120]*4, .5: [130]*4, 1.: [110]*4}
    for t in queue.TEMPERATURES:
        for seed in queue.SEEDS:
            name = f"T{t}_seed{seed}"
            output = tmp_path / "baseline" / name
            run = output / "fresh-run"
            run.mkdir(parents=True)
            cfg = {"alg": {"actor": {"temperature": t}}, "seed": seed, "env_name": "Ant-v4",
                   "total_steps": 1_000_000, "wandb": {"project": "v3_test"},
                   "runtime": {"git_commit": "base", "git_dirty": False}}
            done = {"timesteps": 1_000_000, "updates": 995_000,
                    "wandb_url": f"https://wandb.ai/OptiQ/v3_test/runs/test{seed}"}
            queue.save(run / "config.json", cfg)
            queue.save(run / "completed.json", done)
            for kind in ("actor", "critic"):
                (run / f"{kind}_state_1000000.msgpack").write_bytes(b"test-checkpoint")
            steps = np.arange(900_000, 1_000_001, 5000)
            results = np.full((len(steps), 10), values[t][seed], dtype=float)
            np.savez(run / "evaluations.npz", timesteps=steps, results=results)
            jobs.append({"id": name, "temperature": t, "seed": seed, "output_root": str(output),
                         "resolved_config": cfg})
            entries[name] = {"state": "completed", "returncode": 0}
    return {"jobs": jobs, "source_commit": "base"}, {"phase": "completed", "jobs": entries}


def test_selects_one_temperature_by_equal_four_seed_mean(baseline):
    manifest, state = baseline
    selection = queue.select_temperature(manifest, state)
    assert selection["temperature"] == .5
    assert len(selection["baseline_records"]) == 20
    assert next(x for x in selection["temperature_scores"] if x["temperature"] == .05)["mean"] == 90


@pytest.mark.parametrize("status", ["running", "queued", "failed", "interrupted"])
def test_never_selects_while_even_one_of_twenty_is_not_complete(baseline, status):
    manifest, state = baseline
    state["jobs"][manifest["jobs"][-1]["id"]]["state"] = status
    assert queue.select_temperature(manifest, state) is None


def test_missing_final_artifact_does_not_fall_back_to_available_temperatures(baseline):
    manifest, state = baseline
    output = Path(manifest["jobs"][-1]["output_root"]) / "fresh-run"
    (output / "completed.json").unlink()
    with pytest.raises(FileNotFoundError):
        queue.select_temperature(manifest, state)


def test_last_checkpoint_spike_does_not_replace_full_tail_average(baseline):
    manifest, state = baseline
    item = manifest["jobs"][-1]
    path = Path(item["output_root"]) / "fresh-run" / "evaluations.npz"
    with np.load(path) as f:
        steps, results = f["timesteps"], f["results"]
    results[-1] = 1000
    np.savez(path, timesteps=steps, results=results)
    assert queue.select_temperature(manifest, state)["temperature"] == .5


def test_exact_tie_chooses_lower_temperature(baseline):
    manifest, state = baseline
    for item in manifest["jobs"]:
        if item["temperature"] == .25:
            path = Path(item["output_root"]) / "fresh-run" / "evaluations.npz"
            np.savez(path, timesteps=np.arange(900_000, 1_000_001, 5000), results=np.full((21, 10), 130.))
    assert queue.select_temperature(manifest, state)["temperature"] == .25


def test_missing_seed_in_manifest_is_invalid(baseline):
    manifest, state = baseline
    manifest["jobs"].pop()
    with pytest.raises(ValueError, match="exactly"):
        queue.select_temperature(manifest, state)


def test_all_sixteen_resolved_configs_share_one_temperature(tmp_path):
    campaign = queue.Campaign(tmp_path, tmp_path / "baseline")
    jobs = campaign.make_jobs(.25)
    assert len(jobs) == 16
    assert {j["temperature"] for j in jobs} == {.25}
    assert {j["resolved_config"]["alg"]["actor"]["temperature"] for j in jobs} == {.25}
    for name, mode, target in queue.VARIANTS:
        subset = [j for j in jobs if j["variant"] == name]
        assert {j["seed"] for j in subset} == {0, 1, 2, 3}
        assert all(j["gpu"] == j["seed"] for j in subset)
        assert all(j["mode"] == mode and j["target_entropy_per_dim"] == target for j in subset)
        assert all(j["resolved_config"]["total_steps"] == 1_000_000 for j in subset)


def test_controller_waits_then_launches_all_variants_without_gpu_overlap(tmp_path, monkeypatch, baseline):
    campaign = queue.Campaign(tmp_path / "followup", tmp_path / "predecessor")
    campaign.root.mkdir()
    (campaign.root / "logs").mkdir()
    campaign.baseline.mkdir()
    manifest, state = baseline
    queue.save(campaign.baseline / "manifest.json", manifest)
    full_state = copy.deepcopy(state)
    state["phase"] = "running"
    state["jobs"][manifest["jobs"][-1]["id"]]["state"] = "running"
    queue.save(campaign.baseline / "state.json", state)
    jobs = campaign.make_jobs(.5)
    queue.save(campaign.manifest_path, {"source_identity": {"commit": "followup"}})
    followup_state = {"phase": "waiting_for_all_baseline_runs", "selected_temperature": None,
        "jobs": {j["id"]: {k: j[k] for k in ("id", "variant", "seed", "gpu", "mode")}
        for j in jobs}}
    for item in followup_state["jobs"].values():
        item["state"] = "waiting_for_baseline"
    queue.save(campaign.state_path, followup_state)
    monkeypatch.setattr(campaign, "verify_frozen_inputs", lambda manifest: None)
    monkeypatch.setattr(campaign, "verify_launch", lambda item, state: None)
    monkeypatch.setattr(campaign, "make_jobs", lambda temperature: jobs)
    monkeypatch.setattr(queue.signal, "signal", lambda *a: None)
    launched, active_gpus, sleeps = [], set(), []
    real_completed_result = queue.completed_result
    def completion(item, commit, **kwargs):
        if commit == "base":
            return real_completed_result(item, commit)
        return {"mean_return_900k_1m": 1., "timesteps": 1_000_000}
    monkeypatch.setattr(queue, "completed_result", completion)
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            assert not launched and not (campaign.root / "selection.json").exists()
            queue.save(campaign.baseline / "state.json", full_state)
    monkeypatch.setattr(queue.time, "sleep", sleep)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k: "")
    class Process:
        def __init__(self, command, **kwargs):
            assert queue.read(campaign.root / "selection.json")["temperature"] == .5
            self.gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            assert self.gpu not in active_gpus
            active_gpus.add(self.gpu)
            assert len(active_gpus) <= 4
            self.pid = 10000 + len(launched)
            self.returncode = None
            launched.append(command)
        def poll(self):
            if self.returncode is None:
                active_gpus.remove(self.gpu)
                self.returncode = 0
            return self.returncode
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    campaign.run()
    final = queue.read(campaign.state_path)
    assert len(launched) == 16 and final["phase"] == "completed"
    assert final["selected_temperature"] == .5
    assert {j["temperature"] for j in final["jobs"].values()} == {.5}
    assert all(j["state"] == "completed" for j in final["jobs"].values())
