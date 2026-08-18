from pathlib import Path

import pytest

from optiq.train import parse_args

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/mujoco/default.yaml"


@pytest.mark.parametrize(
    ("name", "environment", "steps"),
    (
        ("hopper", "Hopper-v4", 1_000_000),
        ("walker2d", "Walker2d-v4", 1_500_000),
        ("halfcheetah", "HalfCheetah-v4", 3_000_000),
        ("ant", "Ant-v4", 2_000_000),
        ("humanoid", "Humanoid-v4", 5_000_000),
    ),
)
def test_mujoco_config_merge(name, environment, steps):
    args = parse_args(
        [
            "--config",
            str(BASE_CONFIG),
            "--config",
            str(ROOT / f"configs/mujoco/envs/{name}.yaml"),
        ]
    )

    assert args.env == environment
    assert args.total_steps == steps
    assert args.hidden_dims == (256, 256, 256)
    assert args.num_policy_samples == 16
    assert args.proposals_per_policy_sample == 5
    assert args.eval_interval == 5_000
    assert args.eval_episodes == 10


def test_cli_overrides_config():
    args = parse_args(
        [
            "--config",
            str(BASE_CONFIG),
            "--total-steps",
            "7",
            "--hidden-dims",
            "32,32",
        ]
    )

    assert args.total_steps == 7
    assert args.hidden_dims == (32, 32)


def test_unknown_config_key_is_rejected(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("totl_steps: 1\n")

    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])
