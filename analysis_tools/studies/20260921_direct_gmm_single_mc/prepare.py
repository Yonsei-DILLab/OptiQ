"""Build a new immutable source from verified 0917 DirectGMM + this overlay."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


def build(base, out):
    source = json.loads((base / 'DIRECT_GMM_SOURCE.json').read_text())
    assert source['code_id'] == 'fd918514023ae5ce20a91db53f19801549fd9d249e46e963b643d498834f5f28'
    assert not out.exists(), 'Never patch a running snapshot'
    for rel, digest in source['files'].items():
        p = base / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    p = out / 'optiq_dime/algorithm.py'
    t = p.read_text()
    t = replace_once(t, 'from .policy import OptiQPolicy',
                     'from .policy import OptiQPolicy\nfrom .single_mc import update as single_mc_update')
    t = replace_once(t, '            actor.get("ot_student_action", "sample"),\n        )',
                     '            actor.get("ot_student_action", "sample"),\n            int(self.cfg.alg.critic.get("backup_samples", 1)),\n        )')
    # The two JIT signatures each need K as a static shape parameter.
    t = replace_once(t, '            "backup_mode",\n        ],',
                     '            "backup_mode",\n            "mc_backup_samples",\n        ],')
    t = replace_once(t, '            "backup_mode",\n            "entropy_diagnostics",',
                     '            "backup_mode",\n            "mc_backup_samples",\n            "entropy_diagnostics",')
    assert t.count('            "mc_backup_samples",\n') == 2
    t = replace_once(t, '        backup_mode: str | None = None,\n    ):',
                     '        backup_mode: str | None = None,\n        mc_backup_samples: int = 1,\n    ):\n'
                     '        if mc_backup_samples > 1:\n'
                     '            if crossq_style or num_atoms != 1 or not semi_implicit or backup_mode != "td":\n'
                     '                raise ValueError("MC backup requires scalar target Q and reward-only stochastic policy")\n'
                     '            return single_mc_update(target_actor_state, qf_state, observations, actions,\n'
                     '                next_observations, rewards, dones, gamma, key, mc_backup_samples)\n')
    t = replace_once(t, '        ot_student_action="sample",\n    ):\n        del n_env_interacts',
                     '        ot_student_action="sample",\n        mc_backup_samples=1,\n    ):\n        del n_env_interacts')
    t = replace_once(t, '                backup_mode,\n            )',
                     '                backup_mode,\n                mc_backup_samples,\n            )')
    t = replace_once(t, '            ).reshape(2, batch_size, num_proposals, -1)',
                     '            )\n            source_distributions = source_distributions.reshape(\n'
                     '                source_distributions.shape[0], batch_size, num_proposals, -1)')
    t = replace_once(t, '        core_metrics.update(schedule_metrics)',
                     '        core_metrics.update(schedule_metrics)\n'
                     '        core_metrics.update({"backup_samples", "critic_count", "backup_q_mean",\n'
                     '                             "backup_q_sample_std", "backup_q_mc_se"})')
    p.write_text(t)
    p = out / 'optiq_dime/runtime.py'
    t = p.read_text()
    begin = t.index('    snapshot = ROOT / "DIRECT_GMM_SOURCE.json"')
    end = t.index('    return {', begin)
    t = t[:begin] + ('    source = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())\n'
        '    launch = json.loads((ROOT.parent / "DEPLOYMENT.json").read_text())\n'
        '    git_info = dict(git_commit=launch["commit"], git_dirty=False,\n'
        '                    git_branch="heejoon", source_manifest=source)\n') + t[end:]
    p.write_text(t)
    shutil.copy2(HERE / 'single_mc.py', out / 'optiq_dime/single_mc.py')
    shutil.copy2(HERE / 'config.yaml', out / 'configs/mujoco_direct_gmm_single_mc.yaml')
    # Record actual critic definition in online config, rather than inherited twin metadata.
    p = out / 'run_optiq_dime.py'
    t = p.read_text()
    t = replace_once(t, '    critic = cfg.alg.critic\n',
                     '    critic = cfg.alg.critic\n'
                     '    if critic.get("backup_samples", 1) > 1:\n'
                     '        if not (critic.n_critics == 1 and critic.n_atoms == 1 and backup_mode == "td"\n'
                     '                and actor.distillation_loss == "direct_gmm_nll"\n'
                     '                and critic.backup_samples == actor.num_policy_samples\n'
                     '                and actor.proposals_per_policy_sample == 1):\n'
                     '            raise ValueError("This campaign requires single scalar critic and N=M=K")\n')
    t = replace_once(t, '                    backup_policy="current actor", td_smoothing=False,',
                     '                    backup_policy="current actor", td_smoothing=False,\n'
                     '                    backup_samples=int(cfg.alg.critic.get("backup_samples", 1)),\n'
                     '                    critic_count=int(cfg.alg.critic.n_critics),\n'
                     '                    backup_reduction=("mean over independent policy actions"\n'
                     '                        if cfg.alg.critic.get("backup_samples", 1) > 1 else "twin-min"),')
    p.write_text(t)
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'SOURCE_MANIFEST.json').write_text(json.dumps(dict(base_code_id=source['code_id'],files=files),indent=2)+'\n')
    return files


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print('source_files', len(build(a.base, a.out)))
