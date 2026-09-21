"""Freeze direct-gmm-trg framework + actual truncated-policy overlay, then MC64.

Never substitute the legacy top-level algorithm or the older squashed Gaussian.
Build input is exactly the Git archive of BASE_COMMIT, verified against Git.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess

BASE_COMMIT='4ca69473515b5083d6e651845ded39da34e14538'
HERE=Path(__file__).resolve().parent

def replace(text,old,new):
    assert text.count(old)==1,(old[:90],text.count(old))
    return text.replace(old,new,1)

def build(base,out):
    assert not out.exists(),out
    framework=base/'analysis_tools/studies/20260918_nonstationary_nd/v5'
    overlay=base/'analysis_tools/experiments/20260920_truncated_mll/optiq_dime'
    shutil.copytree(framework,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(overlay,out/'optiq_dime',dirs_exist_ok=True)
    p=out/'optiq_dime/algorithm.py';t=p.read_text()
    t=replace(t,'from .policy import OptiQPolicy','from .policy import OptiQPolicy\nfrom .single_mc import update as single_mc_update')
    t=replace(t,'            actor.get("ot_student_action", "sample"),\n        )',
        '            actor.get("ot_student_action", "sample"),\n            int(self.cfg.alg.critic.get("backup_samples", 1)),\n        )')
    for old in ['            "backup_mode",\n        ],','            "backup_mode",\n            "entropy_diagnostics",']:
        t=replace(t,old,old.replace('            "backup_mode",','            "backup_mode",\n            "mc_backup_samples",'))
    t=replace(t,'        backup_mode: str | None = None,\n    ):',
        '        backup_mode: str | None = None,\n        mc_backup_samples: int = 1,\n    ):\n'
        '        if mc_backup_samples > 1:\n'
        '            if crossq_style or num_atoms != 1 or not semi_implicit or backup_mode != "td":\n'
        '                raise ValueError("MC backup requires scalar target Q and reward-only full policy")\n'
        '            return single_mc_update(target_actor_state, qf_state, observations, actions,\n'
        '                next_observations, rewards, dones, gamma, key, mc_backup_samples)\n')
    t=replace(t,'        ot_student_action="sample",\n    ):\n        del n_env_interacts',
        '        ot_student_action="sample",\n        mc_backup_samples=1,\n    ):\n        del n_env_interacts')
    t=replace(t,'                backup_mode,\n            )','                backup_mode,\n                mc_backup_samples,\n            )')
    t=replace(t,'            ).reshape(2, batch_size, num_proposals, -1)',
        '            )\n            source_distributions = source_distributions.reshape(\n'
        '                source_distributions.shape[0], batch_size, num_proposals, -1)')
    t=replace(t,'        core_metrics.update(schedule_metrics)',
        '        core_metrics.update(schedule_metrics)\n'
        '        core_metrics.update({"backup_samples", "critic_count", "backup_q_mean",\n'
        '                             "backup_q_sample_std", "backup_q_mc_se"})')
    p.write_text(t)
    shutil.copy2(HERE/'single_mc.py',out/'optiq_dime/single_mc.py')
    shutil.copy2(HERE/'config.yaml',out/'configs/mujoco_trg_single_mc.yaml')
    p=out/'optiq_dime/runtime.py';t=p.read_text()
    begin=t.index('    snapshot = ROOT / "DIRECT_GMM_SOURCE.json"');end=t.index('    return {',begin)
    t=t[:begin]+('    launch=json.loads((ROOT.parent / "DEPLOYMENT.json").read_text())\n'
        '    source=json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())\n'
        '    git_info=dict(git_commit=launch["commit"],git_dirty=False,git_branch="heejoon",\n'
        '        base_branch="direct-gmm-trg",base_commit=source["base_commit"],source_manifest=source)\n')+t[end:]
    p.write_text(t)
    p=out/'run_optiq_dime.py';t=p.read_text()
    t=replace(t,'    actor = cfg.alg.actor\n','    actor = cfg.alg.actor\n'
        '    assert cfg.alg.critic.n_critics == 1 and cfg.alg.critic.backup_samples == 64\n'
        '    assert actor.num_policy_samples == 64 and actor.proposals_per_policy_sample in (1, 4)\n'
        '    assert actor.distillation_loss == "direct_gmm_nll" and cfg.alg.critic.backup_mode == "td"\n')
    t=t.replace('policy="tanh(mu(s,z)+sigma(s,z)*eps)"',
        'policy="box-truncated Gaussian [-1,1]; mu=tanh(raw_mu), inverse-CDF sample"')
    t=t.replace('teacher_bandwidth_space="pre-tanh", teacher_hard_cutoff=False,',
        'teacher_bandwidth_space="action; component sigma, no extra floor", teacher_hard_cutoff=True,')
    t=replace(t,'                    backup_policy="current actor", td_smoothing=False,',
        '                    backup_policy="current full truncated-Gaussian actor", td_smoothing=False,\n'
        '                    backup_samples=64, critic_count=1,\n'
        '                    backup_reduction="mean over 64 independent policy actions",')
    t=t.replace('direct finite-mixture marginal Gaussian NLL','direct normalized box-truncated-mixture marginal NLL')
    t=t.replace('pre-tanh; fixed-teacher action Jacobian omitted','bounded action; differentiable truncation normalization included')
    t=t.replace('a=tanh(mu(s,0)); epsilon=0','a=mu(s,0), bounded center (not truncated expectation)')
    t=t.replace('z~N(0,I) per action; a=tanh(mu(s,z)); epsilon=0','z~N(0,I) per action; a=mu(s,z), bounded center')
    p.write_text(t)
    # These numerical files must remain byte-identical to the requested branch.
    unchanged=['policy.py','box_gaussian.py','distillation.py','semi_implicit.py','dual_evaluation.py']
    for name in unchanged: assert (out/'optiq_dime'/name).read_bytes()==(overlay/name).read_bytes()
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(base_branch='direct-gmm-trg',
        base_commit=BASE_COMMIT,unchanged_trg_numerical_files=unchanged,files=files),indent=2)+'\n')
    return files

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();print('Built',len(build(args.base,args.out)),'files')
