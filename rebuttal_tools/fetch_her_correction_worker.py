"""Run Fetch HER protocol variants around the frozen learner worker.

The legacy worker stays unchanged on disk. This entry point also supports the
existing entropy-parameter audit without changing any learner implementations.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types


def corrected_worker_source(source, *, retain_full_training_replay=False, ibolt_gamma=None):
    replacements = [
        ('from fetch_rebuttal.normalization import normalized_ibolt_class, save_normalizer',
         'from rebuttal_tools.fetch_her_normalization import HERGoalNormalize, normalized_ibolt_class, save_normalizer'),
        ("vec = VecNormalize(GroupedFetchVecEnv(n_envs, spec['worker_cpus'], task=task), norm_obs=True,",
         "vec = HERGoalNormalize(GroupedFetchVecEnv(n_envs, spec['worker_cpus'], task=task), task=task, stats_seed=seed, norm_obs=True,"),
        ("    atomic(output/'config.json', settings)",
         "    settings['normalization_recipe']='Completed-episode HER statistics; std floor0.01; raw clip200; normalized clip5; reward unchanged'\n"
         "    atomic(output/'config.json', settings)"),
        ('her_replay_last_sample=model.replay_buffer.last))',
         'her_replay_last_sample=model.replay_buffer.last,\n'
         '                    her_normalization_stats=dict(transitions=vec.stats_transitions,completed_episodes=vec.completed_stats_episodes,last=vec.last_stats_audit)))'),
        ("        atomic(output/'verified-completion.json',result)",
         "        assert vec.stats_transitions>0 and vec.last_stats_audit['relabeled_goals']>0 and vec.last_stats_audit['effective_std_min']>=.01\n"
         "        result['normalization_recipe']=settings['normalization_recipe']\n"
         "        atomic(output/'verified-completion.json',result)"),
    ]
    if retain_full_training_replay:
        replacements.append(
            ('replay_capacity=25600 if smoke else 1000000,',
             'replay_capacity=25600 if smoke else total,'))
    if ibolt_gamma is not None:
        gamma = float(ibolt_gamma)
        if not 0 < gamma < 1:
            raise ValueError('iBOLT gamma must be between zero and one')
        replacements.extend([
            ('            cfg.total_steps=total; cfg.output_root=str(output); cfg.require_gpu=True',
             '            cfg.total_steps=total; cfg.output_root=str(output); cfg.require_gpu=True\n'
             f'            cfg.alg.gamma = {gamma!r}'),
            ("'alg.utd','alg.buffer_size','env_name'",
             "'alg.utd','alg.buffer_size','alg.gamma','env_name'"),
            ('            view=IBOLTView(model)',
             '            assert model.gamma == cfg.alg.gamma\n'
             '            view=IBOLTView(model)'),
        ])
    for needle, replacement in replacements:
        if source.count(needle) != 1:
            raise ValueError('Unexpected frozen worker version: ' + needle[:80])
        source = source.replace(needle, replacement)
    ast.parse(source)
    return source


def install_corrected_worker(spec):
    assert spec['method'] in ('ibolt', 'sac')
    gamma = spec.get('ibolt_gamma')
    if gamma is not None and spec['method'] != 'ibolt':
        raise ValueError('ibolt_gamma applies to iBOLT runs')
    root = Path(spec['source']).resolve()
    assert Path(__file__).resolve().parent.parent == root, 'Launch from the frozen source snapshot'
    source_path = root / 'rebuttal_tools/fetch_suite_5m_worker.py'
    manifest = json.loads(Path(spec['source_manifest']).read_text())
    for relative in ('rebuttal_tools/fetch_her_correction_worker.py',
                     'rebuttal_tools/fetch_her_normalization.py'):
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == manifest[relative]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest[str(source_path.relative_to(root))]
    source = corrected_worker_source(
        source_path.read_text(),
        retain_full_training_replay=spec.get('retain_full_training_replay', False),
        ibolt_gamma=gamma)
    if spec.get('collection_schedule') == 'episode_frozen':
        relative = 'rebuttal_tools/fetch_her_episode_collection.py'
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == manifest[relative]
        from rebuttal_tools.fetch_her_episode_collection import episodic_worker_source
        source = episodic_worker_source(source)
    name = 'rebuttal_tools.fetch_suite_5m_worker'
    assert name not in sys.modules, 'Use a fresh process for each corrected run'
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    module.__package__ = 'rebuttal_tools'
    sys.modules[name] = module
    exec(compile(source, str(source_path) + ':her-normalization-adapter', 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    sys.path.insert(0, str(Path(spec['source']).resolve()))
    module = install_corrected_worker(spec)
    original_atomic = module.atomic
    common_changes = []
    recipe_changes = []
    if spec.get('retain_full_training_replay', False):
        common_changes.append('replay_capacity')
        recipe_changes.append('full training replay retention')
    if spec.get('ibolt_gamma') is not None:
        common_changes.append('gamma')
        recipe_changes.append('iBOLT gamma ' + str(spec['ibolt_gamma']))

    def record_recipe(path, data):
        if path.name in ('config.json', 'verified-completion.json', 'native-config.json', 'sac-config.json'):
            data = dict(data, parameter_selection='HER normalization correction; method parameter value recorded in entropy_search')
            if common_changes:
                data['parameter_selection'] = 'HER normalization with ' + ' and '.join(recipe_changes)
                data['entropy_search'] = dict(
                    data['entropy_search'], common_settings_unchanged=False,
                    changed_common_setting=', '.join(common_changes))
        original_atomic(path, data)

    module.atomic = record_recipe
    from rebuttal_tools.fetch_entropy_worker import install
    install(spec).run(spec, args.smoke)


if __name__ == '__main__':
    main()
