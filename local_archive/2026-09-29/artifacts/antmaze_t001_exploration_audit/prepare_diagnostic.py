"""Adapt the existing forward-only diagnostic to saved evaluation start states."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
text = (root / 'artifacts/antmaze_settings_audit/inspect_saved_policy.py').read_text()
start = text.index("config = json.loads((run / 'config.json').read_text())")
end = text.index("repeated = jnp.repeat(obs, N, axis=0)")
text = text[:start] + '''config = json.loads((run / 'config.json').read_text())
available = []
for path in sorted((run / 'policy-checkpoints').glob('*/policy.pt')):
    step = int(path.parent.name.split('_')[1])
    evaluation = run / 'evaluations' / f'{step:010d}' / 'policy-natural' / 'rollouts.npz'
    if evaluation.exists():
        available.append((path, evaluation))
path, evaluation = available[-1]
proof = json.loads((path.parent / 'verification.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256']
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
assert checkpoint['config']['source_commit'] == config['source_commit']
state = checkpoint['learner']
with np.load(evaluation) as raw:
    observations = raw['initial_full_state'].astype(np.float32)
assert observations.shape == (40, 29) and np.isfinite(observations).all()
# Official AntMaze observation is qpos[:15] concatenated with qvel[:14].
obs = jnp.asarray(observations)
result = dict(run=run.name, source=config['source_commit'], checkpoint_step=checkpoint['step'],
              checkpoint_updates=checkpoint['updates'], diagnostic_seed=923, states=len(obs),
              checkpoint_sha256=proof['sha256'],
              state_population='40 recorded random evaluation initial states, not replay states',
              candidates_per_state=64, compute='CPU forward only; no optimizer or environment steps')
del checkpoint
gc.collect()
B, N, A = len(obs), 64, 8
''' + text[end:]
text = text.replace('[0.25,0.05,0.01,0.005,0.001]', '[0.25,0.05,0.01,0.005,0.003,0.001]')
(Path(__file__).parent / 'inspect_eval_start_policy.py').write_text(text)
