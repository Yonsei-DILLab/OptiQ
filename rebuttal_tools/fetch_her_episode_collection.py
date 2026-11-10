"""Collect a Fetch episode before updating; keep native 32-update chunks and UTD."""


def configure_episode_collection(model, total_steps, *, num_envs=256,
                                 rollout_vector_steps=50, updates_per_vector_step=32):
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
    assert total_steps % num_envs == 0
    assert model.n_envs == num_envs
    assert rollout_vector_steps > 0 and updates_per_vector_step > 0
    original_collect = model.collect_rollouts

    def collect(*args, **kwargs):
        remaining = total_steps - model.num_timesteps
        assert remaining > 0 and remaining % num_envs == 0
        vectors = min(rollout_vector_steps, remaining // num_envs)
        kwargs['train_freq'] = TrainFreq(vectors, TrainFrequencyUnit.STEP)
        result = original_collect(*args, **kwargs)
        assert result.episode_timesteps % num_envs == 0
        model.gradient_steps = (result.episode_timesteps // num_envs) * updates_per_vector_step
        return result

    model.collect_rollouts = collect
    model.train_freq = TrainFreq(rollout_vector_steps, TrainFrequencyUnit.STEP)
    model._episode_collection_configuration = dict(
        vector_steps_per_rollout=rollout_vector_steps, num_envs=num_envs,
        updates_per_vector_step=updates_per_vector_step,
        utd=updates_per_vector_step / num_envs,
        final_rollout_capped_to_exact_budget=True)


def run_optimizer_chunks(model, native_train, batch_size, gradient_steps, *, chunk_size=32):
    assert batch_size == 256 and gradient_steps > 0 and gradient_steps % chunk_size == 0
    # Avoid writing the same intermediate checkpoint 50 times at a shared
    # environment step. The outer worker saves once after the complete update.
    has_path = hasattr(model, 'model_save_path')
    path = getattr(model, 'model_save_path', None)
    if has_path:
        model.model_save_path = None
    try:
        for _ in range(gradient_steps // chunk_size):
            native_train(batch_size=batch_size, gradient_steps=chunk_size)
    finally:
        if has_path:
            model.model_save_path = path


def episodic_worker_source(source):
    import ast
    replacements = [
        ('from rebuttal_tools.fetch_her_normalization import HERGoalNormalize, normalized_ibolt_class, save_normalizer',
         'from rebuttal_tools.fetch_her_normalization import HERGoalNormalize, normalized_ibolt_class, save_normalizer\n'
         '    from rebuttal_tools.fetch_her_episode_collection import configure_episode_collection, run_optimizer_chunks'),
        ("    atomic(output/'config.json', settings)",
         "    settings['collection_schedule']='Collect 50 vector steps, then native 32-update chunks; same UTD and exact final budget'\n"
         "    settings['rollout_vector_steps']=50\n"
         "    atomic(output/'config.json', settings)"),
        ('            assert batch_size==256 and gradient_steps==32\n'
         '            original_train(batch_size=batch_size, gradient_steps=gradient_steps)',
         '            run_optimizer_chunks(model, original_train, batch_size, gradient_steps)'),
        ('        model.train=measured_train',
         '        model.train=measured_train\n'
         '        configure_episode_collection(model, total, num_envs=n_envs)'),
        ("exclude=['train'])", "exclude=['train','collect_rollouts'])"),
    ]
    for needle, replacement in replacements:
        if source.count(needle) != 1:
            raise ValueError('Unexpected HER worker version: ' + needle[:80])
        source = source.replace(needle, replacement)
    ast.parse(source)
    return source
