"""Paired-seed physical rollouts without perturbing training RNG/state."""
import json
import numpy as np


def record(learner, task, folder, step):
    import imageio.v2 as imageio
    import mujoco_py
    from .envs import make_one
    from .learners import evaluation_rng
    destination = folder / 'videos'
    destination.mkdir(exist_ok=True)
    path = destination / f'AntMaze_v1_{step:06d}_full_policy.mp4'
    with evaluation_rng(learner, 90500):
        env = make_one(task, 90500, reward_profile=learner.reward_profile)
        try:
            obs = env.reset()
            sim = env.physics_env.sim
            context = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            context.cam.lookat[:] = [0., 4., 0.]
            context.cam.distance = 24.
            context.cam.elevation = -80.
            context.cam.azimuth = 90.
            total = 0.
            xy = [obs[:2].tolist()]
            with imageio.get_writer(str(path), fps=20, codec='libx264') as writer:
                for t in range(500):
                    context.render(640, 640)
                    writer.append_data(context.read_pixels(640, 640, depth=False)[::-1])
                    action = learner.act(obs[None], 'policy')[0]
                    obs, reward, done, info = env.step(action)
                    total += float(reward)
                    xy.append(obs[:2].tolist())
                    if done:
                        break
                context.render(640, 640)
                writer.append_data(context.read_pixels(640, 640, depth=False)[::-1])
            metadata = dict(training_step=step, policy='full stochastic policy',
                            evaluation_seed=90500, length=t+1, episode_return=total,
                            success=int(info.get('success', 0)), xy=xy,
                            included_in_training_interactions=False)
            path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
        finally:
            env.close()
    return path
