"""Read-only Fetch stage measurements during ordinary policy evaluation."""
import random
import numpy as np


def evaluate(agent, episodes, seed, mode):
    import mujoco
    from fetch_rebuttal.env import make_env
    env = make_env('pick-and-place')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    rows = []
    try:
        with agent.evaluation_rng(seed):
            for episode in range(episodes):
                obs, _ = env.reset(seed=seed + episode)
                initial_z = float(obs[5])
                initial_success = float(np.linalg.norm(obs[3:6] - obs[-3:]) < .05)
                goal_height = float(obs[-1]) - initial_z
                model, data = env.unwrapped.model, env.unwrapped.data
                names = ['object0', 'robot0:r_gripper_finger_link', 'robot0:l_gripper_finger_link']
                obj, right, left = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
                assert min(obj, right, left) >= 0
                total = 0.; any_hit = approached = bilateral = finger = False
                max_lift = 0.; streak = longest = 0
                for length in range(1, 51):
                    action = agent.act(obs[None], mode=mode)[0]
                    if not np.isfinite(action).all():
                        raise FloatingPointError('Nonfinite evaluation action')
                    obs, reward, terminated, truncated, info = env.step(np.clip(action, -1., 1.))
                    touched = set()
                    for contact in data.contact[:data.ncon]:
                        pair = {int(contact.geom1), int(contact.geom2)}
                        if obj in pair:
                            touched.update(pair.intersection({right, left}))
                    finger |= bool(touched); bilateral |= len(touched) == 2
                    approached |= float(np.linalg.norm(obs[:3] - obs[3:6])) < .04
                    lift = float(obs[5]) - initial_z
                    max_lift = max(max_lift, lift)
                    streak = streak + 1 if lift > .05 else 0
                    longest = max(longest, streak)
                    total += float(reward); any_hit |= bool(info['is_success'])
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError('Fetch horizon did not end at 50 control steps')
                rows.append(dict(return_=total, final_success=float(info['is_success']),
                    any_success=float(any_hit), length=length, initial_success=initial_success,
                    goal_height=goal_height, approached_4cm=bool(approached), finger_contact=bool(finger),
                    bilateral_contact=bool(bilateral), max_lift=max_lift, final_lift=lift,
                    longest_lift_steps=longest))
    finally:
        env.close(); random.setstate(python_state); np.random.set_state(numpy_state)
    final = [r['final_success'] for r in rows]; any_hits = [r['any_success'] for r in rows]
    initial = [r['initial_success'] for r in rows]
    strata = {}
    for name, selected in [('all', rows), ('ground', [r for r in rows if abs(r['goal_height']) < .001]),
                           ('low_air', [r for r in rows if .001 <= r['goal_height'] <= .05]),
                           ('air_over5cm', [r for r in rows if r['goal_height'] > .05])]:
        strata[name] = dict(episodes=len(selected),
            final_successes=int(sum(r['final_success'] for r in selected)),
            any_successes=int(sum(r['any_success'] for r in selected)),
            approached_4cm=sum(r['approached_4cm'] for r in selected),
            finger_contact=sum(r['finger_contact'] for r in selected),
            bilateral_contact=sum(r['bilateral_contact'] for r in selected),
            lift_over5cm=sum(r['max_lift'] > .05 for r in selected),
            lift_over10cm=sum(r['max_lift'] > .1 for r in selected),
            sustained_lift_5steps=sum(r['longest_lift_steps'] >= 5 for r in selected))
    return dict(episodes=episodes, success_rate=float(np.mean(final)), any_success_rate=float(np.mean(any_hits)),
        mean_return=float(np.mean([r['return_'] for r in rows])), returns=[r['return_'] for r in rows],
        final_success=final, any_success=any_hits, lengths=[r['length'] for r in rows], initial_success=initial,
        new_final_success_count=int(sum(s and not i for s, i in zip(final, initial))),
        new_any_success_count=int(sum(s and not i for s, i in zip(any_hits, initial))),
        stages=strata, stage_episodes=rows,
        stage_limitations='Finger contact is an endpoint grasp proxy. Lift duration is consecutive control steps.')
