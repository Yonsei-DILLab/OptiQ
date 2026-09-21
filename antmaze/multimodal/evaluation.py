"""Repeated rollouts of ONE learned policy, saved without seed pooling."""
from pathlib import Path
import numpy as np
from antmaze.evaluation import isolated_rng,atomic_json
from .env import make_env
from .policy import PolicyView
from .analysis import summarize


class Evaluator:
    def __init__(self,folder,task,seed,batch=10):
        self.folder=Path(folder);self.task=task;self.seed=seed;self.batch=batch
        self.envs=[make_env(task) for _ in range(batch)];self.histories={}
    def evaluate(self,agent,step,episodes=10,fixed=False,mode="policy",archive=False):
        assert episodes%self.batch==0
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,7406]))
        seeds=rng.integers(0,2**30,episodes)
        policy_seeds=rng.integers(0,2**30,episodes//self.batch)
        horizon=self.envs[0].horizon
        paths=np.full((episodes,horizon+1,2),np.nan,np.float32)
        initial=np.zeros((episodes,29),np.float32)
        returns=np.zeros(episodes);lengths=np.zeros(episodes,int);goals=np.zeros(episodes,int)
        min_distance=np.zeros(episodes);final_distance=np.zeros(episodes)
        policy=PolicyView(agent);initial_simulator=[]
        for start in range(0,episodes,self.batch):
            obs=[]
            for i,e in enumerate(self.envs):
                o,info=e.reset(seed=int(seeds[start+i]),options={"fixed_start":fixed})
                obs.append(o);initial[start+i]=o;paths[start+i,0]=info["xy"]
                initial_simulator.append(e.simulator_state())
                min_distance[start+i]=final_distance[start+i]=info["distance"]
            obs=np.stack(obs);done=np.zeros(self.batch,bool)
            ps=int(policy_seeds[start//self.batch])
            with isolated_rng(ps),policy.evaluation(mode,ps):
                while not done.all():
                    actions=policy.act(obs)
                    if actions.shape!=(self.batch,8) or not np.isfinite(actions).all():raise FloatingPointError("Invalid policy draw")
                    for i,e in enumerate(self.envs):
                        if done[i]:continue
                        ix=start+i
                        obs[i],r,t,tr,info=e.step(actions[i])
                        lengths[ix]+=1;returns[ix]+=r;paths[ix,lengths[ix]]=info["xy"]
                        goals[ix]=info["goal_id"];final_distance[ix]=info["distance"]
                        min_distance[ix]=min(min_distance[ix],info["distance"])
                        done[i]=t or tr
        initial_simulator=np.asarray(initial_simulator)
        if fixed:
            np.testing.assert_array_equal(initial_simulator,np.broadcast_to(initial_simulator[0],initial_simulator.shape))
        summary=summarize(self.task,paths,lengths,goals,returns)
        summary.update(step=step,task=self.task,training_seed=self.seed,evaluation_mode=mode,
                       reset_mode="identical_full_state" if fixed else "training_reset_distribution",
                       mean_min_distance=float(min_distance.mean()),mean_final_distance=float(final_distance.mean()))
        label=f"{mode}-"+("fixed" if fixed else "natural")
        if archive:
            dest=self.folder/"rollouts";dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f"{step}-{label}.npz",xy=paths,initial_state=initial,initial_simulator_state=initial_simulator,returns=returns,
                lengths=lengths,goal_ids=goals,env_seeds=seeds,policy_batch_seeds=policy_seeds,
                final_distance=final_distance,min_distance=min_distance)
            atomic_json(dest/f"{step}-{label}.json",summary)
        if not archive:
            h=self.histories.setdefault(label,[]);h.append(summary)
            atomic_json(self.folder/f"history-{label}.json",h)
        return summary
    def close(self):
        for e in self.envs:e.close()
