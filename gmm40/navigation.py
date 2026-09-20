"""100-step navigation dynamics from the DQS reference, using DiKL's target."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .target import Target


def movement(states,actions):
    actions=np.asarray(actions,dtype=np.float32)
    delta=actions/(np.linalg.norm(actions,axis=-1,keepdims=True)+1e-8)
    return np.clip(np.asarray(states,dtype=np.float32)+delta,-50,50).astype(np.float32)


class GMM40Navigation(gym.Env):
    metadata={"render_modes":["rgb_array"],"render_fps":20}

    def __init__(self,render_mode=None):
        super().__init__()
        self.target=Target()
        self.observation_space=spaces.Box(-50.,50.,shape=(2,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.max_episode_steps=100
        self.render_mode=render_mode
        self.elapsed=0
        self.state=None

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.elapsed=0
        self.state=self.np_random.uniform(-50,50,size=2).astype(np.float32)
        self.path=[self.state.copy()]
        return self.state.copy(),{}

    def step(self,action):
        if self.state is None or self.elapsed>=100: raise RuntimeError("Reset before stepping")
        action=np.asarray(action,dtype=np.float32)
        assert action.shape==(2,) and np.isfinite(action).all()
        self.state=movement(self.state,action)
        self.elapsed+=1
        reward=float(self.target.log_prob(self.state))
        self.path.append(self.state.copy())
        return self.state.copy(),reward,self.elapsed==100,False,{}

    def render(self):
        from .evaluation import background
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,5),constrained_layout=True)
        background(ax,self.target)
        path=np.asarray(self.path)
        ax.plot(*path.T,linewidth=1,color="#dd7932")
        ax.scatter(*path[-1],s=30,c="#dd7932")
        ax.set(xlim=(-50,50),ylim=(-50,50),title=f"GMM40 navigation | step {self.elapsed}/100")
        fig.canvas.draw(); image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy(); plt.close(fig)
        return image


def vector_rollout(action_function,n=1000,seed=20260918):
    """action_function owns an evaluation-only RNG; returned positions are physical."""
    rng=np.random.default_rng(seed)
    state=rng.uniform(-50,50,size=(n,2)).astype(np.float32)
    target=Target()
    positions=[state.copy()]; actions=[]; rewards=[]
    for _ in range(100):
        action=np.asarray(action_function(state),dtype=np.float32)
        state=movement(state,action)
        actions.append(action); positions.append(state.copy()); rewards.append(target.log_prob(state))
    return dict(positions=np.stack(positions),actions=np.stack(actions),rewards=np.stack(rewards))
