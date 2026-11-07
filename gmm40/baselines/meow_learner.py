"""Official MEow Q/value regression specialized to a terminal one-step bandit."""
from pathlib import Path
import numpy as np
import torch

from gmm40.target import SCALE
from .env import GMM40Bandit
from .meow.policy import FlowPolicy


class MEowBanditLearner:
    def __init__(self, target, seed=0, batch=256, learning_rate=3e-4,
                 warmup=1000, buffer_size=1000000, device='cpu', grad_clip=30.):
        if batch < 1 or warmup < 0 or buffer_size < 1 or grad_clip < 0:
            raise ValueError('Invalid MEow training settings')
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.batch, self.warmup, self.grad_clip = batch, warmup, grad_clip
        self.env = GMM40Bandit(target)
        # Keep the official toy flow architecture, prior bounds, dropout and shifts.
        self.policy = FlowPolicy(1., 1., -2., 2, 1, str(self.device))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        # State=next_state=0 and done=1 for every transition; store only a and r.
        self.actions = np.empty((buffer_size, 2), dtype=np.float32)
        self.rewards = np.empty(buffer_size, dtype=np.float32)
        self.position = self.size = self.updates = 0
        self._evaluation_only = False
        self.settings = dict(alpha=1.,prior_log_std_min=-2.,prior_log_std_max=1.,
                             grad_clip=grad_clip,flow_blocks=2,coupling_hidden_dims=[64,64],
                             shifting_hidden_dims=[256,256],flow_dropout=.1,
                             flow_layer_norm=True,actor_updates_per_optimizer_step=1,
                             checkpoint_supports_training_resume=False)

    def _collect(self, random_action=False):
        if random_action:
            action = np.random.uniform(-1., 1., size=2).astype(np.float32)
        else:
            self.policy.eval()
            with torch.no_grad():
                action, _ = self.policy.sample(1, torch.zeros((1,1), device=self.device))
            action = action[0].cpu().numpy()
        _, reward, terminated, truncated, _ = self.env.step(action)
        assert terminated and not truncated
        self.actions[self.position], self.rewards[self.position] = action, reward
        self.position = (self.position+1) % len(self.actions)
        self.size = min(self.size+1, len(self.actions))

    def advance(self, count):
        if count < 1 or self._evaluation_only:
            raise ValueError('Advance requires a positive count and a fresh training learner')
        if self.updates == 0 and self.env.reward_evaluations == 0:
            for _ in range(self.warmup):
                self._collect(random_action=True)
        losses, norms = [], []
        for _ in range(count):
            self._collect()
            indices = np.random.randint(self.size, size=self.batch)
            actions = torch.as_tensor(self.actions[indices], device=self.device)
            reward = torch.as_tensor(self.rewards[indices], device=self.device)
            obs = torch.zeros((2*self.batch,1), device=self.device)
            self.policy.train()
            prediction, _ = self.policy.get_qv(obs, torch.cat((actions,actions)))
            # Original MEow squared TD error: terminal transitions make target=r.
            target = torch.cat((reward,reward))
            loss = torch.square(prediction.flatten()-target).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MEow regression loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),self.grad_clip if self.grad_clip else float('inf'),
                error_if_nonfinite=True)
            self.optimizer.step()
            self.updates += 1
            losses.append(loss.item()); norms.append(norm.item())
        return dict(loss=float(np.mean(losses)),gradient_norm_before_clip=float(np.mean(norms)),
                    actor_updates=self.updates,critic_updates=self.updates,
                    Q_evaluations=self.env.reward_evaluations,replay_size=self.size)

    def evaluate_samples(self, n, seed):
        was_training = self.policy.training
        devices = []
        if self.device.type == 'cuda':
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(seed)
                self.policy.eval()
                with torch.no_grad():
                    actions,_ = self.policy.sample(n,torch.zeros((n,1),device=self.device))
                return SCALE*actions.cpu().numpy()
        finally:
            self.policy.train(was_training)

    def save(self, path):
        path = Path(path).with_suffix('.pt')
        torch.save(dict(policy=self.policy.state_dict(),optimizer=self.optimizer.state_dict(),
                        actor_updates=self.updates,Q_evaluations=self.env.reward_evaluations,
                        settings=self.settings),path)
        return path

    def load_for_evaluation(self, path):
        state = torch.load(path,map_location=self.device,weights_only=True)
        self.policy.load_state_dict(state['policy'])
        self.updates = int(state['actor_updates'])
        self._evaluation_only = True
