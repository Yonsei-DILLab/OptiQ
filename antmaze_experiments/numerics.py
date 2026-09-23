"""Numerical guards for the vendored DIPO learner; upstream files stay intact."""
import torch
import torch.nn.functional as F


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} contains NaN or infinity')


def bounded_probability(value, name, interior=False):
    """Only tolerate roundoff, never replace invalid distributions with numbers."""
    _finite(value, name)
    tolerance = 32 * torch.finfo(value.dtype).eps
    if bool(((value < -tolerance) | (value > 1+tolerance)).any()):
        raise FloatingPointError(f'{name} is outside probability bounds beyond roundoff')
    epsilon = torch.finfo(value.dtype).eps if interior else 0.
    return value.clamp(epsilon, 1-epsilon)


@torch.no_grad()
def stable_projection(next_dist, reward, done, gamma, v_min=-10, v_max=10,
                      num_atoms=51, support=None, device=None):
    """C51 projection with double accumulation, normalization and bounded output."""
    assert support is not None and next_dist.shape[-1] == num_atoms and v_max > v_min
    probabilities = bounded_probability(next_dist, 'next distribution').to(torch.float64)
    _finite(reward, 'reward'); _finite(done, 'done'); _finite(support, 'support')
    mass = probabilities.sum(-1, keepdim=True)
    if bool((mass <= 0).any()):
        raise FloatingPointError('next distribution has zero mass')
    probabilities = probabilities / mass
    delta = (v_max-v_min)/(num_atoms-1)
    target = (reward.double()+(1-done.double())*gamma*support.double()).clamp(v_min,v_max)
    coordinate = ((target-v_min)/delta).clamp(0,num_atoms-1)
    lower, upper = coordinate.floor().long(), coordinate.ceil().long()
    exact = (lower == upper).to(probabilities.dtype)
    projected = torch.zeros_like(probabilities)
    projected.scatter_add_(1, lower, probabilities*(upper-coordinate+exact))
    projected.scatter_add_(1, upper, probabilities*(coordinate-lower))
    projected /= projected.sum(-1,keepdim=True)
    # The final cast can round a unit mass upwards; cap only after normalization.
    projected = projected.to(next_dist.dtype).clamp(0,1)
    _finite(projected, 'projected distribution')
    return projected


def guarded_binary_cross_entropy(prediction, target):
    prediction = bounded_probability(prediction, 'critic probability', interior=True)
    target = bounded_probability(target, 'projected target')
    return F.binary_cross_entropy(prediction, target)


def guarded_optimizer_step(optimizer, objective, max_grad_norm):
    """Keep native clipping; fail before an invalid gradient can update weights."""
    _finite(objective, 'optimizer loss')
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    norm = torch.nn.utils.clip_grad_norm_(parameters,
        max_grad_norm if max_grad_norm is not None else float('inf'),
        error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
        raise FloatingPointError('optimizer produced a nonfinite parameter')
    return norm


def stable_dipo_class():
    from ddiffpg.algo.dipo import AgentDIPO

    class StableDIPO(AgentDIPO):
        def optimizer_update(self, optimizer, objective):
            return guarded_optimizer_step(optimizer, objective, self.cfg.algo.max_grad_norm)

        def update_critic(self, obs, action, reward, next_obs, done):
            next_actions = self.get_tgt_policy_actions(next_obs)
            with torch.no_grad():
                distributions = self.critic_target.get_q1_q2(next_obs,next_actions)
                projected = [stable_projection(d,reward,done,
                    self.cfg.algo.gamma**self.cfg.algo.nstep,
                    self.cfg.algo.v_min,self.cfg.algo.v_max,self.cfg.algo.num_atoms,
                    self.critic.z_atoms) for d in distributions]
                # Preserve the upstream elementwise minimum and BCE objective.
                target = torch.minimum(*projected)
            current = self.critic.get_q1_q2(obs,action)
            loss = sum(guarded_binary_cross_entropy(p,target) for p in current)
            grad_norm = self.optimizer_update(self.critic_optimizer,loss)
            self.projection_diagnostics = dict(
                projection_target_min=float(target.min()),
                projection_target_max=float(target.max()),
                projection_mass_error=max(float((p.sum(-1)-1).abs().max()) for p in projected),
                critic_lower_atom_mass=float(torch.stack([p[:,0].mean() for p in current]).mean()),
                critic_upper_atom_mass=float(torch.stack([p[:,-1].mean() for p in current]).mean()))
            return loss.item(),grad_norm.item()
    return StableDIPO


class DisabledIntrinsic:
    """Native update loops can call this interface without any RND computation."""
    enabled = False
    update_step = 0

    def compute_reward(self, obs, next_obs=None):
        return obs.new_zeros((len(obs),1))

    def update(self, obs):
        return 0.,0.
