"""Matched 256x2 policy/value backbones, with native auxiliary embeddings retained."""
import copy
import numpy as np
import torch
from torch import nn


def linear_shapes(module):
    return [(int(m.in_features), int(m.out_features)) for m in module.modules() if isinstance(m, nn.Linear)]


def reduce_dipo(agent):
    # Called at construction, before any update; preserve the native first two
    # hidden layers and output head, remove only the third hidden layer/Mish.
    def two_hidden(sequence):
        assert len(sequence) == 7 and [m.out_features for m in sequence if isinstance(m, nn.Linear)][:-1] == [256]*3
        return nn.Sequential(*list(sequence.children())[:4], list(sequence.children())[-1])
    agent.actor.model.layer = two_hidden(agent.actor.model.layer)
    agent.critic.q1_model = two_hidden(agent.critic.q1_model)
    agent.critic.q2_model = two_hidden(agent.critic.q2_model)
    agent.actor_target = copy.deepcopy(agent.actor)
    agent.critic_target = copy.deepcopy(agent.critic)
    # Native optimizers must bind exactly the new networks, without removed tensors.
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4, eps=1e-5)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=3e-4, eps=1e-5)


def widen_meow(policy, observation_dim, action_dim):
    from cleanrl.nf.nets import MLP
    from cleanrl.nf.flows import MaskedCondAffineFlow
    policy.prior.log_scale = MLP([observation_dim, 256, 256, action_dim], init='zero').to(policy.device)
    for flow in policy.flows:
        if isinstance(flow, MaskedCondAffineFlow):
            assert flow.s is None
            flow.t = MLP([observation_dim+action_dim, 256, 256, action_dim], init='orthogonal',
                         dropout_rate=.1, layernorm=True).to(policy.device)


def _dense_widths(params):
    keys = sorted((k for k, v in params.items() if k.startswith('Dense_') and 'kernel' in v),
                  key=lambda k: int(k.split('_')[-1]))
    return [int(params[k]['kernel'].shape[-1]) for k in keys]


def audit(agent, method):
    if method == 'dipo':
        native = agent.native
        actor = linear_shapes(native.actor.model.layer)
        critics = [linear_shapes(native.critic.q1_model), linear_shapes(native.critic.q2_model)]
        assert [out for _, out in actor[:-1]] == [256, 256]
        assert all([out for _, out in rows[:-1]] == [256, 256] for rows in critics)
        assert linear_shapes(native.actor_target.model.layer) == actor
        assert linear_shapes(native.critic_target.q1_model) == critics[0]
        assert linear_shapes(native.critic_target.q2_model) == critics[1]
        assert {id(p) for g in native.actor_optimizer.param_groups for p in g['params']} == {id(p) for p in native.actor.parameters()}
        assert {id(p) for g in native.critic_optimizer.param_groups for p in g['params']} == {id(p) for p in native.critic.parameters()}
        return dict(actor_hidden=[256, 256], critic_hidden=[256, 256], actor_linear=actor,
                    critic_linear=critics, auxiliary_time_embedding_linear=linear_shapes(native.actor.model.time_mlp))
    if method == 'meow':
        # Every flow/prior MLP and both value-shifting heads are checked, not
        # merely the constructor's descriptive configuration.
        networks = {}
        for name, module in agent.policy.named_modules():
            if module.__class__.__name__ == 'MLP':
                shapes = linear_shapes(module)
                assert [out for _, out in shapes[:-1]] == [256, 256], (name, shapes)
                networks[name] = shapes
        assert len(networks) == 7, networks
        targets = {name: linear_shapes(module) for name, module in agent.target.named_modules()
                   if module.__class__.__name__ == 'MLP'}
        assert targets == networks
        return dict(actor_hidden=[256, 256], critic_hidden=[256, 256], mlps=networks,
                    critic_representation='Native flow-derived Q with two 256x2 value-shifting heads')
    native = agent.agent
    actor = _dense_widths(native.actor.params['MLP_0'])
    logp = _dense_widths(native.logp_mvel.params['MLP_0'])
    critics = [_dense_widths(state.params['MLP_0']) for state in
               (native.critic_1, native.critic_2, native.target_critic_1, native.target_critic_2)]
    assert actor[:-1] == [256, 256] and logp[:-1] == [256, 256]
    assert all(row == [256, 256] for row in critics)
    embeddings = {name: _dense_widths(p['MLP_0']) for name, p in native.actor.params.items()
                  if name.startswith('TimestepEmbedder_')}
    return dict(actor_hidden=[256, 256], critic_hidden=[256, 256], actor_dense_outputs=actor,
                log_density_dense_outputs=logp, critic_dense_outputs=critics,
                auxiliary_time_embedding_dense_outputs=embeddings)
