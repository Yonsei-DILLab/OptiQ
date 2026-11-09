import hashlib
import shutil
import numpy as np
import pytest
import torch
from pointmaze_rebuttal.agents.dipo import Agent, VENDOR, upstream, vector_memories, verify_vendor
from pointmaze_rebuttal.protocol import configuration


def test_upstream_hash_and_license_guard(tmp_path):
    manifest = verify_vendor()
    assert manifest['revision'] == 'c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc'
    assert 'LICENSE' in manifest['files']
    source = tmp_path/'dipo'
    shutil.copytree(VENDOR,source)
    with (source/'agent/replay_memory.py').open('a') as stream:
        stream.write('\n# accidental change\n')
    with pytest.raises(RuntimeError,match='source changed'):
        verify_vendor(source)


def test_vector_replay_matches_scalar_upstream_and_preserves_replace():
    _, NativeReplay, NativeDiffusion = upstream()
    VectorReplay, VectorDiffusion = vector_memories()
    assert VectorDiffusion.replace is NativeDiffusion.replace
    for Vector, Native, names in (
        (VectorReplay, NativeReplay, ('states','actions','rewards','next_states','masks')),
        (VectorDiffusion, NativeDiffusion, ('states','best_actions'))):
        vector, scalar = Vector(4,2,8,'cpu'), Native(4,2,8,'cpu')
        for offset in (0,6):
            states = np.arange(24,dtype=np.float32).reshape(6,4)+offset
            actions = np.arange(12,dtype=np.float32).reshape(6,2)/30
            args = (states,actions) if Native is NativeDiffusion else (states,actions,np.ones(6),states+1,np.ones(6)*.99)
            vector.append(*args)
            for row in zip(*args):
                scalar.append(*row)
        assert (vector.idx,vector.full) == (scalar.idx,scalar.full)
        for name in names:
            np.testing.assert_array_equal(getattr(vector,name),getattr(scalar,name))
        if Native is NativeDiffusion:
            before=vector.best_actions.copy()
            for memory in (vector,scalar):
                np.random.seed(4)
                _,sampled,indices=memory.sample(16)
                with torch.no_grad(): sampled.add_(.25)
                memory.replace(indices,sampled.detach().numpy())
                np.testing.assert_array_equal(memory.best_actions,before)


def test_real_dipo_update_native_evaluation_rng_and_checkpoint(tmp_path):
    config=configuration('dipo','simple',seed=0,smoke=True)
    agent=Agent(config,tmp_path,device='cpu')
    assert agent.config['diffusion_steps']==100 and agent.config['action_gradient_steps']==20
    assert agent.config['replay_replace_is_upstream']
    rng=np.random.default_rng(1)
    obs=rng.normal(size=(4,4)).astype(np.float32)
    actions=rng.uniform(-1,1,size=(4,2)).astype(np.float32)
    agent.observe(obs,actions,np.ones(4),obs+.1,np.array([False,True,False,True]))
    np.testing.assert_allclose(agent.memory.masks[:4,0],[.99,0,.99,0])
    before={name:[p.detach().clone() for p in getattr(agent.agent,name).parameters()]
            for name in ('actor','critic')}
    agent.update(6)
    assert agent.updates==agent.agent.step==1
    for name in before:
        assert any(not torch.equal(a,b) for a,b in zip(before[name],getattr(agent.agent,name).parameters()))
    agent.assert_finite()
    state=torch.random.get_rng_state().clone()
    with agent.evaluation_rng(12):
        first=agent.act(obs,mode='policy')
    assert torch.equal(state,torch.random.get_rng_state())
    with agent.evaluation_rng(12):
        np.testing.assert_array_equal(first,agent.act(obs,mode='policy'))
    assert np.isfinite(first).all() and np.max(np.abs(first))<=1
    with pytest.raises(ValueError): agent.act(obs,mode='mu_only')
    path=tmp_path/'checkpoint.pt'
    agent.save(path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    assert restored['updates']==1
    for name,value in agent.agent.actor.state_dict().items():
        torch.testing.assert_close(restored['actor'][name],value,rtol=0,atol=0)
