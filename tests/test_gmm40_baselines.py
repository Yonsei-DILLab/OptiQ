from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch

from gmm40.target import Target
from gmm40.baselines.env import GMM40Bandit
from gmm40.baselines.meow_learner import MEowBanditLearner
from gmm40.baselines.td3_learner import TD3BanditLearner


@pytest.fixture(autouse=True)
def single_torch_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_bandit_matches_oracle_and_terminates_without_reference_samples(monkeypatch):
    target = Target()
    monkeypatch.setattr(target,'sample',lambda *a,**kw:pytest.fail('Training queried ground-truth samples'))
    env = GMM40Bandit(target)
    observation,_ = env.reset(seed=3)
    np.testing.assert_array_equal(observation,np.zeros(1,dtype=np.float32))
    for action in [np.array([.1,-.2],dtype=np.float32),np.array([-1.,1.],dtype=np.float32)]:
        state,reward,done,truncated,_ = env.step(action)
        assert reward==float(target.log_prob((40*action)[None])[0])
        assert done and not truncated
        np.testing.assert_array_equal(state,observation)
    assert env.reward_evaluations==2
    with pytest.raises(ValueError):env.step([1.01,0.])


def test_meow_preserves_official_q_value_log_density_identity():
    agent = MEowBanditLearner(Target(),seed=0,batch=4,warmup=4,buffer_size=64)
    agent.policy.eval()
    obs = torch.zeros((20,1))
    actions = torch.linspace(-.8,.8,40).reshape(20,2)
    q,v = agent.policy.get_qv(obs,actions)
    log_prob = agent.policy.log_prob(obs,actions)
    torch.testing.assert_close((q-v).flatten(),log_prob,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(v,agent.policy.get_v(obs))


def test_meow_training_budget_eval_rng_and_checkpoint(tmp_path,monkeypatch):
    target = Target()
    monkeypatch.setattr(target,'sample',lambda *a,**kw:pytest.fail('Training queried ground-truth samples'))
    agent = MEowBanditLearner(target,seed=1,batch=4,warmup=4,buffer_size=64)
    before = {k:v.clone() for k,v in agent.policy.state_dict().items()}
    info = agent.advance(2)
    assert agent.updates==2 and info['Q_evaluations']==6
    assert all(np.isfinite(v) for v in info.values())
    assert any(not torch.equal(before[k],v) for k,v in agent.policy.state_dict().items())
    agent.advance(1)
    torch_rng, numpy_rng = torch.get_rng_state().clone(),np.random.get_state()
    first = agent.evaluate_samples(128,900001)
    np.testing.assert_array_equal(first,agent.evaluate_samples(128,900001))
    torch.testing.assert_close(torch.get_rng_state(),torch_rng,atol=0,rtol=0)
    after = np.random.get_state()
    assert after[0]==numpy_rng[0] and after[2:]==numpy_rng[2:]
    np.testing.assert_array_equal(after[1],numpy_rng[1])
    assert first.shape==(128,2) and np.isfinite(first).all() and np.abs(first).max()<=40
    assert agent.updates==3 and agent.env.reward_evaluations==7
    checkpoint = agent.save(tmp_path/'meow')
    restored = MEowBanditLearner(target,seed=99,batch=4,warmup=4,buffer_size=64)
    restored.load_for_evaluation(checkpoint)
    np.testing.assert_array_equal(first,restored.evaluate_samples(128,900001))


def test_td3_delayed_actor_budget_deterministic_eval_and_checkpoint(tmp_path,monkeypatch):
    target = Target()
    monkeypatch.setattr(target,'sample',lambda *a,**kw:pytest.fail('Training queried ground-truth samples'))
    agent = TD3BanditLearner(target,seed=1,batch=4,warmup=4,buffer_size=64)
    before = {k:v.clone() for k,v in agent.model.actor.state_dict().items()}
    info = agent.advance(2)
    assert info['actor_updates']==2 and info['critic_updates']==4
    assert info['Q_evaluations']==8
    assert any(not torch.equal(before[k],v) for k,v in agent.model.actor.state_dict().items())
    agent.advance(1)
    assert agent.updates==3 and agent.env.reward_evaluations==10
    first = agent.evaluate_samples(128,900001)
    np.testing.assert_array_equal(first,np.repeat(first[:1],128,axis=0))
    np.testing.assert_array_equal(first,agent.evaluate_samples(128,900999))
    assert np.isfinite(first).all() and np.abs(first).max()<=40
    assert agent.env.reward_evaluations==10
    checkpoint = agent.save(tmp_path/'td3')
    restored = TD3BanditLearner(target,seed=99,batch=4,warmup=4,buffer_size=64)
    restored.load_for_evaluation(checkpoint)
    np.testing.assert_array_equal(first,restored.evaluate_samples(128,900001))
