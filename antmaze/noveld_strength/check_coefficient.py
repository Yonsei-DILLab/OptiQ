"""Exercise actual replay-time bonuses while holding RND data/updates fixed."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from antmaze.multimodal.noveld import NovelD, Replay
from antmaze.multimodal.resume import digest


def check():
    torch.set_num_threads(2)
    rng=np.random.default_rng(917)
    obs=rng.normal(size=(256,29)).astype(np.float32)
    nxt=rng.normal(size=(256,29)).astype(np.float32)
    rewards=-np.linalg.norm(nxt[:,:2],axis=1)
    records=[];reference=None
    for coefficient in (.01,.1,1.,5.,10.):
        n=NovelD(29,0,coefficient)
        initial=(digest(n.predictor.state_dict()),digest(n.target.state_dict()))
        replay=Replay(256,0,n,dictionary=True)
        replay.add_batch(obs,np.zeros((256,8),np.float32),rewards,nxt,np.zeros(256,np.float32))
        ix=np.random.default_rng(1729).integers(256,size=256)
        batch=replay.sample(256)
        bonus=batch['rewards']-rewards[ix]
        final=(digest(n.predictor.state_dict()),digest(n.target.state_dict()))
        if reference is None:reference=(initial,final,bonus/.01)
        else:
            assert initial==reference[0] and final==reference[1]
            np.testing.assert_allclose(bonus,reference[2]*coefficient,rtol=3e-5,atol=3e-5)
        np.testing.assert_array_equal(replay.data['rewards'],rewards)
        replay.diagnostic=True;replay.sample(256)
        assert n.updates==1 and digest(n.predictor.state_dict())==final[0]
        records.append(dict(coefficient=coefficient,bonus_mean=float(bonus.mean()),initial=initial,final=final))
    old=Path('/home/heechan/OptiQ-ops/sources/19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5/antmaze/multimodal/noveld.py')
    spec=importlib.util.spec_from_file_location('original_noveld',old)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_n=module.NovelD(29,0);new_n=NovelD(29,0)
    np.testing.assert_array_equal(old_n.reward_and_update(obs,nxt,rewards),new_n.reward_and_update(obs,nxt,rewards))
    assert digest(old_n.predictor.state_dict())==digest(new_n.predictor.state_dict())
    assert old_n.config==new_n.config
    return dict(passed=True,default_preserved=True,replay_bonus_verified=True,records=records)
