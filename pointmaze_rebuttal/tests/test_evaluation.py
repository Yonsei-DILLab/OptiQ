from contextlib import contextmanager
import numpy as np
import pytest
import pointmaze_rebuttal.evaluation as evaluation


@pytest.mark.parametrize('mode', ['policy','mu_only'])
def test_shared_evaluation_retains_native_mode_and_completed_paths(tmp_path,monkeypatch,mode):
    calls=[]
    class Batch:
        def __init__(self,maze,count,seed,obstacle=False):
            self.current=np.zeros((count,4),np.float32)
            self.count,self.t=count,0
            self.lengths=1+np.arange(count)%3
            calls.append((count,seed,obstacle))
        def step(self,actions,active):
            self.t+=1
            self.current[active,0]+=1
            done=active & (self.lengths==self.t)
            goals=np.where(done,np.arange(self.count)%4,-1).astype(np.int8)
            return self.current.copy(),done.astype(np.float32)*100,done,np.zeros(self.count,bool),goals
        def close(self): pass
    class Agent:
        evaluation_mode=mode
        @contextmanager
        def evaluation_rng(self,seed): yield
        def act(self,obs,mode):
            assert mode==self.evaluation_mode
            return np.zeros((len(obs),2),np.float32)
    monkeypatch.setattr(evaluation,'TaskBatch',Batch)
    path=tmp_path/'raw.npz'
    report=evaluation.evaluate(Agent(),'pm_simple',123,130,mode,path)
    assert calls==[(128,123,False),(2,251,False)]
    assert report['success']==1 and sum(report['goals'])==130
    with np.load(path) as data:
        assert str(data['mode'])==mode
        for xy,length in zip(data['xy'],data['lengths']):
            assert np.isfinite(xy[:length+1]).all() and np.isnan(xy[length+1:]).all()
    with pytest.raises(ValueError):
        evaluation.evaluate(Agent(),'pm_simple',123,5,'invalid',tmp_path/'invalid.npz')
