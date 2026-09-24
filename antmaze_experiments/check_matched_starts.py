"""Real MuJoCo checks for training-matched evaluation reset distributions."""
import json
import numpy as np
from .envs import make_one


def main():
 results={}
 for task in ('v1','v2','v3','v4'):
  train=make_one(task,87231,reward_profile='dense')
  evaluation=make_one(task,87231,fixed=task!='v1',reward_profile='dense',random_init=task=='v1')
  try:
   a=[];b=[]
   for _ in range(40):
    train.reset();s=train.state();a.append(np.r_[s['qpos'],s['qvel']])
    evaluation.reset();s=evaluation.state();b.append(np.r_[s['qpos'],s['qvel']])
   a=np.array(a);b=np.array(b);np.testing.assert_array_equal(a,b)
   unique=len(np.unique(b,axis=0));assert unique==(40 if task=='v1' else 1)
   if task!='v1':np.testing.assert_array_equal(b[:,:2],np.zeros((40,2)))
   else:assert np.all(np.abs(b[:,:2])<=2)
   results[task]=dict(passed=True,unique_full_states=unique,matches_training=True,xy_first=b[0,:2].tolist())
  finally:train.close();evaluation.close()
 print(json.dumps(results,indent=2))


if __name__=='__main__':main()
