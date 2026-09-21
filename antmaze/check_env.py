"""Contract checks, no learning or hyperparameter search."""
import numpy as np
from .env import make_env


def main():
    a,b=make_env(3),make_env(3)
    x,_=a.reset(seed=7);y,_=b.reset(seed=7)
    assert x.shape==(109,) and x.dtype==np.float32
    np.testing.assert_array_equal(x,y)
    for step in range(1000):
        action=np.zeros(8,np.float32)
        x,r,t,tr,i=a.step(action); y,r2,t2,tr2,i2=b.step(action)
        np.testing.assert_array_equal(x,y)
        assert r in (0.,1.) and r==r2 and t==t2 and tr==tr2
        assert bool(i["success"])==bool(r)
        if t or tr:
            assert t or step==999
            break
    assert tr and not t # no success: timeout must bootstrap
    assert a.unwrapped.continuing_task is False
    a.close();b.close()
    print("AntMaze contract passed: paired seed,109D,float32,sparse reward,timeout bootstrap")


if __name__=="__main__":main()
