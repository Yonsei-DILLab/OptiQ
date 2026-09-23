import hashlib,json
from pathlib import Path
import flax.serialization
import jax,numpy as np
from scipy.integrate import quad
from .core import Experiment,CENTERS,WIDTH,q_value
from .evaluate import cdf,target_logf,separation,Z,REFERENCE_BACKUP
from ..kl_forward_wide_1d.core import Experiment as WidthOne

def main():
    here=Path(__file__).parent;old=here.parent/'kl_forward_wide_1d'
    c=json.loads((here/'config.json').read_text());prior=json.loads((old/'config.json').read_text())
    diff={k for k in set(c)|set(prior) if c.get(k)!=prior.get(k)}
    assert diff=={'study','target_width'} and c['target_width']==.5
    normalized=(here/'core.py').read_text().replace('target width0.5','target width1').replace('WIDTH = .5','WIDTH = 1.').replace('from ..kl_forward_wide_1d.actor import','from .actor import').replace('from ..kl_forward_wide_1d.box_gaussian import','from .box_gaussian import').replace('from ..kl_forward_wide_1d.distillation import','from .distillation import')
    assert normalized==(old/'core.py').read_text()
    paired=[]
    for seed in c['seeds']:
        e=Experiment(c,'forward',0,seed);p=WidthOne(prior,'forward',0,seed)
        b=flax.serialization.to_bytes({'state':e.state,'key':e.key});b0=flax.serialization.to_bytes({'state':p.state,'key':p.key});assert b==b0
        assert all(np.array_equal(x,y) for x,y in zip(e.samples(1024),p.samples(1024)))
        paired.append(dict(seed=seed,initial_state_sha256=hashlib.sha256(b).hexdigest(),initial_samples_identical=True))
    assert CENTERS==(-5.,0.,5.) and WIDTH==.5
    grid=np.linspace(-10,10,40001);density=np.exp(target_logf(grid))/Z
    peaks=grid[np.where((density[1:-1]>density[:-2])&(density[1:-1]>density[2:]))[0]+1]
    assert len(peaks)==3 and np.allclose(peaks,CENTERS,atol=.001)
    integral=quad(lambda x:float(np.exp(target_logf(x))/Z),-10,10,points=list(CENTERS),epsabs=1e-11)[0]
    assert abs(integral-1)<1e-10 and abs(cdf(10)-cdf(-10)-1)<1e-12
    actions=np.interp((np.arange(32768)+.5)/32768,cdf(grid),grid)
    assert separation(actions)['three_peak_pass'] and not separation(np.linspace(-10,10,32768))['three_peak_pass']
    assert np.allclose(np.asarray(q_value(jax.numpy.asarray(grid[:,None]))),.25*target_logf(grid),atol=2e-5)
    (loss,_),grad=jax.value_and_grad(e.group_loss,has_aux=True)(e.state.params,jax.random.PRNGKey(87))
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(grad))
    print(json.dumps(dict(passed=True,config_changes=sorted(diff),matched_initializations=paired,target_width=WIDTH,peak_positions=peaks.tolist(),target_integral=integral,target_diagnostic=separation(actions),reference_backup=REFERENCE_BACKUP,loss=float(loss),core_unchanged_except_width_and_imports=True),indent=2))

if __name__=='__main__':main()
