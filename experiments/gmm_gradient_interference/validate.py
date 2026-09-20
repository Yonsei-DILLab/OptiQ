from .core import *
from .diagnostics import probe
from .run import verify,write
from flax import serialization
from pathlib import Path
import argparse,time,json


def close_tree(a,b,rtol=1e-5,atol=1e-7):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_allclose(x,y,rtol=rtol,atol=atol)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest=verify();checks={};times={}
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    plan=json.loads((Path(__file__).parent/'plan.json').read_text())
    for n,m in plan['sizes']:
        start=time.monotonic();s,k=initialize(0);f=base(n,m);t,_=f['teacher'](s.params,OBS,k,QARG)
        original=jax.grad(f['loss'])(s.params,t);gs=mode_grads(s.params,t)
        close_tree(original,jax.tree_util.tree_map(lambda x:x.sum(0),gs),rtol=3e-4,atol=3e-6)
        old=s;kk=k
        for _ in range(3):old,kk,_,_,_=f['step'](old,OBS,kk,QARG)
        new,nk,_=block(n,m)(s,k,3);jax.block_until_ready(new.params)
        close_tree(old,new,rtol=3e-4,atol=3e-6);np.testing.assert_array_equal(kk,nk)
        mass,alpha,row=responsibility(s.params,t)
        np.testing.assert_allclose(alpha.sum(),1,atol=2e-6)
        np.testing.assert_allclose(mass.sum(0),np.bincount(np.digitize(np.asarray(t['b'][0,:,0]),[-.3,.3]),weights=np.asarray(t['w'][0]),minlength=3),atol=2e-6)
        # The detached teacher cannot backpropagate into Q/proposal weights or candidates.
        gg=jax.grad(lambda w:mode_terms(s.params,{**t,'w':w}).sum())(t['w']);assert np.count_nonzero(gg)==0
        # Finite differences along normalized full gradient validate sign/magnitude.
        v,unravel=ravel_pytree(s.params);g=ravel_pytree(original)[0];direction=g/jnp.linalg.norm(g);eps=.0001
        fd=(mode_terms(unravel(v+eps*direction),t)-mode_terms(unravel(v-eps*direction),t))/(2*eps)
        analytic=np.array([float(ravel_pytree(jax.tree_util.tree_map(lambda x:x[i],gs))[0]@direction) for i in range(3)])
        np.testing.assert_allclose(fd,analytic,rtol=.03,atol=.005)
        saved=serialization.to_bytes(new);restored=serialization.from_bytes(new,saved)
        close_tree(block(n,m)(new,nk,1),block(n,m)(restored,nk,1),rtol=0,atol=0)
        checks[f'N{n}_M{m}']=dict(decomposed_gradient_matches_original=True,original_updates_preserved=True,
            assignment_mass_conserved=True,teacher_stopped=True,finite_difference_gradient=True,checkpoint_next_update_exact=True)
        times[f'{n}x{m}']=time.monotonic()-start
    n,m=plan['sizes'][0]
    s,k=initialize(0);before=serialization.to_bytes(s);key=np.asarray(k).copy();d,summary=probe(s,k,n,m,0)
    assert serialization.to_bytes(s)==before;np.testing.assert_array_equal(k,key)
    assert d['delta_mu'].shape==(8,2048,1) and d['delta_reference_nll'].shape==(8,3)
    # At step zero the zero-gradient Adam control must not move anything.
    np.testing.assert_array_equal(d['delta_mu'][1],0)
    np.testing.assert_allclose(np.asarray(basin_prob(jnp.zeros((1,1)),jnp.zeros((1,1)))).sum(),1,atol=1e-6)
    checks.update(probe_preserves_training_actor_adam_rng=True,zero_momentum_control=True,fixed_latent_identity=True)
    # Small M can have no candidates from a mode. Its gradient is zero, not evidence of no interference.
    teacher,_=base(n,m)['teacher'](s.params,OBS,k,QARG)
    empty={**teacher,'b':jnp.full_like(teacher['b'],-.6),'u':jnp.full_like(teacher['u'],jnp.arctanh(-.6))}
    eg=mode_grads(s.params,empty)
    for x in jax.tree_util.tree_leaves(eg):np.testing.assert_array_equal(np.asarray(x)[1:],0)
    assert d['teacher_mode_count'].sum()==m
    np.testing.assert_allclose(d['teacher_mode_mass'].sum(),1,atol=2e-6)
    checks['empty_mode_gradient_and_teacher_coverage']=True
    write(a.out/'VALIDATION_PASSED.json',dict(commit=manifest['commit'],passed=True,checks=checks,timing_seconds=times,probe_seconds=summary['probe_seconds']))
    print('VALIDATION_PASSED',checks,flush=True)

if __name__=='__main__':main()
