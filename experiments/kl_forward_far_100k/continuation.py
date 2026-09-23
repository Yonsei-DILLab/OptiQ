"""Exact actor/Adam/RNG continuation from immutable 20K checkpoints."""
from pathlib import Path
import hashlib,json,shutil

ORIGINAL_COMMIT='1153ff9a797173681a980fffea6f2e75414fe40d'
ALLOWED_CONFIG_CHANGES={'study','steps','eval_steps'}

def same_settings(old,new):
    changes={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    assert changes<=ALLOWED_CONFIG_CHANGES,changes
    assert old['steps']==20000 and new['steps']==100000

def restore_parent(exp,parent,out,root,seed):
    import flax.serialization
    import jax
    import numpy as np
    parent,out,root=map(Path,(parent,out,root))
    run=json.loads((parent/'RUN.json').read_text());complete=json.loads((parent/'COMPLETE.json').read_text())
    assert complete['step']==20000 and run['seed']==seed
    assert run['source_commit']==complete['source_commit']==ORIGINAL_COMMIT
    same_settings(run['config'],exp.cfg)
    checkpoint=parent/'checkpoint.msgpack';blob=checkpoint.read_bytes();digest=hashlib.sha256(blob).hexdigest()
    expected=json.loads((root/'PARENT_CHECKPOINTS.json').read_text())[f'N{exp.n}M{exp.m}/s{seed}']
    assert digest==expected['sha256']
    exp.restore(checkpoint)
    assert int(exp.state.step)==20000
    # Check parameters, both Adam moments/count, TrainState step and PRNG key.
    decoded=flax.serialization.msgpack_restore(blob)
    restored=flax.serialization.to_state_dict({'state':exp.state,'key':exp.key})
    old_leaves,old_tree=jax.tree_util.tree_flatten(decoded)
    new_leaves,new_tree=jax.tree_util.tree_flatten(restored)
    assert old_tree==new_tree
    assert all(np.array_equal(x,y) for x,y in zip(old_leaves,new_leaves))
    for source,target in [('checkpoint.msgpack','start_20000.msgpack'),('RUN.json','PARENT_RUN.json'),('COMPLETE.json','PARENT_COMPLETE.json'),('samples_20000.npz','parent_samples_20000.npz'),('metrics_20000.json','parent_metrics_20000.json')]:
        shutil.copy2(parent/source,out/target)
    return dict(parent_run=str(parent),parent_source_commit=ORIGINAL_COMMIT,parent_checkpoint_sha256=digest,parent_step=20000,parent_training_seconds=complete['training_seconds'],restored_leaves_exact=True,restored_leaf_count=len(old_leaves),restored_parameter_sha256=hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest())
