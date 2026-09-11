"""Exact checkpoint-value validation, invariant to dictionary serialization order."""
import hashlib,json
from pathlib import Path
from flax import serialization

def canonical_state_digest(data):
    def ordered(value):
        if isinstance(value,dict):return {k:ordered(value[k]) for k in sorted(value)}
        if isinstance(value,list):return [ordered(v) for v in value]
        if isinstance(value,tuple):return tuple(ordered(v) for v in value)
        return value
    state=serialization.msgpack_restore(data)
    return hashlib.sha256(serialization.msgpack_serialize(ordered(state))).hexdigest()

def verify_initialization(task,mark=False):
    original=Path(task['initial_checkpoint']).read_bytes()
    assert hashlib.sha256(original).hexdigest()==task['initial_sha256'], 'Reference checkpoint file changed'
    observed=(Path(task['out'])/'actor_0.msgpack').read_bytes()
    original_digest=canonical_state_digest(original);observed_digest=canonical_state_digest(observed)
    assert original_digest==observed_digest, 'Initial actor/optimizer checkpoint values differ'
    record=dict(reference_file_sha256=task['initial_sha256'],restored_file_sha256=hashlib.sha256(observed).hexdigest(),
                canonical_state_sha256=original_digest,values_exactly_equal=True,
                ignores_dictionary_order_only=True)
    if mark:(Path(task['out'])/'PAIRED_INITIALIZATION_OK').write_text(json.dumps(record,indent=2))
    return record
