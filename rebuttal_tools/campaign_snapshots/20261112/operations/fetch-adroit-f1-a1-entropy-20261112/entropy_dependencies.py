"""Dependency resolution only; never use confirmation performance to select."""
import hashlib,json
def resolve_dependency(j,root,read):
    key=j['family']+'-'+j['task']
    t=read(root/'entropy-temperature-selections'/(key+'.json')) or read(root/'selections'/(key+'.json'))
    if not t:return
    assert t['pilot_seeds']==[5,6] and t['expected_steps']==5000192
    assert t['temperature'] in [.05,.1,.25,.5,1.]
    assert len(t['common_steps'])==5
    if t.get('selection_scope')=='entropy-current-common-step':
        assert t['authorized_early_freeze'] is True and t['common_steps'][-1]>0
    else:assert t['common_steps'][-1]>4900000
    j['temperature']=t['temperature']
    j['temperature_selection_sha256']=hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest()
    if j['status']=='waiting_temperature':j['status']='queued';return
    h=read(root/'entropy-selections'/(key+'.json'))
    if not h:return
    assert h['temperature']==j['temperature'] and h['pilot_seeds']==[5,6]
    assert h['target_entropy_per_dim'] in [-.9,0.,.25,.5,.75]
    j['dacer_target_entropy_per_dim']=h['target_entropy_per_dim']
    j['entropy_selection_sha256']=hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest()
    if h['target_entropy_per_dim']==-.9:
        original=read(root/'selections'/(key+'.json'))
        if not original:return
        if original['temperature']!=j['temperature']:
            # The fixed early T differs from the original completed-grid winner.
            # No existing confirmation at this setting can be reused.
            j['status']='queued';return
        prefix=key+('-r20cm' if j['task']=='slide' else '')
        # Original temperature labels use 1p0, 0p1, etc.
        tag=str(float(j['temperature'])).replace('.','p')
        j['reference_id']=f"{prefix}-ibolt-s{j['seed']}-t{tag}-confirm"
        j['status']='reused_reference'
    else:j['status']='queued'
