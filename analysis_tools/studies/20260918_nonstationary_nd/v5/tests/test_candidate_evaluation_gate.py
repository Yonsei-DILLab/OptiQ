import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts import evaluate_v2_finite as evaluator
from scripts import finish_v2_finite as finisher
from test_finite_finisher import setup


def test_completion_marker_does_not_substitute_for_actual_final_steps(tmp_path, monkeypatch):
    runs=[]
    for seed in range(4):
        path=tmp_path/str(seed);path.mkdir()
        (path/'completed.json').write_text(json.dumps({'timesteps':8000}))
        runs.append({'seed':seed,'directory':str(path),'service':f'test-{seed}'})
    monkeypatch.setattr(evaluator.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='test EXITED'))
    pending=evaluator.completion_gate({'runs':runs},{'checkpoint_step':1000000})
    assert len(pending)==4 and {r['reason'] for r in pending}=={'incomplete_step_count'}


def test_final_marker_cannot_hide_a_missing_primary_evaluation(tmp_path, monkeypatch):
    runs=[]
    for seed in range(4):
        path=tmp_path/str(seed);path.mkdir()
        (path/'completed.json').write_text(json.dumps({'timesteps':1000000}))
        (path/'config.json').write_text(json.dumps({'seed':seed,'runtime':{'git_commit':'abc'},
            'alg':{'actor':{'latent_prior':'finite'}},'total_steps':1000000}))
        eval_dir=path/'eval'/'run';eval_dir.mkdir(parents=True)
        np.savez(eval_dir/'evaluations.npz',timesteps=np.array([900000,1000000]),results=np.ones((2,10)))
        runs.append({'seed':seed,'directory':str(path),'service':f'test-{seed}','commit':'abc'})
    monkeypatch.setattr(evaluator.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='test EXITED'))
    import pytest
    with pytest.raises(AssertionError):
        evaluator.completion_gate({'runs':runs},{'checkpoint_step':1000000,
            'primary_window_steps':list(range(900000,1000001,5000))})


def test_custom_protocol_routes_reports_and_results_without_touching_old_outputs(tmp_path, monkeypatch):
    base=setup(tmp_path,monkeypatch)
    original=base/'finite_final_evaluation_protocol.json'
    protocol=json.loads(original.read_text())
    protocol.update(candidate_label='Proximal v2 candidate',
        result_path=str(base/'proximal_result.json'),report_output=str(base/'proximal_report'),
        finisher_status=str(base/'proximal_status.json'),finisher_lock=str(base/'proximal.lock'))
    custom=base/'proximal_protocol.json';custom.write_text(json.dumps(protocol))
    monkeypatch.setattr(finisher,'completion_gate',lambda *_:[])
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        if 'evaluate_v2_finite.py' in command[1]:
            Path(protocol['result_path']).write_text(json.dumps({'summary':{'complete':True},
                'results':[{'training_seed':s} for s in range(4)]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(finisher.subprocess,'run',run)
    finisher.main(['--protocol',str(custom)])
    assert commands[0][commands[0].index('--manifest')+1]==protocol['candidate_manifest']
    assert commands[0][commands[0].index('--output')+1]==protocol['report_output']
    assert commands[1][-2:]==['--protocol',str(custom)]
    assert not (base/'finite_independent_1000000.json').exists()
    state=json.loads(Path(protocol['finisher_status']).read_text())
    assert state['stage']=='complete' and state['goal_complete'] is False


def test_stopped_group_reports_all_seeds_without_running_final_episodes(tmp_path,monkeypatch):
    base=setup(tmp_path,monkeypatch)
    path=base/'finite_final_evaluation_protocol.json';protocol=json.loads(path.read_text())
    protocol['report_incomplete']=True;path.write_text(json.dumps(protocol))
    pending=iter([[{'seed':0,'status':'STOPPED'},{'seed':1,'status':'RUNNING'}],
                  [{'seed':s,'status':'STOPPED'} for s in range(4)]])
    monkeypatch.setattr(finisher,'completion_gate',lambda *_:next(pending))
    waits=[];monkeypatch.setattr(finisher.time,'sleep',waits.append)
    commands=[]
    monkeypatch.setattr(finisher.subprocess,'run',lambda c,**kw:(commands.append(c) or SimpleNamespace(returncode=0)))
    finisher.main([])
    assert waits==[30] and len(commands)==1
    assert Path(commands[0][1]).name=='report_v2_finite.py'
    assert '--through-step' not in commands[0]
    state=json.loads((base/'finite_finisher_status.json').read_text())
    assert state['stage']=='screen_incomplete' and state['goal_complete'] is False
