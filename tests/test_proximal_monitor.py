from scripts.monitor_v2_proximal import checkpoint_records
from scripts.monitor_v2_finite import strong_group_decision


def test_stopping_requires_all_eight_nonempty_checkpoints(tmp_path):
    runs=[{'seed':s,'directory':str(tmp_path/str(s))} for s in range(4)]
    for r in runs:
        directory=tmp_path/str(r['seed'])/'checkpoints'/'run'
        directory.mkdir(parents=True)
        (directory/'actor_state_100000.msgpack').write_bytes(b'actor')
        (directory/'critic_state_100000.msgpack').write_bytes(b'critic')
    assert len(checkpoint_records(runs,100000))==8
    last=tmp_path/'3/checkpoints/run/critic_state_100000.msgpack'
    last.write_bytes(b'')
    assert checkpoint_records(runs,100000) is None
    last.unlink()
    assert checkpoint_records(runs,100000) is None


def test_lagging_seed_cannot_be_omitted_from_predeclared_gate():
    def curves(value):return {s:{t:value for t in range(80000,100001,5000)} for s in range(4)}
    candidate=curves(400);continuous=curves(1000);historical=curves(1200)
    candidate[3].pop(100000)
    assert not strong_group_decision(candidate,continuous,historical,100000,.6)['available']
    candidate[3][100000]=400
    review=strong_group_decision(candidate,continuous,historical,100000,.6)
    assert review['available'] and review['stop'] and review['means']['finite']==400
