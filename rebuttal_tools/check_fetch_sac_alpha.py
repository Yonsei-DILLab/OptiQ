"""Check prespecified alpha selection without training a policy."""
import json
from pathlib import Path
import tempfile
from rebuttal_tools.fetch_sac_alpha_campaign import select_alpha


def check_selection():
    with tempfile.TemporaryDirectory() as directory:
        jobs = []
        for alpha,late,early in [(.001,.6,.6),(.01,.8,.3),(.1,.8,.7)]:
            for seed in (0,1):
                folder = Path(directory)/f'{alpha}-{seed}'
                folder.mkdir()
                records = [dict(steps=t,primary_mode='policy',policy=dict(success_rate=v,episodes=10))
                    for t,v in [(0,0),(1000000,early),(4975104,late),(4980224,late),
                        (4985088,late),(4990208,late),(4995072,late)]]
                records.append(dict(steps=5000192,primary_mode='policy',policy=dict(success_rate=1 if alpha==.001 else 0,episodes=100)))
                (folder/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                jobs.append(dict(alpha=alpha,seed=seed,status='completed',output=str(folder)))
        assert select_alpha(jobs)['selected_alpha']==.1
        assert select_alpha(list(reversed(jobs)))['selected_alpha']==.1
        failed=[dict(j,status='failed') if j['alpha']==.1 else j for j in jobs]
        result=select_alpha(failed)
        assert result['selected_alpha']==.01 and result['excluded'][0]['alpha']==.1
        try:
            select_alpha(jobs[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted a one-seed candidate')
    return dict(passed=True,checks=['Two pilot seeds required','Last5 primary; AUC tie-break',
        'Final100 excluded from selection','Input ordering irrelevant','Failed candidates explicitly reported'])


if __name__=='__main__':
    print(json.dumps(check_selection()),flush=True)
