"""Four independent fresh Humanoid seeds for this temperature."""
import subprocess
import time
from run import BUNDLE, context

c, p, original, d = context()
gate = c.read(BUNDLE / 'PREFLIGHT.json')
assert gate['passed'] and gate['commit'] == d['commit']
dest = BUNDLE / 'SUBMISSION.json'
assert not dest.exists(), 'Submission already recorded; inspect before retrying'
partitions = ','.join(x for x in p['partitions'].split(',') if x != 'big_suma_rtx3090')
(BUNDLE / 'logs').mkdir(exist_ok=True)
cmd = ['sbatch', '--parsable', '--array=0-3%4', '--partition=' + partitions,
       '--qos=base_qos', '--exclude=' + p['excluded_nodes'],
       '--job-name=hum-fresh-T' + str(p['temperature']),
       '--output=' + str(BUNDLE / 'logs/%A_%a.out'),
       str(BUNDLE / 'job.sbatch'), str(BUNDLE)]
record = dict(commit=d['commit'], original_commit=original['commit'], source_code_id=p['source_code_id'],
              temperature=p['temperature'], seeds=[0,1,2,3], command=cmd, time=time.time(), state='submitting')
c.write(dest, record)
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
job = result.stdout.strip().split(';')[0]
assert job.isdigit(), result.stdout
record.update(job_id=job, state='submitted')
c.write(dest, record)
print(dest.read_text())
