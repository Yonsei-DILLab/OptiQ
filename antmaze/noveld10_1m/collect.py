"""Collect status and completed1M archives using an explicit frozen commit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
from pathlib import Path
import subprocess
from antmaze.multimodal import collect_dense_noveld as collector
from antmaze.multimodal import dense_noveld_report as report

ROOT='/home/heechan/optiq-experiments/antmaze-dense-noveld10-1m-s0-20260922'
PROFILE='dense-noveld10-1m'


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source',required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/antmaze_dense_noveld10_1m'))
    p.add_argument('--archive-completed',action='store_true')
    p.add_argument('--report',action='store_true')
    a=p.parse_args();assert len(a.source)==40 and all(c in '0123456789abcdef' for c in a.source)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    collector.BASE=out;collector.CODE=collector.CODE.replace(collector.ROOT,ROOT);collector.ROOT=ROOT
    with ThreadPoolExecutor(2) as pool:snapshots=dict(pool.map(collector.collect,('vast-heechan-180','vast-heechan-199')))
    report.write(out/'latest.json',dict(collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=snapshots))
    for host,snapshot in snapshots.items():
        manifest=snapshot['manifest.json'];assert manifest['source_commit']==a.source and manifest['coefficient']==10
        if not a.archive_completed:continue
        for j in snapshot['status.json']['jobs']:
            if j['status']!='completed':continue
            name=j['id'];assert name==f"{j['task']}-{j['method']}-s0"
            assert j['task'] in report.TASKS and j['method'] in report.METHODS
            dest=out/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
            files=['config.json','result.json','verification.json','progress.json','wandb.json','checkpoint.json',
                'parameter-audit.json','intrinsic-audit.json','training_coverage.npz','training_episodes.json',
                'history-native-natural.json','history-policy-natural.json','baseline-config-initial-proof.json']
            filters=[f'--include=/{n}' for n in files]+['--include=/rollouts/***','--include=/resume/',
                '--include=/resume/step_0001000000/***','--exclude=*']
            subprocess.run(['rsync','-az','--checksum',*filters,f'{host}:{ROOT}/runs/{name}/',str(dest)+'/'],check=True)
            subprocess.run(['rsync','-az',f'{host}:{ROOT}/logs/training-{name}.log',str(dest/'training.log')],check=True)
            proof=report.verify_run(dest,expected_source=a.source,expected_steps=1000000,
                expected_profile=PROFILE,expected_coefficient=10)
            assert report.read(dest/'baseline-config-initial-proof.json')['passed']
            report.write(dest/'archive-verification.json',dict(**proof,host=host,remote=f'{ROOT}/runs/{name}'))
            print('ARCHIVED AND VERIFIED',name,flush=True)
    if a.report:
        complete={};proofs={}
        for task in report.TASKS:
            for method in report.METHODS:
                job=f'{task}-{method}-s0';folder=out/'runs'/job
                proofs[job]=report.read(folder/'archive-verification.json')
                assert proofs[job]['passed'] and proofs[job]['training_source_commit']==a.source
                complete[job]=report.read(folder/'result.json')
        report.TRAINING_SHA=a.source
        report.write(out/'report/validation.json',dict(complete=True,verified=proofs,missing=[],
            training_source_commit=a.source,coefficient=10,reporting_file_sha256=report.file_hash(__file__)))
        report.render(out,complete,False)


if __name__=='__main__':main()
