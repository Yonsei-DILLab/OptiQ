"""Read-only local comparison updater; never launches, stops or retries learning."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def update(a):
    root = '/workspace/heejoon-adroit-ibolt-e64u1-20261113'
    source = root + '/source-' + a.source_commit
    command = (f'cd {source} && /workspace/heejoon-adroit-online-20261112/venv/bin/python '
               f'-m adroit_online.collect_curves {root} --profile e64-u1')
    raw = subprocess.check_output(['ssh', '-S', 'none', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '-p', '19159',
        'heejoonorm@175.155.64.205', command], text=True, timeout=120)
    snapshot = json.loads(raw)
    assert snapshot['queue']['source_commit'] == a.source_commit
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'snapshot.json'
    temporary = path.with_suffix('.tmp'); temporary.write_text(raw); temporary.replace(path)
    subprocess.run([sys.executable, '-m', 'adroit_online.compare_e64u1', '--baselines', str(a.baselines),
                    '--snapshot', str(path), '--output', str(a.output)], check=True)
    report = json.loads((a.output / 'comparison.json').read_text())
    jobs = snapshot['queue']['jobs']; assert len(jobs) == 12
    counts = {state: sum(s['status'] == state for s in jobs.values()) for state in sorted({s['status'] for s in jobs.values()})}
    stamp = datetime.fromtimestamp(snapshot['observed_unix'], ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    rows = []
    for row in report['latest']:
        m = row['metrics']; state = jobs.get(row['run_id'], {}).get('status', 'completed baseline')
        rows.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
            (row['run_id'], state, f"{row['step']:,}", row['episodes'],
             f"{100*m['success_any']:.1f}%", f"{100*m['success_25_steps']:.1f}%")) + '</tr>')
    for name, state in jobs.items():
        if name not in {r['run_id'] for r in report['latest']}:
            rows.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(state["status"])}</td><td colspan="4">평가 대기</td></tr>')
    document = '''<!doctype html><meta charset="utf-8"><title>Adroit env64·UTD1 비교</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#203040}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.6}</style>
<h1>Adroit Sparse-v2 · env64 · UTD1</h1>'''
    document += f'<p>갱신: {stamp}<br>iBOLT 12개 실행 상태: {html.escape(str(counts))}</p>'
    document += '<p>동일 seed·환경 step의 SAC/TD3와 비교합니다. 진행 중 곡선은 마지막 평가까지만 표시합니다. 정기 평가는 100회, 3M 최종 평가는 1,000회입니다. 순간 성공은 에피소드 내 한 번 이상 성공, 25-step 성공은 누적 25회 이상이며 연속 조건이 아닙니다.</p>'
    document += '<img src="success_any.png"><img src="success_25_steps.png"><h2>각 실행의 최신 평가</h2><table><tr><th>Run</th><th>상태</th><th>평가 step</th><th>Episodes</th><th>한 번 이상</th><th>25-step 이상</th></tr>'
    document += ''.join(rows) + '</table><p>학습 소스: ' + a.source_commit + '</p>'
    temp = a.output / 'index.tmp'; temp.write_text(document); temp.replace(a.output / 'index.html')
    print(json.dumps(dict(time=time.time(), status=counts, output=str(a.output))), flush=True)
    return all(s['status'] in ('completed', 'failed', 'requires_review') for s in jobs.values())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source-commit', required=True)
    p.add_argument('--baselines', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--once', action='store_true'); a = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', a.source_commit)
    while True:
        try:
            finished = update(a)
            if a.once or finished:
                break
        except Exception as error:
            print(json.dumps(dict(time=time.time(), error=repr(error))), flush=True)
            if a.once:
                raise
        for _ in range(5):
            time.sleep(60)


if __name__ == '__main__':
    main()
