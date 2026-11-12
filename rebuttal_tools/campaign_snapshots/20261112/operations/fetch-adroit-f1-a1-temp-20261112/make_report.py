from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,html
P=Path(__file__).resolve().parent;protocol=json.loads((P/'protocol.json').read_text());status=[json.loads((P/f'launch-status-host{n}.json').read_text()) for n in [3,4,5]]
settings=[('환경','기본 Fetch-v4 sparse','Adroit Sparse-v2'),('과제','Reach · Push · PickAndPlace · Slide','Door · Hammer · Pen · Relocate'),('Actor / critic','256×2 / 256×2','256×2 / 256×2'),('Critic LayerNorm','off','on, ε=10⁻⁶'),('병렬 환경 / batch','256 / 256','256 / 256'),('UTD','1/8','1/4'),('Horizon','50','200'),('데이터','future HER 204/256','동일한 검증 expert25, 시연:온라인 128:128'),('정규화','완료 에피소드의 HER 관측·목표 통계','관측·보상 정규화 off'),('BC 사전학습','없음','없음'),('γ / τ / LR','0.99 / 0.005 / 3×10⁻⁴','0.99 / 0.005 / 3×10⁻⁴'),('iBOLT log σ / DACER noise scale','[-5,1] / 2.0','[-5,-1] / 0.1'),('iBOLT 주 평가','fresh stochastic z, μ only; 마지막 step 성공','z=0, μ only; 에피소드 중 한 번 이상 성공'),('SAC 주 평가','deterministic actor','deterministic actor')]
body='<tr><th>설정</th><th>Fetch F1</th><th>Adroit A1</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in settings)
active='<tr><th>서버</th><th>학습 중</th><th>확인한 step 범위</th><th>여유 공간</th></tr>'
for x in status:
 vals=[r['steps'] for r in x['rows'] if r['steps'] is not None];active+=f'<tr><td>{x["host"]}</td><td>{len(x["rows"])}</td><td>{min(vals):,}–{max(vals):,}</td><td>{x["free_gib"]:.1f} GiB</td></tr>'
when=datetime.fromtimestamp(max(x['at'] for x in status),timezone(timedelta(hours=9))).isoformat(timespec='seconds')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Fetch F1 · Adroit A1 실행 기록</title><style>body{max-width:1100px;margin:45px auto;padding:0 24px;font:16px/1.65 system-ui;color:#17283d;background:#f8fafc}h1{line-height:1.3}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{padding:11px 16px;border-bottom:1px solid #dae2ec;text-align:left}th{background:#eaf1f8}code{overflow-wrap:anywhere}.box{padding:18px 24px;background:#e8f4f1;border-left:4px solid #237c66}a{color:#1e5a9e}</style>'''
page+=f'<h1>Fetch F1 · Adroit A1</h1><p>실행 확인: {when}. 수치는 이 시점의 snapshot이며 <a href="refresh.py">refresh.py</a>로 갱신합니다.</p>'
page+='<div class="box"><b>DACER on 고정 · iBOLT temperature만 탐색</b><br>T = 0.05, 0.1, 0.25, 0.5, 1.0. 탐색 seed 5·6 각각 5M → 과제별 선택 T를 독립 seed 0~4 각각 5M으로 확인. JAX SAC는 자동 entropy를 유지한 seed 0~4 대조군.</div>'
page+='<h2>고정 프로토콜</h2><table>'+body+'</table>'
page+='<p>iBOLT 탐색 80조합 중 기존 F1과 같은 Slide 6개를 재사용하고 74개를 새로 실행합니다. JAX SAC 40개, 선택된 T의 iBOLT 확인 40개를 포함해 최종 유효 run은 160개입니다. 200개 확인 후보 중 선택된 40개만 실행됩니다. 나머지 160개 후보는 실행하지 않습니다. 환경 수의 배수로 실제 종료 step은 5,000,192입니다.</p>'
page+='<h2>선택 및 비교 기준</h2><p>모든 온도의 두 탐색 seed가 5M에 도달한 뒤, 마지막 5회 공통 정기 평가 성공률 평균으로 T를 고릅니다. 동률이면 낮은 T를 사용합니다. 확인 seed 결과에 따른 온도 재선택이나 좋은 반복 추출은 하지 않습니다. 최종 보고는 같은 seed·step·업데이트·평가 조건의 5개 seed 평균과 표본 표준편차, 5M의 100-episode 평가입니다.</p><p>Raw return도 보존합니다. Reward-range score를 사용할 경우 Fetch는 100(R+50)/50, Adroit는 100(R+20)/2020로 정의하며 공식 D4RL normalized score로 표현하지 않습니다.</p>'
page+='<h2>병렬 실행</h2><table>'+active+'</table><p>현재 31개는 Adroit 학습이며 실제 optimizer 업데이트를 확인했습니다. Fetch 8개 연결 검증은 모두 통과했고 본 학습은 동일 큐에서 자원이 나면 시작합니다. 다른 사용자가 사용하는 GPU는 제외하며, 서버1에는 새 실행을 배치하지 않았고 서버2는 사용 대상에서 제외했습니다.</p>'
page+='<h2>중앙 수집과 보존</h2><p><code>ssh heechan@dil.yonsei.ac.kr -p 2200</code><br><code>/data1/heechan/iBOLT/20261112-f1-a1-temp</code></p><p>서버별 경로는 <code>hosts/hostN/workspace/heechan-unified-f1-a1-20261112/</code>입니다. 설정, 고정 소스, expert 데이터, 평가·학습 로그와 체크포인트를 수집합니다. 중앙 SHA256이 일치하는 이전 체크포인트 사본만 학습 서버에서 정리하며 각 run의 최신 두 step은 남깁니다. 다른 사용자의 파일은 정리하지 않습니다. 대규모 과거 기록 복사는 백그라운드에서 이어집니다.</p>'
page+='<p>신규 iBOLT·JAX SAC의 체크포인트 9개 파일을 중앙 사본과 직접 비교한 <a href="archive-end-to-end-check.json">SHA256 검증 결과</a>를 보존했습니다. Adroit 4개 과제의 25/25 저장 시연 재생 검증은 서버3·4·5에서 모두 통과했습니다.</p>'
page+='<p><a href="protocol.json">전체 실행 목록·프로토콜 JSON</a> · <a href="pilot-curves.png">탐색 곡선</a> · <a href="comparison-curves.png">독립 확인 비교 곡선</a> · <a href="snapshot.json">중앙 snapshot</a></p></html>'
(P/'report.html').write_text(page)
(P/'execution-receipt.json').write_text(json.dumps(dict(at=when,active=sum(len(x['rows']) for x in status),hosts=status,archive_check=json.loads((P/'archive-end-to-end-check.json').read_text()),automation='fetch-f1-adroit-a1-sweep-jax-sac',scope='All current active rows are Adroit; Fetch main is automatically queued after successful integration smokes'),indent=2))
print(P/'report.html')
