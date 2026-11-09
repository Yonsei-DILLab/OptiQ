(() => {
  'use strict';
  const A = window.FetchAnalytics;
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const defaults = { task: 'pick-and-place', metric: 'final', temperature: .25, cohort: 'sweep', windowSize: 1, tab: 'compare', band: false };
  let options = { ...defaults };
  try { options = { ...defaults, ...JSON.parse(localStorage.getItem('fetch-live-controls') || '{}') }; } catch (_) { /* defaults */ }
  if (!A.metrics[options.metric]) options.metric = 'final';
  if (!['pick-and-place', 'push', 'slide', 'reach'].includes(options.task)) options.task = defaults.task;
  let state = null, revision = null, currentSeries = [], fetching = false;
  let hoverStep = null;
  const hidden = { compare: new Set(), sweep: new Set(), mfpo: new Set() };
  const statusLabels = { running: '학습 중', initializing: '초기화', queued: '시작 대기', completed: '완료', failed: '실패', waiting_for_previous_task: '이전 환경 대기', waiting_for_selection: 'MFPO 설정 선택 대기', superseded: '기본값 중단 · 결과 보존' };
  const fmtStep = value => value >= 1000000 ? `${(value / 1000000).toFixed(1)}M` : value >= 1000 ? `${(value / 1000).toFixed(value % 1000 ? 1 : 0)}K` : String(value);
  const fmtValue = (value, metric = options.metric) => value === undefined || value === null ? '—' : metric === 'return_' ? value.toFixed(1) : `${(value * 100).toFixed(1).replace(/\.0$/, '')}%`;
  const clock = stamp => stamp ? new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(new Date(stamp * 1000)) : '—';

  function element(tag, props = {}, text) {
    const node = document.createElement(tag);
    Object.entries(props).forEach(([key, value]) => {
      if (key === 'class') node.className = value;
      else if (key === 'style') node.setAttribute('style', value);
      else node.setAttribute(key, value);
    });
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function svg(tag, props = {}, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(props).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function saveOptions() {
    try { localStorage.setItem('fetch-live-controls', JSON.stringify(options)); } catch (_) { /* optional */ }
  }
  function syncControls() {
    $('task').value = options.task;
    $('metric').value = options.metric;
    $('temperature').value = String(options.temperature);
    $('smooth').value = String(options.windowSize);
    $('band').checked = options.band;
    const selected = A.selection(state, options.task);
    $('cohort').options[1].disabled = !selected;
    if (!selected && options.cohort === 'final') options.cohort = 'sweep';
    $('cohort').value = options.cohort;
    const compare = options.tab === 'compare';
    $('temp-control').hidden = !compare;
    $('cohort-control').hidden = !compare;
    for (const tab of ['compare', 'sweep', 'mfpo']) {
      $(`tab-${tab}`).classList.toggle('selected', options.tab === tab);
      $(`tab-${tab}`).setAttribute('aria-selected', String(options.tab === tab));
    }
  }
  function header() {
    if (!state) return;
    const now = Date.now() / 1000;
    const hosts = state.hosts || [];
    const stale = hosts.filter(h => !h.ok || !h.last_success || now - h.last_success > 90);
    const timestamps = hosts.map(h => h.last_success).filter(Boolean);
    const fetched = timestamps.length ? Math.min(...timestamps) : null;
    $('connection-state').textContent = stale.length ? `${stale.length}개 서버 연결 확인 필요` : '두 서버 연결 정상';
    $('last-update').textContent = fetched ? `마지막 수집 ${clock(fetched)} KST${state.collecting ? ' · 수집 중' : ''}` : '첫 데이터 수집 중…';
    $('warning').hidden = stale.length === 0;
    $('warning').textContent = stale.map(h => `Server ${h.id}: ${h.last_success ? `${clock(h.last_success)}의 마지막 데이터를 표시 중` : '아직 수집된 데이터 없음'}${h.error ? ` · ${h.error}` : ''}`).join(' / ');
    $('hosts').textContent = hosts.map(h => `Server ${h.id} ${h.ok ? '●' : '○'} ${clock(h.last_success)}`).join('  ·  ');
    const jobs = A.allJobs(state);
    const counts = jobs.reduce((a, j) => { a[j.status] = (a[j.status] || 0) + 1; return a; }, {});
    $('active-count').textContent = fetched ? (counts.running || 0) + (counts.initializing || 0) : '—';
    $('init-count').textContent = `초기화 ${counts.initializing || 0}개 포함`;
    $('done-count').textContent = fetched ? counts.completed || 0 : '—';
    $('failed-count').textContent = fetched ? counts.failed || 0 : '—';
    $('failed-count').classList.toggle('danger', !!counts.failed);
    $('waiting-count').textContent = fetched ? (counts.waiting_for_previous_task || 0) + (counts.waiting_for_selection || 0) + (counts.queued || 0) : '—';
  }
  function legend() {
    $('legend').replaceChildren();
    for (const s of currentSeries) {
      const button = element('button', { type: 'button', 'aria-pressed': String(!hidden[options.tab].has(s.id)), style: `--series:${s.color}` });
      button.append(element('span', { class: 'line-key' }), document.createTextNode(s.label), element('small', {}, `n=${s.expected}`));
      button.addEventListener('click', () => {
        const set = hidden[options.tab];
        if (set.has(s.id)) set.delete(s.id); else set.add(s.id);
        button.setAttribute('aria-pressed', String(!set.has(s.id)));
        draw();
      });
      $('legend').append(button);
    }
  }
  function table() {
    const series = options.tab === 'mfpo' ? A.mfpoSweep(state, options) : A.comparison(state, options);
    $('summary').replaceChildren();
    $('value-heading').textContent = A.metrics[options.metric].label;
    for (const s of series) {
      const tr = element('tr');
      const name = element('td');
      const label = element('span', { class: 'method-name', style: `--series:${s.color}` });
      label.append(element('i'), document.createTextNode(s.label), element('small', {}, '+HER'));
      name.append(label); tr.append(name);
      tr.append(element('td', {}, `${s.available}/${s.expected}`));
      const progress = element('td');
      const percentage = Math.min(100, s.minStep / 10000);
      const track = element('span', { class: 'progress-track', style: `--series:${s.color}` });
      track.append(element('i', { style: `width:${percentage}%` }));
      progress.append(track, document.createTextNode(s.jobs.length ? `${(s.minStep / 10000).toFixed(1)}–${(s.maxStep / 10000).toFixed(1)}%` : '—'));
      progress.append(element('span', { class: 'step-range' }, `${fmtStep(s.minStep)}–${fmtStep(s.maxStep)}`)); tr.append(progress);
      const resultCell = element('td', {}, fmtValue(s.latestRaw?.mean));
      if (s.latestRaw) resultCell.append(element('span', {class:'step-range'}, `${s.latestRaw.episodesPerSeed?.[0] || '—'}회 / 시드`));
      tr.append(resultCell, element('td', {}, fmtValue(s.latest5?.mean)), element('td', {}, s.latestRaw ? fmtStep(s.latestRaw.step) : '—'));
      const status = element('td');
      let labelText = '평가 대기', cls = 'waiting';
      if (s.statuses.failed) { labelText = `실패 ${s.statuses.failed}`; cls = 'failed'; }
      else if (s.statuses.running) { labelText = `학습 ${s.statuses.running}`; cls = ''; }
      else if (s.statuses.initializing) { labelText = '초기화'; cls = ''; }
      else if (s.statuses.completed === s.expected) { labelText = '완료'; cls = ''; }
      else if (s.statuses.waiting_for_previous_task) labelText = '이전 환경 대기';
      else if (s.statuses.superseded) labelText = '기본값 중단';
      else if (s.statuses.waiting_for_selection) labelText = 'MFPO 설정 선택 대기';
      status.append(element('span', { class: `status ${cls}` }, labelText)); tr.append(status);
      $('summary').append(tr);
    }
    const jobs = A.allJobs(state).filter(j => j.task === options.task);
    $('run-total').textContent = `(${jobs.length})`;
    $('runs').replaceChildren();
    for (const j of jobs) {
      const tr = element('tr');
      tr.append(element('td', {}, j.method.toUpperCase() + (j.method === 'ibolt' ? ` · T=${j.temperature}` : j.phase === 'mfpo-hyperparameter-sweep' ? ` · H=${4*j.target_entropy_coeff} · α₀=${j.initial_temperature}` : j.phase === 'mfpo-selected-5seed' ? ' · 선택 후 재실행' : '')),
        element('td', {}, j.seed), element('td', {}, fmtStep(j.steps || 0)), element('td', {}, `${((j.steps || 0) / 10000).toFixed(1)}%`),
        element('td', {}, fmtValue(j.evaluations?.at(-1)?.final, 'final')), element('td', {}, `${j.host} / ${j.gpu}`));
      const td = element('td');
      td.append(element('span', { class: `status ${j.status === 'failed' ? 'failed' : j.status.includes('waiting') || j.status === 'queued' ? 'waiting' : ''}` }, statusLabels[j.status] || j.status));
      tr.append(td); $('runs').append(tr);
    }
  }
  function draw() {
    const plot = $('plot');
    const width = Math.max(260, $('chart').clientWidth), height = $('chart').clientHeight;
    const mobile = width < 500;
    const margin = { left: mobile ? 44 : 52, right: mobile ? 15 : 26, top: 23, bottom: 43 };
    const w = width - margin.left - margin.right, h = height - margin.top - margin.bottom;
    const metric = A.metrics[options.metric];
    const visible = currentSeries.filter(s => !hidden[options.tab].has(s.id));
    const allPoints = visible.flatMap(s => s.points);
    const maximum = Math.max(20000, ...allPoints.map(p => p.step));
    const domainMax = Math.ceil(maximum / (maximum < 100000 ? 20000 : 50000)) * (maximum < 100000 ? 20000 : 50000);
    const x = step => margin.left + step / domainMax * w;
    const y = value => margin.top + (1 - (value - metric.min) / (metric.max - metric.min)) * h;
    plot.setAttribute('viewBox', `0 0 ${width} ${height}`);
    plot.setAttribute('aria-label', `${metric.label} 학습 곡선. ${visible.length}개 실험, 최대 ${fmtStep(domainMax)} transitions. 표에서 정확한 값을 확인할 수 있습니다.`);
    plot.replaceChildren();
    for (let i = 0; i <= 5; i++) {
      const value = metric.min + (metric.max - metric.min) * i / 5;
      plot.append(svg('line', { class: 'grid', x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value) }));
      plot.append(svg('text', { x: margin.left - 10, y: y(value) + 3, 'text-anchor': 'end' }, metric.unit ? `${Math.round(value * 100)}%` : Math.round(value)));
    }
    const ticks = mobile ? 4 : 5;
    for (let i = 0; i <= ticks; i++) {
      const step = domainMax * i / ticks;
      plot.append(svg('text', { x: x(step), y: height - 24, 'text-anchor': 'middle' }, fmtStep(Math.round(step))));
    }
    plot.append(svg('text', { class: 'axis-label', x: margin.left + w / 2, y: height - 4, 'text-anchor': 'middle' }, 'Environment transitions'));
    plot.append(svg('line', { class: 'axis', x1: margin.left, x2: width - margin.right, y1: margin.top + h, y2: margin.top + h }));
    if (allPoints.some(p => p.episodesPerSeed?.some(n => n === 100))) {
      plot.append(svg('line', {class:'grid', x1:x(1000000),x2:x(1000000),y1:margin.top,y2:margin.top+h,'stroke-dasharray':'3 4'}));
      plot.append(svg('text', {x:x(1000000)-3,y:margin.top-9,'text-anchor':'end'}, '■ 최종 100회 / 시드'));
    }
    for (const s of visible) {
      if (!s.points.length) continue;
      const segments = [];
      for (const point of s.points) {
        const key = (point.episodesPerSeed || []).join(',');
        if (!segments.length || segments.at(-1).key !== key) segments.push({key,points:[]});
        segments.at(-1).points.push(point);
      }
      for (const segment of segments) {
      if (options.band && segment.points.length > 1) {
        const upper = segment.points.map(p => `${x(p.step)},${y(Math.min(metric.max, p.mean + p.sd))}`);
        const lower = segment.points.slice().reverse().map(p => `${x(p.step)},${y(Math.max(metric.min, p.mean - p.sd))}`);
        plot.append(svg('polygon', { points: [...upper, ...lower].join(' '), fill: s.color, opacity: '.09', 'pointer-events': 'none' }));
      }
      if (segment.points.length > 1) plot.append(svg('polyline', { class: 'series', points: segment.points.map(p => `${x(p.step)},${y(p.mean)}`).join(' '), stroke: s.color, 'stroke-width': s.id === 'ibolt' || options.tab === 'sweep' && s.id === String(options.temperature) ? 2.6 : 1.8, 'stroke-dasharray': s.dash }));
      const last = segment.points.at(-1);
      if (last.episodesPerSeed?.every(n => n === 100)) plot.append(svg('rect', {x:x(last.step)-4,y:y(last.mean)-4,width:8,height:8,fill:s.color,stroke:'var(--surface)','stroke-width':1.5}));
      else plot.append(svg('circle', { cx: x(last.step), cy: y(last.mean), r: 3.2, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 1.5 }));
      }
    }
    const hoverLine = svg('line', { class: 'hover-line', y1: margin.top, y2: height - margin.bottom, visibility: 'hidden' });
    plot.append(hoverLine);
    const hit = svg('rect', { x: margin.left, y: margin.top, width: w, height: h, fill: 'transparent', tabindex: '0', role: 'slider', 'aria-label': '평가 step 선택. 좌우 화살표로 이동', 'aria-valuemin': 0, 'aria-valuemax': domainMax, 'aria-valuenow': 0 });
    plot.append(hit);
    function show(step) {
      hoverStep = Math.max(0, Math.min(domainMax, Math.round(step / 5000) * 5000));
      hit.setAttribute('aria-valuenow', hoverStep);
      hoverLine.setAttribute('x1', x(hoverStep)); hoverLine.setAttribute('x2', x(hoverStep)); hoverLine.setAttribute('visibility', 'visible');
      const tip = $('tooltip'); tip.replaceChildren(element('strong', {}, `${fmtStep(hoverStep)} transitions · ${hoverStep === 1000000 ? '최종 100회' : '중간 10회'} / 시드`));
      for (const s of visible) {
        const point = s.points.find(p => p.step === hoverStep);
        const row = element('div', { class: 'tip-row' });
        const name = element('span'); name.append(element('i', { class: 'tip-dot', style: `background:${s.color}` }), document.createTextNode(s.label));
        row.append(name, element('b', {}, point ? fmtValue(point.mean) : '—')); tip.append(row);
      }
      tip.hidden = false;
      tip.style.left = `${Math.max(4, Math.min(width - tip.offsetWidth - 6, x(hoverStep) + 14))}px`;
      tip.style.top = '27px';
    }
    hit.addEventListener('pointermove', e => {
      const rect = plot.getBoundingClientRect();
      show(((e.clientX - rect.left) / rect.width * width - margin.left) / w * domainMax);
    });
    hit.addEventListener('pointerleave', () => { $('tooltip').hidden = true; hoverLine.setAttribute('visibility', 'hidden'); });
    hit.addEventListener('focus', () => show(hoverStep || 0));
    hit.addEventListener('blur', () => { $('tooltip').hidden = true; hoverLine.setAttribute('visibility', 'hidden'); });
    hit.addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); show((hoverStep || 0) + (e.key === 'ArrowRight' ? 5000 : -5000)); }
      if (e.key === 'Escape') { $('tooltip').hidden = true; hoverLine.setAttribute('visibility', 'hidden'); }
    });
    $('empty').hidden = allPoints.length > 0;
    $('tooltip').hidden = true;
  }
  function render() {
    syncControls(); header();
    currentSeries = options.tab === 'mfpo' ? A.mfpoSweep(state, options) : options.tab === 'sweep' ? A.sweep(state, options) : A.comparison(state, options);
    const metric = A.metrics[options.metric];
    $('chart-title').textContent = `${metric.label}${options.windowSize > 1 ? ` · 최근 ${options.windowSize}회 이동평균` : ''}`;
    $('chart-description').textContent = '같은 step·모든 지정 시드 평균 · 중간 고정 10회, 최종 100회/시드. 표본이 바뀌면 곡선을 분리하고 평활화를 초기화해요.';
    $('metric-note').textContent = metric.note;
    $('seed-note').textContent = options.tab === 'sweep' || options.cohort === 'sweep' ? 'iBOLT 탐색 2시드 · baseline 5시드 · 온도 선택 전 중간 결과' : '최종 5시드 · 선택에 사용한 seed 0·1 포함';
    if (options.tab === 'mfpo') $('seed-note').textContent = '9조합 × seed 0·1 · H는 실제 target entropy · 후반 성공률로 선택 후 5시드 재실행';
    $('empty').querySelector('strong').textContent = options.tab === 'mfpo' ? (options.task === 'pick-and-place' ? 'MFPO 탐색의 초기 평가를 기다리고 있어요.' : 'MFPO 그리드는 PickAndPlace에서 진행해요.') : '아직 이 환경의 평가가 없어요.';
    $('empty').querySelector('span').textContent = options.tab === 'mfpo' ? (options.task === 'pick-and-place' ? '각 조합의 seed 0·1 평가가 모이면 곡선이 자동으로 표시돼요.' : '선택된 설정으로 다른 환경을 실행하며, 방법 비교 탭에서 확인할 수 있어요.') : '각 방법의 이전 환경 5시드가 완료되면 자동으로 시작돼요.';
    legend(); draw(); table();
  }
  async function load() {
    if (fetching) return;
    fetching = true;
    try {
      const response = await fetch('/api/state', { cache: 'no-store' });
      if (!response.ok) throw new Error('응답 오류');
      state = await response.json();
      if (state.revision !== revision) { revision = state.revision; render(); } else header();
    } catch (_) {
      $('warning').hidden = false;
      $('warning').textContent = '대시보드 연결이 끊겼어요. 마지막으로 수집한 그래프를 표시하고 있어요.';
      $('connection-state').textContent = '로컬 수집기 연결 확인 필요';
    } finally { fetching = false; }
  }
  for (const name of ['task', 'metric', 'temperature', 'cohort', 'smooth', 'band']) {
    $(name).addEventListener('change', () => {
      options.task = $('task').value; options.metric = $('metric').value;
      options.temperature = Number($('temperature').value); options.cohort = $('cohort').value;
      options.windowSize = Number($('smooth').value); options.band = $('band').checked;
      const selected = A.selection(state, options.task);
      if (options.cohort === 'final' && selected) {
        if (name === 'cohort') options.temperature = selected.temperature;
        else if (Number(selected.temperature) !== options.temperature) options.cohort = 'sweep';
      }
      saveOptions(); render();
    });
  }
  for (const tab of ['compare', 'sweep', 'mfpo']) $(`tab-${tab}`).addEventListener('click', () => { options.tab = tab; saveOptions(); render(); });
  $('refresh').addEventListener('click', async () => {
    $('refresh').disabled = true;
    try { await fetch('/api/refresh', { method: 'POST' }); await load(); } finally { setTimeout(() => { $('refresh').disabled = false; }, 1500); }
  });
  $('auto').addEventListener('change', () => { if ($('auto').checked) load(); });
  let lastWidth = 0;
  new ResizeObserver(entries => { const width = Math.round(entries[0].contentRect.width); if (width !== lastWidth) { lastWidth = width; requestAnimationFrame(draw); } }).observe($('chart'));
  setInterval(() => { if ($('auto').checked) load(); }, 5000);
  setInterval(header, 10000);
  render(); load();
  window.FetchDashboard = { getState: () => state, getSeries: () => currentSeries, getOptions: () => ({ ...options }), refresh: load };
})();
