(function (root) {
  'use strict';
  const methods = [
    { id: 'ibolt', label: 'iBOLT', color: '#e55b25', dash: '' },
    { id: 'sac', label: 'SAC', color: '#2563eb', dash: '' },
    { id: 'idac', label: 'IDAC', color: '#8b5cf6', dash: '8 3' },
    { id: 'idac-nodgn', label: 'IDAC-noDGN', color: '#0891b2', dash: '3 3' },
    { id: 'mfpo', label: 'MFPO', color: '#b58306', dash: '10 3 2 3' },
    { id: 'meow', label: 'MEOW', color: '#c83b83', dash: '6 3' },
  ];
  const temperatures = [.05, .1, .25, .5, 1, 5, 10];
  const tempColors = ['#a855f7', '#2563eb', '#e55b25', '#0891b2', '#169960', '#b58306', '#c83b83'];
  const metrics = {
    final: { label: '최종 성공률', unit: '%', min: 0, max: 1, note: '에피소드 마지막 스텝에서 원래 목표를 달성한 비율' },
    new_final: { label: '신규 최종 성공률', unit: '%', min: 0, max: 1, note: '처음부터 목표 안에 있던 성공을 제외한 비율 · 분모는 전체 평가 횟수' },
    any: { label: '도중 목표 도달률', unit: '%', min: 0, max: 1, note: '에피소드 중 한 번이라도 원래 목표를 달성한 비율' },
    new_any: { label: '신규 도중 도달률', unit: '%', min: 0, max: 1, note: '처음에는 목표 밖이었고, 에피소드 도중 목표에 도달한 비율' },
    return_: { label: '평균 에피소드 보상', unit: '', min: -50, max: 0, note: '원래 sparse 보상 합계 · 50스텝, 스텝당 −1 또는 0' },
  };
  function allJobs(state) {
    return (state?.hosts || []).flatMap(h => (h.jobs || []).map(j => ({ ...j, host: h.id, hostOk: h.ok })));
  }
  function selection(state, task) {
    for (const host of state?.hosts || []) {
      const found = host.selections?.[task];
      if (found?.status === 'selected') return found;
    }
    return null;
  }
  function aggregate(jobs, seeds, metric, windowSize = 1) {
    const recordMaps = seeds.map(seed => {
      const job = jobs.find(j => j.seed === seed);
      return new Map((job?.evaluations || []).map(e => [e.step, e]));
    });
    const maps = recordMaps.map(records => {
      const sorted = [...records.keys()].sort((a, b) => a - b);
      const map = new Map();
      sorted.forEach((step, i) => {
        const episodes = records.get(step).episodes;
        const values = sorted.slice(Math.max(0, i - windowSize + 1), i + 1)
          .filter(k => records.get(k).episodes === episodes).map(k => records.get(k)[metric]);
        if (values.every(Number.isFinite)) map.set(step, values.reduce((a, b) => a + b, 0) / values.length);
      });
      return map;
    });
    if (!maps.length || maps.some(m => !m.size)) return [];
    return [...maps[0].keys()].filter(step => maps.every(m => m.has(step))).sort((a, b) => a - b).map(step => {
      const values = maps.map(m => m.get(step));
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const sd = values.length > 1 ? Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1)) : 0;
      return { step, mean, sd, n: values.length, values,
        episodesPerSeed: recordMaps.map(records => records.get(step).episodes) };
    });
  }
  function series(state, method, task, temperature, cohort, metric, windowSize) {
    let jobs = allJobs(state).filter(j => j.task === task && j.method === method.id);
    let suffix = '';
    if (method.id === 'mfpo') {
      const selected = (state?.hosts || []).map(h => h.mfpo_selection).find(s => s?.status === 'selected');
      if (selected) {
        jobs = jobs.filter(j => j.phase === 'mfpo-selected-5seed');
        suffix = ` · H=${4*selected.target_entropy_coeff}, α₀=${selected.initial_temperature}`;
      } else {
        jobs = jobs.filter(j => j.phase !== 'mfpo-hyperparameter-sweep' && j.phase !== 'mfpo-selected-5seed');
        if (jobs.some(j => j.status === 'superseded')) suffix = ' · 기본값 중단';
      }
    }
    const selected = selection(state, task);
    let seeds = [0, 1, 2, 3, 4];
    if (method.id === 'ibolt') {
      seeds = cohort === 'final' ? seeds : [0, 1];
      jobs = jobs.filter(j => Number(j.temperature) === Number(temperature));
      if (cohort === 'final' && (!selected || Number(selected.temperature) !== Number(temperature))) jobs = [];
      if (cohort !== 'final') jobs = jobs.filter(j => seeds.includes(j.seed));
    }
    const points = aggregate(jobs, seeds, metric, windowSize);
    const latestRaw = aggregate(jobs, seeds, metric, 1).at(-1);
    const latest5 = aggregate(jobs, seeds, metric, 5).at(-1);
    const steps = jobs.map(j => j.steps || 0);
    return { ...method, jobs, seeds, points, latestRaw, latest5,
      label: method.label + (method.id === 'ibolt' ? ` · T=${temperature}` : suffix),
      expected: seeds.length, available: seeds.filter(s => jobs.some(j => j.seed === s && j.evaluations?.length)).length,
      minStep: steps.length ? Math.min(...steps) : 0, maxStep: steps.length ? Math.max(...steps) : 0,
      statuses: jobs.reduce((a, j) => { a[j.status] = (a[j.status] || 0) + 1; return a; }, {}) };
  }
  function comparison(state, options) {
    return methods.map(m => series(state, m, options.task, options.temperature, options.cohort, options.metric, options.windowSize));
  }
  function sweep(state, options) {
    return temperatures.map((t, i) => ({ ...series(state, methods[0], options.task, t, 'sweep', options.metric, options.windowSize), id: String(t), label: `T=${t}`, color: tempColors[i], dash: i > 3 ? '7 3' : '' }));
  }
  function mfpoSweep(state, options) {
    const jobs = allJobs(state).filter(j => j.task === options.task && j.phase === 'mfpo-hyperparameter-sweep');
    return [-.5, -.25, 0].flatMap((coeff, ci) => [.01, .1, 1].map((initial, ai) => {
      const members = jobs.filter(j => Number(j.target_entropy_coeff) === coeff && Number(j.initial_temperature) === initial);
      const seeds = [0, 1], steps = members.map(j => j.steps || 0);
      return { id: `mfpo-${coeff}-${initial}`, label: `H=${4*coeff} · α₀=${initial}`, color: tempColors[(ci*3+ai)%tempColors.length],
        dash: ci === 0 ? '' : ci === 1 ? '7 3' : '2 3', jobs: members, seeds, expected: 2,
        available: seeds.filter(s => members.some(j => j.seed === s && j.evaluations?.length)).length,
        points: aggregate(members, seeds, options.metric, options.windowSize),
        latestRaw: aggregate(members, seeds, options.metric, 1).at(-1),
        latest5: aggregate(members, seeds, options.metric, 5).at(-1),
        minStep: steps.length ? Math.min(...steps) : 0, maxStep: steps.length ? Math.max(...steps) : 0,
        statuses: members.reduce((a,j) => { a[j.status] = (a[j.status] || 0)+1; return a; }, {}) };
    }));
  }
  const api = { methods, temperatures, metrics, allJobs, selection, aggregate, series, comparison, sweep, mfpoSweep };
  root.FetchAnalytics = api;
  if (typeof module !== 'undefined') module.exports = api;
})(globalThis);
