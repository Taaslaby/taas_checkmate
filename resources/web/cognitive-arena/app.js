const $ = (id) => document.getElementById(id);
const canvas = $('arenaCanvas');
const ctx = canvas.getContext('2d');
let core;
let WasmClasses;
let mission = 'vault';
let platformerCore;
let snapshot;
let playing = false;
let raf;
let lastFrame = 0;
let accumulator = 0;
let training = false;
let trainTimer = null;
let autoRepeat = true;

const el = {
  runtime: $('runtimePill'), status: $('missionStatus'), detail: $('missionDetail'), overlay: $('canvasOverlay'),
  play: $('playButton'), step: $('stepButton'), run: $('runButton'), reset: $('resetButton'), train: $('trainButton'), trainLabel: $('trainButton'), evaluate: $('evaluateButton'), replayBest: $('replayBestButton'),
  speed: $('speedInput'), speedValue: $('speedValue'),
  stepValue: $('stepValue'), retention: $('retentionValue'), uncertainty: $('uncertaintyValue'), messages: $('messageValue'), coordination: $('coordinationValue'), decoy: $('decoyValue'), activity: $('activityValue'), episodesValue: $('episodesValue'), survival: $('survivalValue'), tags: $('tagsValue'),
  episodes: $('episodeCount'), events: $('eventList'),
  tabVault: $('tabVault'), tabHideSeek: $('tabHideSeek'), tabPlatformer: $('tabPlatformer'),
  heroEyebrow: $('heroEyebrow'), heroTitle: $('heroTitle'), heroCopy: $('heroCopy'), arenaTitle: $('arenaTitle'),
  vaultLegend: $('vaultLegend'), hsLegend: $('hsLegend'), hsAgentCards: $('hsAgentCards'),
  scoutCard: $('scoutCard'), carrierCard: $('carrierCard'), trainingNote: $('trainingNote'),
  metricSurvival: $('metricSurvival'), metricTags: $('metricTags'),
  curriculumStage: $('curriculumStage'), curriculumEpisode: $('curriculumEpisode'), curriculumProgress: $('curriculumProgress'),
  trainingSuccess: $('trainingSuccess'), trainingBest: $('trainingBest'), trainingScore: $('trainingScore'), trainingLab: $('trainingLab'), trainingActions: $('trainingActions'),
  evaluationSummary: $('evaluationSummary'), evaluationSuccess: $('evaluationSuccess'), evaluationScore: $('evaluationScore'), evaluationWorst: $('evaluationWorst'),
};

function setRuntime(ready, text) {
  el.runtime.classList.toggle('ready', ready);
  el.runtime.querySelector('span').textContent = text;
}

function applySnapshot(json) {
  const next = typeof json === 'string' ? JSON.parse(json) : json;
  if (!next || !Number.isInteger(next.step)) {
    throw new Error('Unsupported Rust snapshot schema');
  }    if (next.mission === 'platformer') {
    if (next.schema_version !== 3 || !next.platforms || !next.brain || !Number.isFinite(next.threat_density)) throw new Error('Unsupported Platformer schema');
    mission = 'platformer';
  } else if (next.mission === 'hide_seek') {
    if (next.schema_version !== 6) throw new Error('Unsupported Hide&Seek schema');
    mission = 'hide_seek';
  } else {
    if (next.schema_version !== 4 || !next.scout || !next.carrier) throw new Error('Unsupported Vault schema');
    mission = 'vault';
  }
  snapshot = next;
  setMissionUI(mission);
  render(snapshot);
}

function safelyLevelName(s) {
  return s && typeof s.level_name === 'string' ? s.level_name : 'NEON RUN';
}

function setMissionUI(m) {
  const hs = m === 'hide_seek';
  const pf = m === 'platformer';
  el.tabVault.classList.toggle('active', !hs && !pf);
  el.tabHideSeek.classList.toggle('active', hs);
  el.tabPlatformer.classList.toggle('active', pf);
  el.tabVault.setAttribute('aria-selected', String(!hs && !pf));
  el.tabHideSeek.setAttribute('aria-selected', String(hs));
  el.tabPlatformer.setAttribute('aria-selected', String(pf));
  el.vaultLegend.hidden = hs || pf;
  el.hsLegend.hidden = !hs;
  if (pf) {
    el.vaultLegend.hidden = false;
    el.vaultLegend.innerHTML = '<span><i class="dot scout"></i>Runner</span><span><i class="dot carrier"></i>Flag</span><span><i class="dot trap"></i>Enemy</span><span><i class="dot bluekey"></i>Coin</span>';
  } else if (!hs) {
    el.vaultLegend.innerHTML = '<span><i class="dot scout"></i>Scout</span><span><i class="dot carrier"></i>Carrier</span><span><i class="dot key"></i>Red key</span><span><i class="dot bluekey"></i>Blue key</span><span><i class="dot trap"></i>Trap</span>';
  }
  el.scoutCard.hidden = hs;
  el.carrierCard.hidden = hs || pf;
  el.hsAgentCards.hidden = !hs;
  el.metricSurvival.hidden = !hs;
  el.metricTags.hidden = !hs;
  if (el.trainingLab) el.trainingLab.hidden = !pf;
  if (el.trainingActions) el.trainingActions.hidden = !pf;
  if (pf) {
    el.heroEyebrow.textContent = 'AUTONOMOUS MISSION 04 · NEON RUN';
    el.heroTitle.innerHTML = 'Learned timing.<br />Cross the <em>gap.</em>';
    el.heroCopy.textContent = 'A spiking brain controls a platform runner through compact physical sensors. It must accelerate, jump over pits, defeat enemies and reach the flag. The level and physics are owned by Rust.';
    el.arenaTitle.textContent = `${safelyLevelName(snapshot)} · threat ${snapshot?.threat_level ?? 1} · density ${Number(snapshot?.threat_density ?? 1).toFixed(2)} · brain-controlled`;
    el.trainingNote.innerHTML = 'Brain-only control: no reflex jumps, no scripted steering. The network sees an anticipatory field, emits every motor action, and learns from physics consequences. <code>Train ∞</code> rotates the curriculum.';
  } else if (hs) {
    el.heroEyebrow.textContent = 'COMPETITIVE MISSION 03 · HIDE & SEEK';
    el.heroTitle.innerHTML = 'Emergent hide.<br />Learned <em>barricades.</em>';
    el.heroCopy.textContent = 'Two hiders vs four seekers. Vision is honest line-of-sight — walls and boxes are real cover — and teams share pack intel: any spotted agent publishes a fading trail the whole team converges on. No expert script, only reward-modulated STDP.';
    el.arenaTitle.textContent = 'Procedural complex · 48×36 · dynamic light';
    el.trainingNote.innerHTML = 'Hider brains learn survival (flee, shelter, barricade); seeker brains learn chasing and unblocking. <code>Train ∞</code> runs episodes continuously; the curve plots hider survival (steps) and tags per episode.';
  } else {
    el.heroEyebrow.textContent = 'COOPERATIVE MISSION 02 · TWO-KEY VAULT';
    el.heroTitle.innerHTML = 'Dichromatic memory.<br />Synchronized <em>opening.</em>';
    el.heroCopy.textContent = 'Each agent sees only one key color. They must share, remember, coordinate at the vault — and survive the trap. The simulation lives in Rust; this interface only renders its snapshots.';
    el.arenaTitle.textContent = 'Partial observation arena';
    el.trainingNote.innerHTML = 'Reward-modulated STDP (eligibility × dopamine) on the workspace→action weights. <code>Train ∞</code> runs episodes until stopped; watch the curve drop and the W2 heatmaps sharpen.';
  }
  // During training the button keeps its 'Stop training' state; only
  // re-label when idle.
  if (!training) el.train.textContent = 'Train ∞';
}

function rustStep() {
  if (!core) return;
  try {
    applySnapshot(mission === 'platformer' ? platformerCore.step() : (mission === 'hide_seek' ? core.hideSeek.step() : core.step()));
  } catch (error) {
    runtimeError(error);
  }
}

// ── Continuous training ────────────────────────────────────────────────

function activeCore() {
  return mission === 'platformer' ? platformerCore : (mission === 'hide_seek' ? core.hideSeek : core);
}

function startTraining() {
  if (!core) return;
  training = true;
  el.train.textContent = 'Stop training';
  el.train.classList.add('training-active');
  // One episode per tick — fast, but leaves the UI responsive.
  const tick = () => {
    if (!training) return;
    try {
      applySnapshot(activeCore().train_one(mission === 'hide_seek' ? 300 : 800));
    } catch (error) {
      stopTraining();
      runtimeError(error);
      return;
    }
    trainTimer = setTimeout(tick, 0);
  };
  trainTimer = setTimeout(tick, 0);
}

function stopTraining() {
  training = false;
  if (trainTimer) { clearTimeout(trainTimer); trainTimer = null; }
  el.train.textContent = 'Train ∞';
  el.train.classList.remove('training-active');
}

function runtimeError(error) {
  playing = false;
  setRuntime(false, 'rust call failed');
  el.status.textContent = 'ERROR';
  el.status.style.color = 'var(--carrier)';
  el.detail.textContent = error && error.message ? error.message : String(error);
  el.overlay.textContent = 'Rust snapshot error — check console';
  console.error(error);
}

function render(s) {
  el.overlay.textContent = s.finished ? 'mission complete · Rust core' : `Rust snapshot · step ${s.step}`;
  el.status.textContent = s.finished ? 'COMPLETE' : (s.step ? 'RUNNING' : 'READY');
  el.status.style.color = s.finished ? 'var(--green)' : 'var(--amber)';    el.detail.textContent = s.finished ? (mission === 'platformer' ? `flag reached at step ${s.step}` : `vault opened at step ${s.step}`) : (mission === 'hide_seek' ? `visibility ${(Number(s.visibility ?? 1) * 100).toFixed(0)}% · stamina + acoustic trails active` : mission === 'platformer' ? `x ${Number(s.player_x).toFixed(1)} / ${Number(s.goal_x).toFixed(0)} · Rust physics` : 'state owned by CognitiveArenaWasm');
  el.stepValue.textContent = s.step;
  if (mission === 'platformer') {
    // Platformer has a different telemetry contract: never leak Vault's
    // fields into this panel as NaN or stale values when switching missions.
    el.retention.textContent = `${Math.round(Math.min(1, Number(s.player_x) / Math.max(1, Number(s.goal_x))) * 100)}%`;
    el.uncertainty.textContent = Math.abs(Number(s.velocity_y) / Math.max(0.1, Number(s.jump_velocity))).toFixed(2);
    el.messages.textContent = s.coins ?? 0;
    el.coordination.textContent = s.enemies_defeated ?? 0;
    el.decoy.textContent = s.dead ? 1 : 0;
    el.activity.textContent = (s.brain?.spikes || []).reduce((sum, value) => sum + Number(value || 0), 0).toFixed(1);
    el.episodesValue.textContent = s.training_episodes ?? 0;
  } else {
    el.retention.textContent = Number(s.memory_retention).toFixed(3);
    el.uncertainty.textContent = Number(s.belief_uncertainty).toFixed(3);
    el.messages.textContent = s.messages_sent;
    el.coordination.textContent = s.coordination_delay ?? 0;
    el.decoy.textContent = s.decoy_hits ?? 0;
    el.activity.textContent = Number(s.brain_activity ?? 0).toFixed(1);
    el.episodesValue.textContent = (s.training || {}).episodes ?? 0;
  }
  const accessible = $('accessibleSnapshot');
  const t = s.training || {};
  if (mission === 'platformer') {
    $('scoutRole').textContent = 'NEON RUNNER';
    $('scoutPosition').textContent = `(${Number(s.player_x).toFixed(1)}, ${Number(s.player_y).toFixed(1)})`;
    $('scoutAction').textContent = s.dead ? 'dead' : (s.finished ? 'flag reached' : `${s.executed_action || (s.grounded ? 'grounded' : 'airborne')} · brain ${s.brain_action || '—'}`);
    $('scoutConfidence').textContent = `${Math.round(Math.min(1, Math.abs(Number(s.velocity_x)) / 1.55) * 100)}%`;
    $('scoutConfidenceBar').style.width = $('scoutConfidence').textContent;
    $('scoutMemoryAge').textContent = s.coins;
    $('scoutMap').textContent = s.enemies_defeated;
    $('scoutPath').textContent = `${Number(s.player_x).toFixed(1)} / ${Number(s.goal_x).toFixed(0)}`;
    $('scoutCarrying').textContent = `${s.coins} coins`;
    $('scoutPartner').textContent = `${Number(s.velocity_x).toFixed(2)} vx · brain-only`;
    renderBrain('scout', s.brain);
    renderLearning({
      episodes: s.training_episodes || 0,
      rewards: s.training_rewards || [],
      steps: s.training_steps || [],
      curriculum_stage: s.curriculum_stage || 0,
      curriculum_label: s.curriculum_label || 'calibration · learn timing',
      successful_episodes: s.successful_episodes || 0,
      best_steps: s.best_steps || 0,
      best_score: s.best_score || 0,
      evaluation_episodes: s.evaluation_episodes || 0,
      evaluation_successes: s.evaluation_successes || 0,
      evaluation_mean_score: s.evaluation_mean_score || 0,
      evaluation_worst_progress: s.evaluation_worst_progress || 0,
      replaying_best: s.replaying_best || false,
      has_best_checkpoint: s.has_best_checkpoint || false,
    });
    accessible.textContent = `Neon Run step ${s.step}. Player at ${Number(s.player_x).toFixed(1)} of goal ${s.goal_x}. ${s.coins} coins, ${s.enemies_defeated} enemies defeated. ${s.finished ? 'Level complete.' : s.dead ? 'Player dead.' : 'Level active.'}`;
  } else if (mission === 'hide_seek') {
    el.survival.textContent = Number(s.survival ?? 0).toFixed(1);
    el.tags.textContent = s.tags ?? 0;
    renderHsAgents(s);
    renderHsLearning(s.training);
    accessible.textContent = `Hide & Seek step ${s.step}. Visibility ${Math.round(Number(s.visibility ?? 1) * 100)} percent. ${s.hiders.filter(h => h.alive).length} of ${s.hiders.length} hiders alive. ${s.tags} tags. Mean survival ${Number(s.survival ?? 0).toFixed(1)} steps. Brain activity ${Number(s.brain_activity ?? 0).toFixed(1)} spikes per tick. ${s.finished ? 'Round ended.' : 'Round active.'}`;
  } else {
    renderAgent('scout', s.scout);
    renderAgent('carrier', s.carrier);
    renderBrain('scout', s.scout.brain);
    renderBrain('carrier', s.carrier.brain);
    renderLearning(s.training || { episodes: 0, rewards: [], steps: [] });
    accessible.textContent = `Rust arena step ${s.step}. ${s.scout.role} at ${s.scout.x}, ${s.scout.y}, action ${s.scout.action}, carrying ${s.scout.carrying}. ${s.carrier.role} at ${s.carrier.x}, ${s.carrier.y}, action ${s.carrier.action}, carrying ${s.carrier.carrying}. Memory retention ${Number(s.memory_retention).toFixed(2)}. Brain activity ${Number(s.brain_activity ?? 0).toFixed(1)} spikes per tick. Training episodes ${t.episodes ?? 0}. ${s.finished ? 'Vault opened.' : 'Mission active.'}`;
  }
  renderEvents(s.events || []);
  renderArena(s);
  el.play.querySelector('.play-glyph').textContent = playing ? 'Ⅱ' : '▶';
  el.play.childNodes[el.play.childNodes.length - 1].textContent = playing ? ' Pause' : ' Play';
}

function renderAgent(prefix, agent) {
  $(`${prefix}Role`).textContent = agent.role.toUpperCase();
  $(`${prefix}Position`).textContent = `(${agent.x}, ${agent.y})`;
  $(`${prefix}Action`).textContent = agent.action;
  $(`${prefix}Confidence`).textContent = `${Math.round(agent.confidence * 100)}%`;
  $(`${prefix}ConfidenceBar`).style.width = `${Math.round(agent.confidence * 100)}%`;
  $(`${prefix}MemoryAge`).textContent = agent.memory_age;
  $(`${prefix}Map`).textContent = agent.known_obstacles;
  $(`${prefix}Path`).textContent = agent.path_steps;
  const carryingEl = $(`${prefix}Carrying`);
  if (carryingEl) carryingEl.textContent = agent.carrying || 'none';
  const partnerEl = $(`${prefix}Partner`);
  if (partnerEl) partnerEl.textContent = agent.partner_key_known ? 'decoded' : '—';
}

// ── Spiking brain panels ────────────────────────────────────────────────

function setBars(id, rates, color) {
  const host = $(id);
  if (host) setBarsFrom(host, rates, color);
}

function renderHeatmap(id, heat) {
  const cv = $(id);
  if (!cv || !heat || !heat.length) return;
  const c = cv.getContext('2d');
  const rows = 5, cols = Math.round(heat.length / rows);
  const cw = cv.width / cols, ch = cv.height / rows;
  const maxAbs = Math.max(1e-3, ...heat.map(Math.abs));
  c.clearRect(0, 0, cv.width, cv.height);
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      const w = heat[r * cols + col] ?? 0;
      // divergent: negative → cool blue, positive → hot orange
      const t = w / maxAbs;
      const hue = t < 0 ? 210 : 22;
      const light = 18 + Math.abs(t) * 38;
      c.fillStyle = `hsl(${hue} 78% ${light}%)`;
      c.fillRect(col * cw, r * ch, cw + 0.5, ch + 0.5);
    }
  }
}

function renderBrain(prefix, brain) {
  if (!brain) return;
  $(`${prefix}Dopamine`).textContent = brain.dopamine.toFixed(2);
  $(`${prefix}Reward`).textContent = brain.reward.toFixed(2);
  $(`${prefix}Baseline`).textContent = brain.baseline.toFixed(2);
  $(`${prefix}Spikes`).textContent = `${brain.spikes[0]}/${brain.spikes[1]}/${brain.spikes[2]}`;
  const scout = prefix === 'scout';
  setBars(`${prefix}SensorBars`, brain.sensor_rates, scout ? '#a78bfa' : '#f39b67');
  setBars(`${prefix}WorkspaceBars`, brain.workspace_rates, '#54d9f5');
  setBars(`${prefix}ActionBars`, brain.action_rates, '#ffca70');
  setBars(`${prefix}MotorBars`, brain.region_gate || [], '#7ee0a5');
  renderHeatmap(`${prefix}Heatmap`, brain.weight_heat);
  renderRaster(`${prefix}Raster`, brain.spike_history || []);
}

// ── Learning curve ──────────────────────────────────────────────────────

function renderLearning(training) {
  const cv = $('learningCanvas');
  updateTrainingLab(training);
  if (!cv) return;
  const c = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  const padL = 8, padR = 8, padT = 10, padB = 16;
  const innerW = w - padL - padR, innerH = h - padT - padB;
  c.clearRect(0, 0, w, h);
  c.fillStyle = '#0a192d';
  c.fillRect(0, 0, w, h);
  c.strokeStyle = 'rgba(153,190,230,.10)';
  c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (innerH / 4) * i;
    c.beginPath(); c.moveTo(padL, y); c.lineTo(w - padR, y); c.stroke();
  }
  const steps = training.steps || [];
  const rewards = training.rewards || [];
  const episodes = training.episodes || Math.max(steps.length, rewards.length);
  const plotCount = Math.max(steps.length, rewards.length);
  el.episodes.innerHTML = `<i></i> ${episodes} episodes`;
  el.train.disabled = !core;
  if (!episodes || (!steps.length && !rewards.length)) {
    c.fillStyle = '#64809b';
    c.font = '10px DM Mono';
    c.textAlign = 'center';
    c.fillText('press Train — R-STDP episodes will plot here', w / 2, h / 2 + 3);
    return;
  }
  const xAt = (i) => padL + (innerW * i) / Math.max(1, plotCount - 1);
  const yAt = (v, max) => padT + innerH - (innerH * Math.min(1, v / max));
  const line = (values, color, max) => {
    if (!values.length) return;
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.beginPath();
    values.forEach((v, i) => { const x = xAt(i), y = yAt(v, max); i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.stroke();
    values.forEach((v, i) => {
      c.fillStyle = color;
      c.beginPath(); c.arc(xAt(i), yAt(v, max), 2.4, 0, Math.PI * 2); c.fill();
    });
  };
  line(steps, '#65e0b1', Math.max(1, ...steps, 10));
  const scoreAbsMax = Math.max(1, ...rewards.map((value) => Math.abs(value)), 1);
  const zeroY = padT + innerH / 2;
  c.strokeStyle = 'rgba(255,202,112,.28)';
  c.setLineDash([3, 3]);
  c.beginPath(); c.moveTo(padL, zeroY); c.lineTo(w - padR, zeroY); c.stroke();
  c.setLineDash([]);
  const scoreY = (value) => zeroY - (innerH * value) / (2 * scoreAbsMax);
  if (rewards.length) {
    c.strokeStyle = '#ffca70';
    c.lineWidth = 2;
    c.beginPath();
    rewards.forEach((value, i) => {
      const x = xAt(i), y = scoreY(Number(value) || 0);
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
  }
  c.fillStyle = '#6d88a3';
  c.font = '9px DM Mono';
  c.textAlign = 'right';
  const lastSteps = steps.length ? steps[steps.length - 1] : 0;
  const lastReward = rewards.length ? rewards[rewards.length - 1] : 0;
  c.fillText(`${episodes} ep · last ${lastSteps} steps`, w - padR, h - 4);
}

function updateTrainingLab(training) {
  const episodes = Number(training.episodes || 0);
  const stage = Math.min(3, Number(training.curriculum_stage || 0));
  const success = episodes ? Math.round((Number(training.successful_episodes || 0) / episodes) * 100) : 0;
  if (el.curriculumStage) el.curriculumStage.textContent = training.curriculum_label || 'calibration · learn timing';
  if (el.curriculumEpisode) el.curriculumEpisode.textContent = `episode ${episodes}`;
  if (el.curriculumProgress) el.curriculumProgress.style.width = `${Math.min(100, (stage / 3) * 100 + (episodes % 3) * 11)}%`;
  if (el.trainingSuccess) el.trainingSuccess.textContent = `${success}%`;
  if (el.trainingBest) el.trainingBest.textContent = training.best_steps ? `${training.best_steps} steps` : (training.has_best_checkpoint ? 'checkpoint' : '—');
  const bestScore = Number(training.best_score);
  if (el.trainingScore) el.trainingScore.textContent = Number.isFinite(bestScore) ? bestScore.toFixed(1) : '0.0';
  const evalEpisodes = Number(training.evaluation_episodes || 0);
  const evalSuccesses = Number(training.evaluation_successes || 0);
  if (el.evaluationSummary) el.evaluationSummary.textContent = evalEpisodes ? `${evalSuccesses}/${evalEpisodes} profiles passed` : 'not evaluated';
  if (el.evaluationSuccess) el.evaluationSuccess.textContent = evalEpisodes ? `${Math.round((evalSuccesses / evalEpisodes) * 100)}%` : '—';
  if (el.evaluationScore) el.evaluationScore.textContent = evalEpisodes ? Number(training.evaluation_mean_score || 0).toFixed(1) : '—';
  if (el.evaluationWorst) el.evaluationWorst.textContent = evalEpisodes ? `${Math.round(Number(training.evaluation_worst_progress || 0) * 100)}%` : '—';
  if (el.evaluate) el.evaluate.disabled = !core || mission !== 'platformer';
  if (el.replayBest) el.replayBest.disabled = !core || mission !== 'platformer' || !training.has_best_checkpoint;
}

// ── Hide & Seek rendering ─────────────────────────────────────────────

function hsCardTemplate(agent, colorVar, avatar, label) {
  return `<article class="panel agent-card hs-card" style="--hs:var(--${colorVar})">
    <div class="agent-heading"><span class="agent-avatar hs-avatar">${avatar}</span><div><span class="eyebrow">${label}</span><h3>${escapeHtml(agent.team.toUpperCase())} ${agent.id + 1}</h3></div>${agent.alive ? '<span class="live-badge">ALIVE</span>' : '<span class="live-badge dead-badge">TAGGED</span>'}</div>      <div class="agent-position"><span>POSITION</span><strong>(${agent.x}, ${agent.y})</strong><span class="action-chip">${escapeHtml(agent.action)}</span></div>
      <div class="hs-telemetry"><span>STAMINA <b>${Math.round(Number(agent.energy ?? 1) * 100)}%</b></span><span>NOISE <b>${Math.round(Number(agent.noise ?? 0) * 100)}%</b></span></div>
      <div class="brain-block">
      <div class="brain-head"><span>SPIKING GNW BRAIN · POPULATION RASTER</span><span class="dop">dopamine <b>${Number(agent.brain.dopamine).toFixed(2)}</b></span></div>
      <div class="rate-row"><label>SENS</label><div class="bars hs-sens" data-rates='${JSON.stringify(agent.brain.sensor_rates)}'></div></div>
      <div class="rate-row"><label>GNW</label><div class="bars hs-gnw" data-rates='${JSON.stringify(agent.brain.workspace_rates)}'></div></div>
      <div class="rate-row"><label>ACT</label><div class="bars action-bars" data-rates='${JSON.stringify(agent.brain.action_rates)}'></div></div>
      <div class="rate-row"><label>MOT</label><div class="bars motor-bars" data-region='${JSON.stringify(agent.brain.region_gate || [])}' title="GNW motor-access gate: which brain region wins the motors"></div></div>
      <canvas class="heatmap hs-heat" width="256" height="24" data-heat='${JSON.stringify(agent.brain.weight_heat)}'></canvas>
      <canvas class="raster hs-raster" width="256" height="58" data-raster='${JSON.stringify(agent.brain.spike_history || [])}' aria-label="${label} spike raster plot"></canvas>
      <div class="brain-stats"><span>reward <b>${Number(agent.brain.reward).toFixed(2)}</b></span><span>baseline <b>${Number(agent.brain.baseline).toFixed(2)}</b></span><span>spikes <b>${agent.brain.spikes.join('/')}</b></span></div>
    </div>
  </article>`;
}

function renderHsAgents(s) {
  const cards = [
    ...s.hiders.map((a, i) => hsCardTemplate(a, 'scout', 'H', 'HIDER')),
    ...s.seekers.map((a, i) => hsCardTemplate(a, 'carrier', 'S', 'SEEKER')),
  ];
  el.hsAgentCards.innerHTML = cards.join('');
  el.hsAgentCards.querySelectorAll('.hs-card').forEach((card, i) => {
    // Only rate bars carry data-rates; the MOTOR bar carries data-region and
    // is handled separately below.
    card.querySelectorAll('.bars[data-rates]').forEach((host) => {
      const rates = JSON.parse(host.dataset.rates);
      const color = host.classList.contains('hs-sens') ? (i < s.hiders.length ? 'var(--scout)' : 'var(--carrier)') : (host.classList.contains('hs-gnw') ? '#54d9f5' : '#ffca70');
      setBarsFrom(host, rates, color);
    });
    const motor = card.querySelector('.motor-bars');
    const region = JSON.parse(motor.dataset.region || '[]');
    setBarsFrom(motor, region, '#7ee0a5');
    const heat = JSON.parse(card.querySelector('.hs-heat').dataset.heat);
    renderHeatmapInto(card.querySelector('.hs-heat'), heat);
    const raster = JSON.parse(card.querySelector('.hs-raster').dataset.raster || '[]');
    renderRasterInto(card.querySelector('.hs-raster'), raster);
  });
}

function setBarsFrom(host, rates, color) {
  const count = rates.length;
  while (host.children.length < count) {
    const bar = document.createElement('i');
    bar.style.background = color;
    host.appendChild(bar);
  }
  while (host.children.length > count) host.removeChild(host.lastChild);
  const max = Math.max(1e-3, ...rates);
  for (let i = 0; i < count; i++) {
    host.children[i].style.height = `${Math.max(3, (rates[i] / max) * 100)}%`;
  }
}

function renderRaster(id, history) {
  const cv = $(id);
  if (cv) renderRasterInto(cv, history);
}

function renderRasterInto(cv, history) {
  const c = cv.getContext('2d');
  const width = cv.width, height = cv.height;
  c.clearRect(0, 0, width, height);
  c.fillStyle = '#071321'; c.fillRect(0, 0, width, height);
  c.strokeStyle = 'rgba(153,190,230,.11)'; c.lineWidth = 1;
  for (let row = 1; row < 3; row++) {
    const y = (height / 3) * row;
    c.beginPath(); c.moveTo(0, y); c.lineTo(width, y); c.stroke();
  }
  c.font = '8px DM Mono'; c.textBaseline = 'middle'; c.textAlign = 'left';
  const labels = [['SENS','#a78bfa'], ['GNW','#54d9f5'], ['OUT','#ffca70']];
  labels.forEach(([label, color], i) => {
    c.fillStyle = color; c.fillText(label, 4, (height / 3) * (i + .5));
  });
  if (!history || !history.length) return;
  const columns = history.length;
  const left = 32, plotWidth = width - left - 4;
  const max = Math.max(1, ...history.flatMap(row => row.map(Number)));
  history.forEach((row, t) => row.forEach((value, group) => {
    const activity = Math.max(0, Number(value) || 0) / max;
    if (activity <= 0) return;
    const x = left + ((t + .5) / columns) * plotWidth;
    const y = (height / 3) * (group + .5);
    const radius = 1 + activity * 2.1;
    c.fillStyle = labels[group][1];
    c.globalAlpha = .35 + activity * .65;
    c.beginPath(); c.arc(x, y, radius, 0, Math.PI * 2); c.fill();
  }));
  c.globalAlpha = 1;
}

function renderHeatmapInto(cv, heat) {
  if (!heat || !heat.length) return;
  const c = cv.getContext('2d');
  const rows = 5, cols = Math.round(heat.length / rows);
  const cw = cv.width / cols, ch = cv.height / rows;
  const maxAbs = Math.max(1e-3, ...heat.map(Math.abs));
  c.clearRect(0, 0, cv.width, cv.height);
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      const w = heat[r * cols + col] ?? 0;
      const t = w / maxAbs;
      const hue = t < 0 ? 210 : 22;
      c.fillStyle = `hsl(${hue} 78% ${18 + Math.abs(t) * 38}%)`;
      c.fillRect(col * cw, r * ch, cw + 0.5, ch + 0.5);
    }
  }
}

function renderHsLearning(training) {
  const cv = $('learningCanvas');
  if (!cv) return;
  const c = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  const padL = 8, padR = 8, padT = 10, padB = 16;
  const innerW = w - padL - padR, innerH = h - padT - padB;
  c.clearRect(0, 0, w, h);
  c.fillStyle = '#0a192d';
  c.fillRect(0, 0, w, h);
  c.strokeStyle = 'rgba(153,190,230,.10)';
  c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (innerH / 4) * i;
    c.beginPath(); c.moveTo(padL, y); c.lineTo(w - padR, y); c.stroke();
  }
  const survival = (training && training.survival) || [];
  const tags = (training && training.tags) || [];
  const episodes = (training && training.episodes) || Math.max(survival.length, tags.length);
  const plotCount = Math.max(survival.length, tags.length);
  el.episodes.innerHTML = `<i></i> ${episodes} episodes`;
  if (!episodes || (!survival.length && !tags.length)) {
    c.fillStyle = '#64809b';
    c.font = '10px DM Mono';
    c.textAlign = 'center';
    c.fillText('press Train — hider survival will plot here', w / 2, h / 2 + 3);
    return;
  }
  const xAt = (i) => padL + (innerW * i) / Math.max(1, plotCount - 1);
  const yAt = (v, max) => padT + innerH - (innerH * Math.min(1, v / max));
  const line = (values, color, max) => {
    if (!values.length) return;
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.beginPath();
    values.forEach((v, i) => { const x = xAt(i), y = yAt(v, max); i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.stroke();
    values.forEach((v, i) => {
      c.fillStyle = color;
      c.beginPath(); c.arc(xAt(i), yAt(v, max), 2.4, 0, Math.PI * 2); c.fill();
    });
  };
  line(survival, '#65e0b1', Math.max(1, ...survival, 10));
  line(tags, '#ffca70', Math.max(1, ...tags, 1));
  c.fillStyle = '#6d88a3';
  c.font = '9px DM Mono';
  c.textAlign = 'right';
  const last = survival.length ? survival[survival.length - 1] : 0;
  c.fillText(`${episodes} ep · last survival ${Number(last).toFixed(0)}`, w - padR, h - 4);
}

function renderEvents(events) {
  if (!events.length) {
    el.events.innerHTML = '<div class="empty-event">Rust events will appear here…</div>';
    return;
  }
  el.events.innerHTML = events.slice(-9).map((event, i) => `<div class="event"><span class="time">${String(i + 1).padStart(2, '0')}</span>${escapeHtml(event)}</div>`).join('');
  el.events.scrollTop = el.events.scrollHeight;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function renderArena(s) {
  if (mission === 'platformer') { renderPlatformerArena(s); return; }
  if (mission === 'hide_seek') { renderHsArena(s); return; }
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 960;
  const height = rect.height || width * .625;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cell = Math.min(width / s.width, height / s.height);
  const ox = (width - cell * s.width) / 2, oy = (height - cell * s.height) / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#081727'; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(94,145,186,.10)'; ctx.lineWidth = 1;
  for (let x = 0; x <= s.width; x++) { ctx.beginPath(); ctx.moveTo(ox+x*cell,oy);ctx.lineTo(ox+x*cell,oy+s.height*cell);ctx.stroke(); }
  for (let y = 0; y <= s.height; y++) { ctx.beginPath(); ctx.moveTo(ox,oy+y*cell);ctx.lineTo(ox+s.width*cell,oy+y*cell);ctx.stroke(); }
  const tile = (x, y, fill, radius = 3) => { ctx.fillStyle=fill;ctx.beginPath();ctx.roundRect(ox+x*cell+2,oy+y*cell+2,cell-4,cell-4,radius);ctx.fill(); };
  const known = new Set([...(s.scout_known_obstacles || []), ...(s.carrier_known_obstacles || [])].map(([x,y]) => `${x},${y}`));
  for (const [x,y] of s.obstacles) tile(x,y,known.has(`${x},${y}`) ? '#28435d' : '#162a40',2);
  const keyMarker = (pos, fill, glyph) => { tile(pos[0],pos[1],fill,5); ctx.fillStyle='#fff';ctx.font=`${Math.max(10,cell*.55)}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(glyph,ox+(pos[0]+.5)*cell,oy+(pos[1]+.5)*cell); };
  keyMarker(s.red_key,'#a34a4a','◆');
  keyMarker(s.blue_key,'#3f6fb5','◆');
  const trapX=ox+(s.decoy[0]+.5)*cell, trapY=oy+(s.decoy[1]+.5)*cell;
  ctx.save(); ctx.strokeStyle='#c26a3f'; ctx.lineWidth=2; ctx.beginPath();
  ctx.moveTo(trapX,trapY-cell*.32); ctx.lineTo(trapX+cell*.28,trapY+cell*.3); ctx.lineTo(trapX-cell*.28,trapY+cell*.3); ctx.closePath(); ctx.stroke(); ctx.restore();
  tile(s.door[0],s.door[1],s.finished?'#327b68':'#463e72',4);ctx.fillStyle='#d9cbff';ctx.fillText(s.finished?'✓':'▣',ox+(s.door[0]+.5)*cell,oy+(s.door[1]+.5)*cell);
  drawAgent(s.scout, '#a78bfa', 'A', ox, oy, cell, s.width, s.height);
  drawAgent(s.carrier, '#f39b67', 'B', ox, oy, cell, s.width, s.height);
}

function renderPlatformerArena(s) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 960;
  const height = rect.height || width * .5;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#071321'; ctx.fillRect(0, 0, width, height);
  const viewWidth = 42;
  const camera = Math.max(0, Math.min(Number(s.width) - viewWidth, Number(s.player_x) - 10));
  const sx = width / viewWidth, sy = height / Number(s.height);
  const wx = (x) => (x - camera) * sx;
  const wy = (y) => height - y * sy;
  ctx.strokeStyle = 'rgba(94,145,186,.10)'; ctx.lineWidth = 1;
  for (let x = Math.ceil(camera); x <= camera + viewWidth; x += 2) { ctx.beginPath(); ctx.moveTo(wx(x), 0); ctx.lineTo(wx(x), height); ctx.stroke(); }
  for (let y = 0; y <= Number(s.height); y += 2) { ctx.beginPath(); ctx.moveTo(0, wy(y)); ctx.lineTo(width, wy(y)); ctx.stroke(); }
  for (const p of s.platforms || []) {
    ctx.fillStyle = p.y > 0 ? '#345d72' : '#27445b';
    ctx.fillRect(wx(p.x), wy(p.y + p.height), p.width * sx, p.height * sy);
    ctx.fillStyle = p.y > 0 ? '#65e0b1' : '#4b91a8';
    ctx.fillRect(wx(p.x), wy(p.y + p.height), p.width * sx, Math.max(2, sy * .08));
  }
  for (const c of s.coin_items || []) if (!c.collected && c.x >= camera && c.x <= camera + viewWidth) {
    ctx.fillStyle = '#ffca70'; ctx.shadowColor = '#ffca70'; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(wx(c.x), wy(c.y), Math.max(3, sx * .16), 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
  }
  for (const e of s.enemies || []) if (e.alive && e.x >= camera - 1 && e.x <= camera + viewWidth + 1) {
    ctx.fillStyle = '#e0634a'; ctx.shadowColor = '#e0634a'; ctx.shadowBlur = 12;
    ctx.beginPath(); ctx.arc(wx(e.x), wy(e.y), Math.max(5, sx * .22), 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
  }
  ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 2; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(wx(s.goal_x), 12); ctx.lineTo(wx(s.goal_x), 0); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = s.finished ? '#65e0b1' : '#a78bfa'; ctx.fillRect(wx(s.goal_x), wy(10.5), Math.max(14, sx * 1.6), sy * .65);
  const px = wx(s.player_x), py = wy(s.player_y);
  ctx.save(); ctx.fillStyle = s.dead ? '#6d5660' : '#f39b67'; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 16;
  ctx.beginPath(); ctx.arc(px, py, Math.max(7, sx * .28), 0, Math.PI * 2); ctx.fill(); ctx.restore();
  ctx.fillStyle = '#9ab2cb'; ctx.font = '10px DM Mono'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(`${safelyLevelName(s)}  ·  ${s.finished ? 'FLAG REACHED' : s.dead ? 'RETRY REQUIRED' : `x ${Number(s.player_x).toFixed(1)} / ${Number(s.goal_x).toFixed(0)}`}`, 12, 12);
}

function renderHsArena(s) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 960;
  const height = rect.height || width * .5;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cell = Math.min(width / s.width, height / s.height);
  const ox = (width - cell * s.width) / 2, oy = (height - cell * s.height) / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#081727'; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(94,145,186,.10)'; ctx.lineWidth = 1;
  for (let x = 0; x <= s.width; x++) { ctx.beginPath(); ctx.moveTo(ox+x*cell,oy);ctx.lineTo(ox+x*cell,oy+s.height*cell);ctx.stroke(); }
  for (let y = 0; y <= s.height; y++) { ctx.beginPath(); ctx.moveTo(ox,oy+y*cell);ctx.lineTo(ox+s.width*cell,oy+y*cell);ctx.stroke(); }
  const tile = (x, y, fill, radius = 1) => { ctx.fillStyle=fill;ctx.beginPath();ctx.roundRect(ox+x*cell+1,oy+y*cell+1,cell-2,cell-2,radius);ctx.fill(); };
  const wallSet = new Set(s.walls.map(([x,y]) => `${x},${y}`));
  for (const [x,y] of s.walls) tile(x,y,'#162a40',1);
  // Doorways: subtle glowing gaps.
  for (const [x,y] of s.doorways) tile(x,y,'#2c4663',1);
  // Shelters: dim safe markers.
  for (const [x,y] of s.shelters) { tile(x,y,'#1e3a4f',3); }
  // Boxes: bright, pushable.
  for (const [x,y] of s.boxes) {
    tile(x,y,'#c98a3f',2);
    ctx.fillStyle='#ffd9a0';ctx.font=`${Math.max(8,cell*.5)}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText('▣',ox+(x+.5)*cell,oy+(y+.5)*cell);
  }
  // Sight rings around seekers (red) and hiders (blue).
  for (const a of s.seekers) {
    ctx.save(); ctx.strokeStyle='rgba(243,155,103,.28)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(ox+(a.x+.5)*cell, oy+(a.y+.5)*cell, cell * 13 * Number(s.visibility ?? 1), 0, Math.PI*2); ctx.stroke(); ctx.restore();
  }
  drawHsAgents(s.hiders, '#5c8fdc', 'H', ox, oy, cell);
  drawHsAgents(s.seekers, '#e0634a', 'S', ox, oy, cell);
  // Barricade badge: box sitting on a doorway.
  const doorSet = new Set(s.doorways.map(([x,y]) => `${x},${y}`));
  for (const [x,y] of s.boxes) {
    if (doorSet.has(`${x},${y}`)) {
      tile(x,y,'#7a4f8f',2);
      ctx.fillStyle='#e6ccff';ctx.font=`${Math.max(8,cell*.5)}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText('⛌',ox+(x+.5)*cell,oy+(y+.5)*cell);
    }
  }
  // Dynamic dusk is visible in the world, not only in the status text. The
  // Rust visibility value is deterministic, so this overlay never affects
  // simulation—only tells the operator why sight ranges are currently short.
  const visibility = Math.max(0, Math.min(1, Number(s.visibility ?? 1)));
  const dim = (1 - visibility) * 0.42;
  if (dim > 0.001) {
    ctx.fillStyle = `rgba(4, 8, 18, ${dim})`;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.fillStyle = visibility < 0.72 ? '#b8aaff' : '#8fd6b8';
  ctx.font = '10px DM Mono';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(`LIGHT ${Math.round(visibility * 100)}%`, 12, 12);
}

function drawHsAgents(agents, color, letter, ox, oy, cell) {
  for (const a of agents) {
    const x = ox + (a.x + .5) * cell, y = oy + (a.y + .5) * cell;
    ctx.save();
    if (!a.alive) ctx.globalAlpha = .25;
    ctx.shadowColor = color; ctx.shadowBlur = 12;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, Math.max(4, cell * .3), 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#08111f'; ctx.font = `700 ${Math.max(8, cell * .34)}px DM Mono`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(letter, x, y);
    ctx.restore();
  }
}

function drawAgent(agent, color, letter, ox, oy, cell, width, height) {
  const x=ox+(agent.x+.5)*cell,y=oy+(agent.y+.5)*cell;
  ctx.save(); ctx.globalAlpha=.10;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,cell*5.2,0,Math.PI*2);ctx.fill();
  ctx.globalAlpha=1;ctx.shadowColor=color;ctx.shadowBlur=14;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,Math.max(5,cell*.34),0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#08111f';ctx.font=`700 ${Math.max(9,cell*.38)}px DM Mono`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(letter,x,y);ctx.restore();
}

function loop(time) {
  if (!playing) return;
  if (!lastFrame) lastFrame=time;
  accumulator += (time-lastFrame) * Number(el.speed.value) / 1000;
  lastFrame=time;    const isFinished = () => mission === 'platformer' ? platformerCore.finished() : (mission === 'hide_seek' ? core.hideSeek.finished() : core.finished());
  while (accumulator >= .045) {
    if (isFinished()) {
      // Auto-repeat: the round ended — start a fresh one with the trained
      // brains preserved (unless the user turned auto-repeat off).
      if (!autoRepeat) { playing = false; break; }
      try {
        applySnapshot(activeCore().next_round());
      } catch (error) {
        runtimeError(error);
        return;
      }
      accumulator = 0;
      break;
    }
    try {
      rustStep();
    } catch (error) {
      runtimeError(error);
      return;
    }
    accumulator -= .045;
  }
  if (isFinished() && !autoRepeat) playing = false;
  raf=requestAnimationFrame(loop);
}

function switchMission(m) {
  if (mission === m) return;
  playing = false;
  mission = m;
  if (core) {
    try {
      const json = m === 'platformer' ? platformerCore.snapshot() : ((m === 'hide_seek') ? core.hideSeekSnapshot() : core.snapshot());
      applySnapshot(json);
    } catch (error) {
      runtimeError(error);
    }
  }
}

el.tabVault.addEventListener('click', () => switchMission('vault'));
el.tabHideSeek.addEventListener('click', () => switchMission('hide_seek'));
el.tabPlatformer.addEventListener('click', () => switchMission('platformer'));
el.play.addEventListener('click', () => {
  playing=!playing; lastFrame=0;
  if (playing) {
    // Play on a finished arena starts a fresh round (trained brains kept)
    // so the game never locks after a mission or training.
    if (mission === 'hide_seek' ? core.hideSeek.finished() : core.finished()) {
      try {
        applySnapshot(activeCore().next_round());
      } catch (error) { runtimeError(error); return; }
    }
    raf=requestAnimationFrame(loop);
  }
  render(snapshot);
});
el.step.addEventListener('click', () => { playing=false; rustStep(); });
el.run.addEventListener('click', () => { playing=false; try { applySnapshot(activeCore().run(500)); } catch (error) { runtimeError(error); } });
el.train.addEventListener('click', () => {
  playing = false;
  if (training) { stopTraining(); }
  else { startTraining(); }
});
el.evaluate.addEventListener('click', () => {
  if (mission !== 'platformer' || !platformerCore) return;
  playing = false;
  try { applySnapshot(platformerCore.evaluate(8, 900)); } catch (error) { runtimeError(error); }
});
el.replayBest.addEventListener('click', () => {
  if (mission !== 'platformer' || !platformerCore) return;
  playing = false;
  try { applySnapshot(platformerCore.replay_best()); } catch (error) { runtimeError(error); }
});
el.reset.addEventListener('click', () => { playing=false; try { applySnapshot(activeCore().reset()); } catch (error) { runtimeError(error); } });
el.speed.addEventListener('input', () => { el.speedValue.textContent=`${el.speed.value}×`; });
window.addEventListener('keydown', (event) => { if(event.key==='ArrowRight') { event.preventDefault(); playing=false; rustStep(); } if(event.key===' ') { event.preventDefault(); el.play.click(); } });
window.addEventListener('resize', () => snapshot && renderArena(snapshot));

async function boot() {
  try {
    const mod = await import('./pkg/taas_checkmate.js');
    const init = mod.default;
    await init();
    WasmClasses = mod;
  core = new mod.CognitiveArenaWasm();
  // Bridge: the missions share one browser shell but retain independent Rust state.
  core.hideSeek = new mod.HideSeekWasm();
  platformerCore = new mod.PlatformerWasm();
  core.hideSeekSnapshot = () => core.hideSeek.snapshot();
  const autoEl = $('autoRepeatInput');
  if (autoEl) {
    autoRepeat = autoEl.checked;
    autoEl.addEventListener('change', () => { autoRepeat = autoEl.checked; });
  }
  setRuntime(true, 'rust wasm connected');
    [el.play,el.step,el.run,el.reset,el.train,el.evaluate,el.replayBest].forEach(button => button.disabled=false);
    applySnapshot(core.snapshot());
  } catch (error) {
    setRuntime(false, 'wasm unavailable');
    el.status.textContent='OFFLINE'; el.status.style.color='var(--carrier)';
    el.detail.textContent = error && error.message ? error.message : String(error);
    el.overlay.textContent='Build wasm package and serve this directory';
    console.error(error);
  }
}

boot();
