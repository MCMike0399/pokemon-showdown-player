/* Read-only public battle renderer. No login, account creation, chat or choices. */
const $id = id => document.getElementById(id);
let follow = true, shown = null, latest = null, battle = null, stream = null;
let replay = false, relayLog = [], recentSignature = null;
let retryTimer = null, liveTimer = null;
let streamBattle = null;
let alternateArtwork = new Set();
const roomPattern = /^battle-[a-z0-9]+-[0-9]+(?:-[a-z0-9]+)?$/;
const battleURL = room => 'https://play.pokemonshowdown.com/' + room;
Dex.resourcePrefix = 'https://play.pokemonshowdown.com/';
Dex.fxPrefix = Dex.resourcePrefix + 'fx/';
BattleSound.setMute(true);

// Playback diagnostics: a bounded in-page ring (window.viewerDiag) plus batched
// reports to /api/diag, which the server appends to live-watch-client.jsonl.
// ?debug=1 also shows the counters over the arena.
const debugView = /[?&]debug(?:[=&]|$)/.test(globalThis.location?.search || '');
const diagStart = Date.now();
const diagSession = Math.random().toString(36).slice(2, 10);
const diagEvents = [], diagOutbox = [];
const diagCounts = {frames: 0, seeks: 0, skippedSteps: 0, resets: 0, jank: 0, reconnects: 0};
window.viewerDiag = {session: diagSession, events: diagEvents, counts: diagCounts};
function diag(event, data = {}) {
  const record = {t: Date.now() - diagStart, event, room: shown, turn: battle?.turn, ...data};
  diagEvents.push(record);
  if (diagEvents.length > 400) diagEvents.shift();
  if (diagOutbox.length < 300) diagOutbox.push(record);
}
function flushDiag() {
  if (!diagOutbox.length || !window.fetch) return;
  fetch('/api/diag', {method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session: diagSession, events: diagOutbox.splice(0)})}).catch(() => {});
}
window.setInterval?.(flushDiag, 5000);
// Main-thread stalls show up as long gaps between animation frames.
let lastFrame = 0;
function watchFrames(now) {
  if (lastFrame && !document.hidden && now - lastFrame > 120) {
    diagCounts.jank++;
    diag('jank', {gap: Math.round(now - lastFrame), animating: !!battle && !battle.atQueueEnd});
  }
  lastFrame = document.hidden ? 0 : now;
  window.requestAnimationFrame(watchFrames);
}
window.requestAnimationFrame?.(watchFrames);
if (debugView) {
  const overlay = document.createElement('pre');
  overlay.id = 'diag-overlay';
  overlay.style.cssText = 'position:absolute;left:6px;bottom:6px;z-index:50;margin:0;padding:6px 8px;' +
    'font:11px/1.35 ui-monospace,monospace;background:rgba(0,0,0,.72);color:#9fe;border-radius:6px;' +
    'pointer-events:none;max-width:calc(100% - 12px);white-space:pre-wrap';
  $id('arena-wrap').append(overlay);
  window.setInterval(() => {
    const pending = battle ? battle.stepQueue.length - battle.currentStep : 0;
    const last = diagEvents.filter(e => e.event === 'drained').slice(-1)[0];
    overlay.textContent = 'turn ' + (battle?.turn ?? '—') + ' / live ' + (relayLast?.turn ?? '—') +
      '  pending ' + pending + '  speed ' + (battle?.scene?.acceleration ?? '—') + 'x\n' +
      Object.entries(diagCounts).map(([k, v]) => k + ' ' + v).join('  ') +
      (last ? '\nlast turn played in ' + (last.ms / 1000).toFixed(1) + 's' : '');
  }, 250);
}

function noteAlternateArtwork(species) {
  alternateArtwork.add(species);
  $id('sprite-notice').textContent = 'Alternate artwork for ' + [...alternateArtwork].join(', ') + '.';
  $id('sprite-notice').classList.remove('hidden');
}

// These forms have no art on the asset host yet. Resolve their related artwork
// through the same Dex API so animated GIFs and their dimensions are preserved.
const spriteData = Dex.getSpriteData.bind(Dex);
const relatedArtwork = {'garchomp-megaz': 'Garchomp-Mega', 'golisopod-mega': 'Golisopod'};
Dex.getSpriteData = (pokemon, isFront, options = {}) => {
  const original = spriteData(pokemon, isFront, options);
  const file = original.url.split('/').pop().split('.')[0];
  const related = relatedArtwork[file];
  if (!related) return original;
  noteAlternateArtwork(Dex.species.get(file.replaceAll('-', '')).name);
  return {...spriteData(related, isFront, options), cryurl: original.cryurl};
};

// Some newer Mega forms lack sprites on Showdown's asset host. Keep the
// battler visible with available related artwork and label the substitution.
document.addEventListener('error', event => {
  const sprite = event.target;
  if (sprite.tagName !== 'IMG' || !sprite.closest('#battle')) return;
  const url = new URL(sprite.src);
  if (url.hostname !== 'play.pokemonshowdown.com') return;
  const name = url.pathname.split('/').pop();
  if (!/-mega[a-z]?\.(png|gif)$/.test(name)) return;
  const fallback = name.startsWith('garchomp-megaz.')
    ? name.replace('garchomp-megaz.', 'garchomp-mega.')
    : name.replace(/-mega[a-z]?(\.(png|gif))$/, '$1');
  const species = Dex.species.get(name.split('.')[0].replaceAll('-', '')).name;
  sprite.src = sprite.src.replace(name, fallback);
  sprite.title = 'Alternate artwork for ' + species;
  noteAlternateArtwork(species);
}, true);

function resizeBattle() {
  const width = $id('arena-wrap').clientWidth;
  const scale = Math.min(width / 640, 1.8);
  $id('battle').style.transform = 'scale(' + scale + ')';
  $id('arena-wrap').style.height = Math.ceil(360 * scale) + 'px';
}
new ResizeObserver(resizeBattle).observe($id('arena-wrap'));

function newBattle(room, side) {
  if (battle) { battle.pause(); battle.destroy(); }
  jQuery('#battle,#battle-log').empty();
  alternateArtwork.clear();
  $id('sprite-notice').classList.add('hidden');
  battle = new Battle({id: room, $frame: jQuery('#battle'), $logFrame: jQuery('#battle-log'), paused: true,
    subscription: playbackState});
  battle.roomid = room;
  battle.joinButtons = false;
  // Showdown's own fast animation mode retains move effects at twice the pace.
  battle.messageFadeTime = baseFadeTime;
  hurrying = false;
  if (side) battle.setViewpoint(side);
  resizeBattle();
}

let relayCount = 0, relayRoom = null, relayLast = null;

function rendererLog(lines) {
  // Older filtered relays omit the argument-free |start event. Showdown needs
  // it to clear team-preview sprites before displaying the active battlers.
  let started = false;
  return lines.flatMap(line => {
    if (line === '|start' || line.startsWith('|start|')) started = true;
    if (!started && /^\|(switch|drag|replace)\|/.test(line)) {
      started = true;
      return ['|start', line];
    }
    return [line];
  });
}

function applyBattle(data) {
  if (!data.room || !data.log?.length) {
    $id('viewer-status').textContent = 'Waiting for the player relay…';
    return;
  }
  if (data.room !== shown) return; // A previous stream cannot overwrite a selected replay.
  const log = rendererLog(data.log);
  const reset = relayRoom !== data.room || log.length < relayCount ||
    relayLog.some((line, index) => line !== log[index]);
  if (reset) {
    const first = relayLog.findIndex((line, index) => line !== log[index]);
    if (battle) diagCounts.resets++;
    diag('reset', {reason: relayRoom !== data.room ? 'room' : log.length < relayCount ? 'shrink' : 'rewrite',
      at: first, lines: log.length});
    newBattle(data.room, data.side);
    relayCount = 0; relayRoom = data.room;
  }
  if (log.length > relayCount) {
    diagCounts.frames++;
    drainSince ??= Date.now();
    diag('frame', {appended: log.length - relayCount, liveTurn: data.turn, live: data.live,
      pending: battle.stepQueue.length - battle.currentStep});
    battle.addBatch(log.slice(relayCount));
  }
  relayCount = log.length; relayLog = log; relayLast = data;
  // Entering a game shows the current position instantly. A reconnect to the
  // game already on screen resumes in place: live turns play out in full and
  // pace() only speeds up or skips real backlog.
  if (!replay && reset) catchUp('reset');
  // addBatch resumes playback itself. Starting a second loop causes scene errors.
  if (reset) battle.play();
  pace();
  updateViewerStatus();
}

// Live pacing. A frame normally carries one turn (its actions end with the next
// |turn| line), so the renderer is one turn behind while it animates. Queued
// turns play at hyperfast speed; only with four or more queued does it jump to
// the start of the newest turn, which still animates.
const baseFadeTime = 100, hurryFadeTime = 40;
let hurrying = false;
function pace() {
  if (!battle || replay || !relayLast) return setHurry(false);
  const behind = () => Number(relayLast.turn || 0) - Math.max(battle.turn, 0);
  if (behind() >= 4 && !document.hidden) catchUp('backlog', relayLast.turn - 1);
  setHurry(behind() >= 2);
}
function setHurry(on) {
  if (!battle || hurrying === on) return;
  hurrying = on;
  battle.messageFadeTime = on ? hurryFadeTime : baseFadeTime;
  battle.scene.updateAcceleration?.(); // Applies to the animation in progress too.
  diag('speed', {hurry: on});
}

// Time from a frame's arrival until its animations finish playing.
let drainSince = null;
function playbackState(state) {
  if (state === 'atqueueend' && drainSince !== null && battle?.seeking === null) {
    diag('drained', {ms: Date.now() - drainSince});
    drainSince = null;
  }
  if (state === 'turn' || state === 'atqueueend') pace();
  updateViewerStatus();
}

// The only path that skips animations; every skip is logged with its reason.
// A target turn stops the skip at that |turn| line and animates from there.
function catchUp(reason, turn = Infinity) {
  if (turn <= battle.turn) return; // Seeking backwards would rebuild the whole battle.
  const from = battle.currentStep, active = !battle.atQueueEnd;
  battle.seekTurn(turn);
  const skipped = battle.currentStep - from;
  if (active && skipped > 0) {
    // Entering a game shows the current position; only later skips are cuts.
    const entry = reason === 'reset';
    if (!entry) { diagCounts.seeks++; diagCounts.skippedSteps += skipped; }
    diag(entry ? 'enter' : 'seek', {reason, skipped, to: turn === Infinity ? 'end' : turn});
  }
}

function updateViewerStatus() {
  if (!relayLast || !battle) return;
  if (latest?.feed_stale && !replay) {
    $id('viewer-status').textContent = 'Player feed interrupted · Last received turn ' + (relayLast.turn || 0);
    return;
  }
  const turn = Math.max(0, battle.turn || 0);
  const lagging = relayLast.live && turn < (relayLast.turn || 0) - 1;
  $id('viewer-status').textContent = relayLast.live
    ? lagging ? 'Catching up · Turn ' + turn + ' → ' + relayLast.turn
      : relayLast.turn ? 'Live · Turn ' + relayLast.turn : 'Live · Team preview'
    : replay ? 'Recorded game replay' : 'Match finished';
}

function connect() {
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
  if (stream) stream.close();
  streamBattle = null;
  stream = new EventSource('/api/events?deltas=1' + (!follow && shown ? '&room=' + encodeURIComponent(shown) : ''));
  const current = stream;
  stream.addEventListener('state', event => {
    if (stream !== current) return;
    const data = JSON.parse(event.data);
    const frame = data.battle;
    if (frame?.log_start !== undefined) {
      if (streamBattle?.room !== frame.room || streamBattle.log.length !== frame.log_start) {
        connect(); // Recover a lost base with a complete snapshot.
        return;
      }
      data.battle = {...frame, log: [...streamBattle.log, ...frame.log]};
    }
    streamBattle = data.battle;
    applyStatus(data.status);
    if ((!shown || follow && !data.status.room) && data.battle?.room) {
      const game = data.status.recent?.find(row => row.room === data.battle.room)?.game || data.status.game || '—';
      show(data.battle.room, game);
    }
    applyBattle(data.battle);
    if (data.error) connectionInterrupted(data.error);
  });
  stream.onerror = () => {
    if (stream !== current) return;
    diagCounts.reconnects++;
    diag('stream-error', {state: current.readyState});
      connectionInterrupted('Connection interrupted. Reconnecting to the live feed…');
    // EventSource retries dropped sockets itself, but an HTTP 503 can close it
    // permanently. Retry that case too (e.g. while launchd restarts the viewer).
    if (current.readyState === 2) {
      retryTimer = setTimeout(() => { if (stream === current) connect(); }, 1500);
    }
  };
}

function connectionInterrupted(message) {
  $id('signal').textContent = 'Reconnecting'; $id('dot').classList.add('wait');
  $id('viewer-status').textContent = message;
  $id('note').classList.add('error');
  setPulseState('player-state', 'Reconnecting', 'warn');
  $id('note').textContent = 'Showing the last received state. Data will refresh when the viewer reconnects.';
}

function show(room, game) {
  if (!roomPattern.test(room) || shown === room) return;

  shown = room;
  $id('open').href = battleURL(room);
  $id('open').classList.remove('hidden');
  $id('watching').textContent = 'Watching game ' + game;
  $id('empty').classList.add('hidden');
  $id('inline-player').classList.remove('hidden');
}

$id('follow').onclick = () => {
  follow = !follow;
  $id('follow').setAttribute('aria-pressed', String(follow));
  $id('follow').textContent = follow ? 'Auto-follow on' : 'Auto-follow paused';
  if (follow) {
    replay = false;
    if (latest?.room) show(latest.room, latest.game);
  }
  connect();
};
$id('sound').onclick = () => {
  const muted = $id('sound').getAttribute('aria-pressed') !== 'true';
  BattleSound.setMute(!muted);
  $id('sound').setAttribute('aria-pressed', String(muted));
  $id('sound').textContent = muted ? 'Sound on' : 'Sound off';
};
$id('viewpoint').onclick = () => { if (battle) battle.switchViewpoint(); };
$id('catchup').onclick = () => {
  if (replay || !follow) {
    follow = true; replay = false;
    $id('follow').setAttribute('aria-pressed', 'true');
    $id('follow').textContent = 'Auto-follow on';
    if (latest?.room) show(latest.room, latest.game);
    connect();
  } else if (battle) {
    catchUp('jump-button');
  }
};
$id('reconnect').onclick = connect;
window.addEventListener('beforeunload', () => {
  if (retryTimer) clearTimeout(retryTimer);
  if (stream) stream.close();
  if (liveTimer) clearInterval(liveTimer);
});
// Animations always finish on their own; only a renderer that stops advancing
// (e.g. a scene error) is skipped forward, after a generous wait.
let stallStep = -1, stallSince = 0;
liveTimer = window.setInterval?.(() => {
  if (replay || !battle || !relayLast?.live || document.hidden || battle.atQueueEnd ||
      battle.currentStep !== stallStep) {
    stallStep = battle ? battle.currentStep : -1; stallSince = Date.now();
  } else if (Date.now() - stallSince > 15000) {
    catchUp('stalled');
  }
}, 1000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    // Background tabs barely run animations; drop the backlog, keep the newest turn.
    if (relayLast && !replay && battle) pace();
  }
});

function performanceSummary(data) {
  const [wins, losses, ties] = data.score;
  const total = wins + losses + ties;
  const rate = total ? wins / total * 100 : null;
  const recent = (data.recent || []).filter(row => ['Win', 'Loss', 'Tie'].includes(row.result));
  const streak = recent.findIndex(row => row.result !== recent[0].result);
  const streakLength = streak < 0 ? recent.length : streak;
  const streakText = !recent.length ? '—' : (streak < 0 && total > recent.length ? '≥' : '') +
    streakLength + ' ' + ({Win:'win', Loss:'loss', Tie:'tie'}[recent[0].result]) +
    (streakLength === 1 ? '' : recent[0].result === 'Loss' ? 'es' : 's');
  return {total, rate, recent, streakText};
}

function decisionTime(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  return ms >= 1000 ? (ms / 1000).toFixed(1) + ' s' : Math.round(ms).toLocaleString() + ' ms';
}

function setPulseState(id, text, tone) {
  const node = $id(id);
  node.textContent = text;
  node.classList.toggle('state-ok', tone === 'ok');
  node.classList.toggle('state-warn', tone === 'warn');
}

function applyModelPulse(data) {
  const model = data.model || {}, system = data.system || {};
  const search = model.mode === 'search', policy = model.mode === 'policy';
  $id('model-name').textContent = search ? 'Simulator search' : policy ? 'Learned policy' : 'Awaiting model data';
  $id('model-description').textContent = search ?
    'Simulated turns. Learned team preview.' : policy ?
    'A learned policy ranks legal actions.' : 'Details appear after the player records a decision.';
  $id('inference-last').textContent = decisionTime(model.last_inference_ms);
  $id('inference-mean').textContent = decisionTime(model.mean_inference_ms);
  $id('revision-label').textContent = search ? 'Preview checkpoint' : 'Checkpoint';
  $id('model-revision').textContent = typeof model.revision === 'string' ? model.revision.slice(0, 8) : 'Not reported';
  $id('model-revision').title = model.revision || 'No recorded checkpoint identity';
  for (const [id, value] of [['search-worlds', model.worlds], ['search-engines', model.engines]]) {
    $id(id + '-row').classList.toggle('hidden', !search);
    $id(id).textContent = Number.isFinite(value) ? value.toLocaleString() : 'Not reported';
  }
  $id('search-worlds').title = 'Opponent team configurations sampled per search decision';
  $id('search-engines').title = 'Configured parallel simulator processes';
  $id('policy-updates-row').classList.toggle('hidden', !policy || model.updates == null);
  $id('updates').textContent = model.updates?.toLocaleString() ?? '—';
  $id('choices-label').textContent = data.room && (!model.room || model.room === data.room) ? 'Decisions this game' : 'Decisions last game';
  $id('choices').textContent = model.decisions?.toLocaleString() ?? data.choices?.toLocaleString() ?? '—';
  const at = Number.isFinite(model.last_decision_at) ? new Date(model.last_decision_at * 1000) : null;
  $id('model-recorded').textContent = at && Number.isFinite(at.getTime()) ?
    'Last recorded at ' + at.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}) : 'No decision timestamp reported.';
  $id('model-recorded').title = at && Number.isFinite(at.getTime()) ? at.toISOString() : '';
  const stopped = ['complete', 'ladder_target_complete', 'deadline_reached', 'stopped'].includes(data.phase);
  const paused = ['paused', 'blocked', 'paused_for_adjustments'].includes(data.phase);
  setPulseState('player-state', data.feed_stale || data.phase === 'interrupted' ? 'Interrupted' :
    stopped ? 'Stopped' : paused ? 'Paused' : data.room ? 'Receiving' : 'Between games',
    data.feed_stale || data.phase === 'interrupted' || paused ? 'warn' : stopped ? null : 'ok');
  const phases = {idle:'Idle', training:'Training', learning:'Learning', evaluating:'Evaluating',
    collecting:'Collecting', scout_training:'Training scout', preparing:'Preparing', job_complete:'Job complete',
    error:'Error', deferred:'Deferred', source_reload_requested:'Reload pending',
    stopped:'Stopped', unavailable:'Not reported'};
  for (const lane of ['learner', 'evaluator']) {
    const phase = system[lane] || 'unavailable';
    setPulseState(lane + '-state', phases[phase] || 'Not reported',
      ['training', 'learning', 'evaluating', 'collecting', 'scout_training'].includes(phase) ? 'ok' :
        ['error', 'deferred', 'source_reload_requested'].includes(phase) ? 'warn' : null);
  }
}

function applyStatus(data) {
    latest = data;
    const stats = performanceSummary(data);
    ['wins', 'losses', 'ties'].forEach((id, i) => { $id(id).textContent = data.score[i]; });
    $id('completed').textContent = stats.total.toLocaleString() + (stats.total === 1 ? ' verified game' : ' verified games');
    $id('win-rate').textContent = stats.rate === null ? '—' : stats.rate.toFixed(1) + '%';
    $id('streak').textContent = stats.streakText;
    applyModelPulse(data);
    if (eloCompleted !== null && eloCompleted !== stats.total) refreshElo();  // a game just finished
    eloCompleted = stats.total;
    $id('signal').textContent = data.feed_stale ? 'Player feed interrupted' : data.room ? 'Live game' : data.phase==='paused_for_adjustments' ? 'Adjustment checkpoint' :
      ['blocked', 'paused'].includes(data.phase) ? 'Run paused' :
      ['complete', 'ladder_target_complete', 'deadline_reached', 'stopped'].includes(data.phase) ? 'Run stopped' : 'Between games';
    $id('dot').classList.toggle('wait', !data.room || data.feed_stale);
    if (follow && data.room) show(data.room, data.game);
    if (!shown && data.recent?.length) show(data.recent[0].room, data.recent[0].game);
    const signature = JSON.stringify(data.recent || []);
    if (signature !== recentSignature) {
    recentSignature = signature;
    const form = $id('form');
    form.replaceChildren();
    form.setAttribute('aria-label', stats.recent.length ? 'Recent results, oldest to newest: ' +
      [...stats.recent].reverse().map(row => row.result).join(', ') : 'No recent results');
    for (const row of [...stats.recent].reverse()) {
      const marker = document.createElement('span');
      marker.textContent = row.result[0]; marker.className = row.result.toLowerCase();
      marker.title = 'Game ' + row.game + ': ' + row.result;
      form.append(marker);
    }
    $id('recent').replaceChildren();
    if (!data.recent?.length) $id('recent').textContent = 'No completed games yet.';
    for (const row of data.recent || []) {
      const a = document.createElement('a'); a.href = battleURL(row.room);
      a.onclick = event => {
        event.preventDefault();
        if (follow) $id('follow').click();
        replay = true; relayRoom = null; show(row.room, row.game); connect();
      };
      const label = document.createElement('span'); label.textContent = 'Game ' + row.game;
      const result = document.createElement('strong'); result.textContent = row.result;
      a.append(label, result); $id('recent').append(a);
    }
    }
    $id('note').classList.remove('error');
    $id('note').textContent = data.feed_stale ? 'Player feed interrupted. Showing the last recorded model activity.' :
      replay ? 'Watching a replay. Model and system data describe the latest player session.' :
      'Model and system data describe the latest player session, including while watching replays.';
}
// Ladder performance panel: /api/elo is computed server-side (cached on ledger
// mtimes) and polled slowly, never per SSE tick.
const eloText = perf => !perf || perf.rating == null ? '—' :
  (perf.clamped === 'wins' ? '≥' : perf.clamped === 'losses' ? '≤' : '') + Math.round(perf.rating) + (perf.ci != null ? ' ±' + Math.round(perf.ci) : '');
const eloGames = perf => perf && perf.n ? perf.n.toLocaleString() + (perf.n === 1 ? ' game' : ' games') : 'no games';
// SVG content is built with textContent so ledger labels never become markup.
function chartElement(svg, tag, attrs = {}, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const key in attrs) el.setAttribute(key, attrs[key]);
  if (text != null) el.textContent = text;
  svg.append(el);
  return el;
}
function chartTitle(el, text) { chartElement(el, 'title', {}, text); }
function setSpark(data) {
  const svg = $id('perf-spark');
  svg.replaceChildren();
  const series = (data.series || []).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  const window = data.series_window || 50;
  $id('perf-caption').textContent = window + '-game window';
  if (series.length < 2) {
    chartElement(svg, 'text', {x:130, y:75, 'text-anchor':'middle'}, 'Waiting for ' + window + ' rated games');
    svg.setAttribute('aria-label', 'Waiting for rolling performance data');
    return;
  }
  const left = 34, right = 253, top = 16, bottom = 125;
  const values = series.map(p => p[1]);
  const lo = Math.floor((Math.min(...values) - 20) / 100) * 100;
  const hi = Math.ceil((Math.max(...values) + 20) / 100) * 100;
  const first = series[0][0], last = series[series.length - 1][0];
  const x = index => left + (index - first) / (last - first) * (right - left);
  const y = value => top + (hi - value) / (hi - lo) * (bottom - top);
  const start = (data.segment_starts || [])[data.segments?.length - 1];
  if (Number.isFinite(start) && start <= last) {
    const cut = x(Math.max(first, start));
    chartElement(svg, 'rect', {class:'band', x:cut, y:top, width:right-cut, height:bottom-top});
    if (start >= first) chartElement(svg, 'line', {class:'cut', x1:cut, x2:cut, y1:top, y2:bottom});
    chartElement(svg, 'text', {x:right, y:10, 'text-anchor':'end'}, 'Current deploy');
  }
  for (const value of [lo, (lo + hi) / 2, hi]) {
    chartElement(svg, 'line', {class:'grid', x1:left, x2:right, y1:y(value), y2:y(value)});
    chartElement(svg, 'text', {x:left-6, y:y(value)+3, 'text-anchor':'end'}, Math.round(value));
  }
  for (const [index, anchor] of [[first, 'start'], [last, 'end']])
    chartElement(svg, 'text', {x:x(index), y:143, 'text-anchor':anchor}, 'Game ' + (index+1).toLocaleString());
  chartElement(svg, 'polyline', {class:'line', points:series.map(p => x(p[0]).toFixed(1)+','+y(p[1]).toFixed(1)).join(' ')});
  for (const point of series) {
    const dot = chartElement(svg, 'circle', {class:'end', cx:x(point[0]), cy:y(point[1]), r:point[0] === last ? 3 : 4, 'fill-opacity':point[0] === last ? 1 : 0, 'pointer-events':'all'});
    // A wide invisible stroke makes each sample inspectable with a mouse.
    dot.setAttribute('stroke', 'transparent'); dot.setAttribute('stroke-width', '8');
    chartTitle(dot, 'Game ' + (point[0]+1) + ': ' + Math.round(point[1]));
  }
  svg.setAttribute('aria-label', 'Rolling ' + window + '-game performance, ' + Math.round(values[0]) + ' to ' + Math.round(values.at(-1)) + '. Shaded area: current deploy.');
}
function setDeployChart(data) {
  const svg = $id('perf-deploys');
  svg.replaceChildren();
  const all = data.segments || [];
  const rows = all.slice(-4).filter(s => Number.isFinite(s.rating));
  const height = Math.max(76, rows.length * 33 + 30);
  svg.setAttribute('viewBox', '0 0 260 ' + height);
  if (!rows.length) {
    chartElement(svg, 'text', {x:130, y:38, 'text-anchor':'middle'}, 'No rated deploys yet');
    svg.setAttribute('aria-label', 'No rated deploys yet');
    return;
  }
  const lo = Math.floor(Math.min(...rows.map(s => s.rating - (s.ci || 0))) / 100) * 100 - 25;
  const hi = Math.ceil(Math.max(...rows.map(s => s.rating + (s.ci || 0))) / 100) * 100 + 25;
  const x = value => 68 + (value-lo)/(hi-lo)*185;
  for (const value of [Math.ceil(lo/100)*100, Math.floor(hi/100)*100]) {
    chartElement(svg, 'line', {class:'grid', x1:x(value), x2:x(value), y1:4, y2:height-22});
    chartElement(svg, 'text', {x:x(value), y:height-5, 'text-anchor':'middle'}, value);
  }
  const descriptions = [];
  rows.forEach((seg, i) => {
    const current = seg === all.at(-1), y = 17 + i * 33;
    const label = current ? 'Live' : '#' + (seg.index + 1);
    chartElement(svg, 'text', {x:0, y:y-1}, label);
    chartElement(svg, 'text', {x:0, y:y+11, style:'font-size:9px'}, seg.n.toLocaleString() + ' games');
    const group = chartElement(svg, 'g');
    const description = label + ': ' + eloText(seg) + ', ' + eloGames(seg) + '. ' + (seg.label || '');
    chartTitle(group, description); descriptions.push(description);
    if (seg.ci != null) {
      chartElement(group, 'line', {class:'interval' + (current ? ' current' : ''), x1:x(seg.rating-seg.ci), x2:x(seg.rating+seg.ci), y1:y, y2:y});
      for (const bound of [seg.rating-seg.ci, seg.rating+seg.ci])
        chartElement(group, 'line', {class:'interval' + (current ? ' current' : ''), x1:x(bound), x2:x(bound), y1:y-4, y2:y+4});
    }
    chartElement(group, 'circle', {class:current ? 'end' : 'previous', cx:x(seg.rating), cy:y, r:3.5});
    if (seg.clamped) chartElement(group, 'text', {x:x(seg.rating), y:y+13, 'text-anchor':'middle'}, seg.clamped === 'wins' ? '≥' : '≤');
  });
  svg.setAttribute('aria-label', 'Deploy performance, 95% confidence intervals. ' + descriptions.join(' '));
}
function applyElo(data) {
  if (!data || data.error) return;
  $id('perf').classList.remove('hidden');
  $id('perf-rating').textContent = data.rating == null ? '—' : data.rating.toLocaleString();
  const change = data.rating_change_50, delta = $id('perf-delta');
  delta.className = change == null ? '' : change > 0 ? 'up' : change < 0 ? 'down' : '';
  delta.textContent = change == null ? 'Current ladder rating' : (change > 0 ? '+' : change < 0 ? '−' : '±') + Math.abs(change) + ' last 50 games';
  const segs = data.segments || [], cur = segs[segs.length - 1], prev = segs[segs.length - 2];
  const label = seg => 'Deploy #' + (seg.index + 1) + ' · ' + eloGames(seg);
  $id('perf-cur-label').textContent = cur ? label(cur) : '';
  $id('perf-cur-label').title = cur ? cur.label + ' (' + cur.first + ' → ' + cur.last + ')' : '';
  $id('perf-cur').textContent = eloText(cur);
  $id('perf-prev-label').textContent = prev ? label(prev) : 'No earlier deploy';
  $id('perf-prev-label').title = prev ? prev.label + ' (' + prev.first + ' → ' + prev.last + ')' : '';
  $id('perf-prev').textContent = prev ? eloText(prev) : '—';
  for (const w of ['50', '100', '200']) {
    const perf = data.windows?.[w];
    $id('perf-w' + w).textContent = perf && perf.n ? (perf.clamped === 'wins' ? '≥' : perf.clamped === 'losses' ? '≤' : '') + Math.round(perf.rating) : '—';
    $id('perf-w' + w + 'n').textContent = perf && perf.n ? (perf.ci != null ? '±' + Math.round(perf.ci) + ' · ' : '') + 'win ' + perf.win_rate.toFixed(0) + '%' : '';
  }
  setSpark(data);
  setDeployChart(data);
  $id('perf-win-rate').textContent = cur?.win_rate == null ? '—' : cur.win_rate.toFixed(1) + '%';
  $id('perf-opponents').textContent = cur?.mean_opp == null ? '—' : Math.round(cur.mean_opp).toLocaleString();
  const gain = cur?.rating != null && prev?.rating != null && !cur.clamped && !prev.clamped ? Math.round(cur.rating-prev.rating) : null;
  $id('perf-gain').textContent = gain == null ? '—' : (gain > 0 ? '+' : '') + gain + ' Elo';
  $id('perf-gain').className = gain > 0 ? 'up' : gain < 0 ? 'down' : '';
  $id('perf-gain').title = 'Difference in performance estimates; see confidence intervals above';
  $id('perf-coverage').textContent = data.rated_games == null ? '—' : data.rated_games.toLocaleString() + ' / ' + data.games.toLocaleString();
  const updated = data.updated ? new Date(data.updated) : null;
  $id('perf-updated').textContent = updated && Number.isFinite(updated.getTime()) ? 'Last result ' + updated.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}) : 'No completed games';
  $id('perf-rating').title = 'Current ladder rating';
  $id('perf-cur').title = 'Opponent-adjusted performance rating ±95% confidence interval';
}
let eloTimer = null, eloCompleted = null;
async function refreshElo() {
  if (typeof fetch !== 'function' || typeof setTimeout !== 'function') return;  // headless test harness
  clearTimeout(eloTimer);
  try {
    const response = await fetch('/api/elo', {cache: 'no-store'});
    if (response.ok) applyElo(await response.json());
  } catch (error) { /* keep the last panel; try again on the next tick */ }
  eloTimer = setTimeout(refreshElo, 30000);
}
refreshElo();
connect();
