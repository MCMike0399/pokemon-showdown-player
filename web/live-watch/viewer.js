/* Read-only public battle renderer. No login, account creation, chat or choices. */
const $id = id => document.getElementById(id);
let follow = true, shown = null, latest = null, battle = null, stream = null;
let catchUpNext = true, replay = false, relayLog = [], recentSignature = null;
let retryTimer = null, liveTimer = null;
let relayReceivedAt = 0, streamBattle = null;
let alternateArtwork = new Set();
const roomPattern = /^battle-[a-z0-9]+-[0-9]+(?:-[a-z0-9]+)?$/;
const battleURL = room => 'https://play.pokemonshowdown.com/' + room;
Dex.resourcePrefix = 'https://play.pokemonshowdown.com/';
Dex.fxPrefix = Dex.resourcePrefix + 'fx/';
BattleSound.setMute(true);

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
    subscription: updateViewerStatus});
  battle.roomid = room;
  battle.joinButtons = false;
  // Showdown's own fast animation mode retains move effects at twice the pace.
  battle.messageFadeTime = 100;
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
    newBattle(data.room, data.side);
    relayCount = 0; relayRoom = data.room;
  }
  if (log.length > relayCount) {
    relayReceivedAt = Date.now();
    battle.addBatch(log.slice(relayCount));
  }
  relayCount = log.length; relayLog = log; relayLast = data;
  const liveView = !replay;
  const lag = Number(data.turn || 0) - battle.turn;
  const pending = (battle.stepQueue?.length || 0) - (battle.currentStep || 0);
  if (liveView && (reset || catchUpNext || document.hidden || lag > 1 || pending > 80 || !data.live && pending > 0)) {
    battle.seekTurn(Infinity);
  }
  // addBatch resumes playback itself. Starting a second loop causes scene errors.
  if (reset) battle.play();
  catchUpNext = false;
  updateViewerStatus();
}

function updateViewerStatus() {
  if (!relayLast || !battle) return;
  if (latest?.feed_stale) {
    $id('viewer-status').textContent = 'Player feed interrupted · Last received turn ' + (relayLast.turn || 0);
    return;
  }
  const turn = Math.max(0, battle.turn || 0);
  const lagging = relayLast.live && turn < (relayLast.turn || 0);
  $id('viewer-status').textContent = relayLast.live
    ? lagging ? 'Catching up · Turn ' + turn + ' → ' + relayLast.turn
      : relayLast.turn ? 'Live · Turn ' + relayLast.turn : 'Live · Team preview'
    : replay ? 'Recorded game replay' : 'Match finished';
}

function connect() {
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
  if (stream) stream.close();
  catchUpNext = true;
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
    catchUpNext = true;
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
  $id('note').textContent = 'The match keeps running. Live insights will refresh when the feed reconnects.';
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
    battle.seekTurn(Infinity);
  }
};
$id('reconnect').onclick = connect;
window.addEventListener('beforeunload', () => {
  if (retryTimer) clearTimeout(retryTimer);
  if (stream) stream.close();
  if (liveTimer) clearInterval(liveTimer);
});
// A turn can contain enough animations to fall behind without another relay
// arriving. Bound that delay even when the latest server turn has not changed.
liveTimer = window.setInterval?.(() => {
  if (!replay && battle && relayLast?.live && !document.hidden &&
      battle.currentStep < battle.stepQueue.length && Date.now() - relayReceivedAt > 1800) {
    battle.seekTurn(Infinity);
  }
}, 500);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    catchUpNext = true;
    if (relayLast && !replay && battle) battle.seekTurn(Infinity);
  }
});

function performanceSummary(data) {
  const [wins, losses, ties] = data.score;
  const total = wins + losses + ties;
  const rate = total ? wins / total * 100 : null;
  const recent = (data.recent || []).filter(row => ['Win', 'Loss', 'Tie'].includes(row.result));
  const recentWins = recent.filter(row => row.result === 'Win').length;
  const recentRate = recent.length ? recentWins / recent.length * 100 : null;
  const streak = recent.findIndex(row => row.result !== recent[0].result);
  const streakLength = streak < 0 ? recent.length : streak;
  const streakText = !recent.length ? '—' : (streak < 0 && total > recent.length ? '≥' : '') +
    streakLength + ' ' + ({Win:'win', Loss:'loss', Tie:'tie'}[recent[0].result]) +
    (streakLength === 1 ? '' : recent[0].result === 'Loss' ? 'es' : 's');
  let insight = 'Insights appear as verified games finish.';
  if (recent.length) {
    const gap = recentRate - rate;
    insight = recentWins + (recentWins === 1 ? ' win' : ' wins') + ' in the last ' + recent.length +
      (recent.length === 1 ? ' game. ' : ' games. ');
    insight += total === recent.length ? 'Building the first performance baseline.' :
      Math.abs(gap) < 0.05 ? 'Recent win rate matches the full-run average.' :
      'Recent win rate is ' + Math.abs(gap).toFixed(1) + ' percentage points ' +
        (gap > 0 ? 'above' : 'below') + ' the full-run average.';
  }
  return {total, rate, recent, recentRate, streakText, insight};
}

function applyStatus(data) {
    latest = data;
    const stats = performanceSummary(data);
    ['wins', 'losses', 'ties'].forEach((id, i) => { $id(id).textContent = data.score[i]; });
    $id('completed').textContent = stats.total.toLocaleString() + (stats.total === 1 ? ' verified game' : ' verified games');
    $id('win-rate').textContent = stats.rate === null ? '—' : stats.rate.toFixed(1) + '%';
    $id('recent-label').textContent = stats.recent.length ? 'Win rate · last ' + stats.recent.length : 'Recent win rate';
    $id('recent-rate').textContent = stats.recentRate === null ? '—' : stats.recentRate.toFixed(1) + '%';
    $id('streak').textContent = stats.streakText;
    $id('updates').textContent = data.updates ?? '—'; $id('choices').textContent = data.choices ?? '—';
    $id('samples').textContent = data.scout_samples?.toLocaleString() ?? '—';
    $id('signal').textContent = data.feed_stale ? 'Player feed interrupted' : data.room ? 'Live game' : data.phase==='paused_for_adjustments' ? 'Adjustment checkpoint' :
      ['blocked', 'paused'].includes(data.phase) ? 'Run paused' :
      ['complete', 'ladder_target_complete', 'deadline_reached'].includes(data.phase) ? 'Run stopped' : 'Between games';
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
    $id('note').textContent = stats.insight;
}
connect();
