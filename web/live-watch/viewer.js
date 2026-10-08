/* Read-only public battle renderer. No login, account creation, chat or choices. */
const $id = id => document.getElementById(id);
let follow = true, shown = null, latest = null, battle = null, stream = null;
let catchUpNext = true, replay = false, relayLog = [], recentSignature = null;
let retryTimer = null;
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
  battle = new Battle({id: room, $frame: jQuery('#battle'), $logFrame: jQuery('#battle-log'), paused: true});
  battle.roomid = room;
  battle.joinButtons = false;
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
  battle.addBatch(log.slice(relayCount));
  relayCount = log.length; relayLog = log; relayLast = data;
  const liveView = !replay && (data.live || data.source === 'player-relay');
  const lag = Number(data.turn || 0) - battle.turn;
  const pending = (battle.stepQueue?.length || 0) - (battle.currentStep || 0);
  if (liveView && (reset || catchUpNext || document.hidden || lag > 1 || pending > 80)) {
    battle.seekTurn(Infinity);
  }
  // addBatch resumes playback itself. Starting a second loop causes scene errors.
  if (reset) battle.play();
  catchUpNext = false;
  $id('viewer-status').textContent = data.live
    ? data.turn ? 'Live · Turn ' + data.turn : 'Live · Team preview'
    : replay ? 'Recorded game replay' : 'Match finished';
}

function connect() {
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
  if (stream) stream.close();
  catchUpNext = true;
  stream = new EventSource('/api/events' + (!follow && shown ? '?room=' + encodeURIComponent(shown) : ''));
  const current = stream;
  stream.addEventListener('state', event => {
    if (stream !== current) return;
    const data = JSON.parse(event.data);
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
    connectionInterrupted('Connection interrupted. Reconnecting to DiveMac…');
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
  $id('note').textContent = 'The match keeps running on DiveMac. This viewer will catch up when connected.';
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
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    catchUpNext = true;
    if (relayLast && !replay && battle) battle.seekTurn(Infinity);
  }
});

function applyStatus(data) {
    latest = data;
    ['wins', 'losses', 'ties'].forEach((id, i) => { $id(id).textContent = data.score[i]; });
    $id('progress').textContent = (data.completed || 0) + ' of ' + (data.target || 100) + ' games finished';
    $id('bar').value = data.completed || 0; $id('bar').max = data.target || 100;
    $id('updates').textContent = data.updates ?? '—'; $id('choices').textContent = data.choices ?? '—';
    $id('samples').textContent = data.scout_samples?.toLocaleString() ?? '—';
    $id('signal').textContent = data.room ? 'Live game' : data.phase==='paused_for_adjustments' ? 'Adjustment checkpoint' : data.phase==='blocked' ? 'Campaign paused' : data.completed >= data.target ? 'Campaign finished' : 'Between games';
    $id('dot').classList.toggle('wait', !data.room);
    if (follow && data.room) show(data.room, data.game);
    if (!shown && data.recent?.length) show(data.recent[0].room, data.recent[0].game);
    const signature = JSON.stringify(data.recent || []);
    if (signature !== recentSignature) {
    recentSignature = signature;
    $id('recent').replaceChildren();
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
    $id('note').textContent = 'DiveMac keeps the match state while you’re away. Scores use verified terminal recordings.';
}
connect();
