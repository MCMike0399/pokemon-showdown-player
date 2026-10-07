/* Read-only public battle renderer. No login, account creation, chat or choices. */
const $id = id => document.getElementById(id);
let follow = true, shown = null, latest = null, battle = null, socket = null;
let socketReady = false, reconnectDelay = 1500, reconnectTimer = null, manualView = false;
const roomPattern = /^battle-[a-z0-9]+-[0-9]+(?:-[a-z0-9]+)?$/;
const battleURL = room => 'https://play.pokemonshowdown.com/' + room;
Dex.resourcePrefix = 'https://play.pokemonshowdown.com/';
Dex.fxPrefix = Dex.resourcePrefix + 'fx/';
BattleSound.setMute(true);

function resizeBattle() {
  const width = $id('arena-wrap').clientWidth;
  const scale = Math.min(width / 640, 1.8);
  $id('battle').style.transform = 'scale(' + scale + ')';
  $id('arena-wrap').style.height = Math.ceil(360 * scale) + 'px';
}
new ResizeObserver(resizeBattle).observe($id('arena-wrap'));

function newBattle(room) {
  if (battle) battle.destroy();
  jQuery('#battle,#battle-log').empty();
  battle = new Battle({id: room, $frame: jQuery('#battle'), $logFrame: jQuery('#battle-log'), paused: false});
  battle.roomid = room;
  battle.joinButtons = false;
  manualView = false;
  resizeBattle();
}

let relayCount=0, relayRoom=null, relayLast=null;
function joinCurrent(){ relayCount=0;relayRoom=null; }
async function connect(){
 try{
  const response=await fetch('/api/battle'+(shown?'?room='+encodeURIComponent(shown):''),{cache:'no-store'});
  if(!response.ok)throw Error('relay unavailable');
  const data=await response.json();
  if(data.room&&data.log?.length){
   const reset=relayRoom!==data.room||data.log.length<relayCount;
   if(reset){newBattle(data.room);relayCount=0;relayRoom=data.room;}
   battle.addBatch(data.log.slice(relayCount));relayCount=data.log.length;
   if(reset){if(data.live)battle.seekTurn(Infinity);if(data.side)battle.setViewpoint(data.side);}
   battle.play();relayLast=data;
   $id('viewer-status').textContent=data.live?'Live player relay':'Recorded game replay';
  }else{$id('viewer-status').textContent='Waiting for the player relay. Recorded games remain available below.';}
 }catch(error){$id('viewer-status').textContent='Player relay reconnecting…';console.error('Player relay:',error);}
 reconnectTimer=setTimeout(connect,document.hidden?3000:700);
}

function show(room, game) {
  if (!roomPattern.test(room) || shown === room) return;

  shown = room;
  $id('open').href = battleURL(room);
  $id('open').classList.remove('hidden');
  $id('watching').textContent = 'Watching game ' + game;
  $id('empty').classList.add('hidden');
  $id('inline-player').classList.remove('hidden');
  newBattle(room);
  joinCurrent();
}

$id('follow').onclick = () => {
  follow = !follow;
  $id('follow').setAttribute('aria-pressed', String(follow));
  $id('follow').textContent = follow ? 'Auto-follow on' : 'Auto-follow paused';
  if (follow && latest?.room) show(latest.room, latest.game);
};
$id('sound').onclick = () => {
  const muted = $id('sound').getAttribute('aria-pressed') !== 'true';
  BattleSound.setMute(!muted);
  $id('sound').setAttribute('aria-pressed', String(muted));
  $id('sound').textContent = muted ? 'Sound on' : 'Sound off';
};
$id('viewpoint').onclick = () => { if (battle) { manualView = true; battle.switchViewpoint(); } };
$id('catchup').onclick = () => { if (battle) { battle.seekTurn(Infinity); battle.play(); } };
$id('reconnect').onclick = () => {
  const previous = socket;
  socket = null;
  if (previous) previous.close();
  connect();
};
window.addEventListener('beforeunload', () => { if (socket) socket.close(); });

async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    if (!response.ok) throw Error('status unavailable');
    const data = await response.json(); latest = data;
    ['wins', 'losses', 'ties'].forEach((id, i) => { $id(id).textContent = data.score[i]; });
    $id('progress').textContent = (data.completed || 0) + ' of ' + (data.target || 100) + ' games finished';
    $id('bar').value = data.completed || 0; $id('bar').max = data.target || 100;
    $id('updates').textContent = data.updates ?? '—'; $id('choices').textContent = data.choices ?? '—';
    $id('samples').textContent = data.scout_samples?.toLocaleString() ?? '—';
    $id('signal').textContent = data.room ? 'Live game' : data.phase==='paused_for_adjustments' ? 'Adjustment checkpoint' : data.phase==='blocked' ? 'Campaign paused' : data.completed >= data.target ? 'Campaign finished' : 'Between games';
    $id('dot').classList.toggle('wait', !data.room);
    if (follow && data.room) show(data.room, data.game);
    if (!shown && data.recent?.length) show(data.recent[0].room, data.recent[0].game);
    $id('recent').replaceChildren();
    for (const row of data.recent || []) {
      const a = document.createElement('a'); a.href = battleURL(row.room);
      a.onclick = event => { event.preventDefault(); if (follow) $id('follow').click(); show(row.room, row.game); };
      const label = document.createElement('span'); label.textContent = 'Game ' + row.game;
      const result = document.createElement('strong'); result.textContent = row.result;
      a.append(label, result); $id('recent').append(a);
    }
    $id('note').classList.remove('error');
    $id('note').textContent = 'Scores use verified terminal recordings. Private and public games use the existing player’s read-only relay.';
  } catch (error) {
    console.error('Campaign viewer:',error);
    $id('signal').textContent = 'Reconnecting'; $id('dot').classList.add('wait');
    $id('note').classList.add('error');
    $id('note').textContent = 'Campaign connection interrupted. Retrying; the live battle stays open.';
  }
  setTimeout(refresh, document.hidden ? 8000 : 2000);
}
connect(); refresh();
