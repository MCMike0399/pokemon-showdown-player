"""Renderer protocol compatibility for filtered live relays and recordings."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


VIEWER = Path(__file__).resolve().parents[1] / "web/live-watch/viewer.js"


@pytest.mark.parametrize("score,results,expected", [
    ([0, 0, 0], [], [None, '—']),
    ([202, 321, 0], ['Loss', 'Loss', 'Win', 'Loss', 'Loss', 'Loss', 'Loss', 'Win'],
     [202 / 523 * 100, '2 losses']),
    ([1, 1, 2], ['Tie', 'Tie', 'Loss', 'Win'], [25, '2 ties']),
    ([12, 0, 0], ['Win'] * 8, [100, '≥8 wins']),
    ([8, 0, 0], ['Win'] * 8, [100, '8 wins']),
    ([1, 0, 0], ['Win'], [100, '1 win']),
])
def test_performance_rates_and_streaks_use_verified_results(score, results, expected):
    node = shutil.which('node')
    if not node:
        pytest.skip('viewer checks require Node')
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = new Map();
const element = id => {
  if (!nodes.has(id)) nodes.set(id, {classList:{add(){},remove(){},toggle(){}},
    replaceChildren(){}, setAttribute(){}, append(){}});
  return nodes.get(id);
};
const context = vm.createContext({
  document: {getElementById:element, createElement:() => element(Symbol()), addEventListener(){}},
  window: {addEventListener(){}}, ResizeObserver:class {observe(){}},
  Dex: {getSpriteData(){}}, BattleSound: {setMute(){}},
  EventSource:class {addEventListener(){} close(){}},
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
context.data = {score:input.score, completed:523, target:500, phase:'playing',
  recent:input.results.map((result, i) => ({result, game:523-i, room:'invalid'}))};
const stats = vm.runInContext('performanceSummary(data)', context);
vm.runInContext('applyStatus(data)', context);
process.stdout.write(JSON.stringify({stats, signal:element('signal').textContent,
  completed:element('completed').textContent, note:element('note').textContent}));
"""
    response = subprocess.run([node, '-e', script], text=True, capture_output=True, check=True,
                              input=json.dumps({'viewer': str(VIEWER), 'score': score, 'results': results}))
    data = json.loads(response.stdout)
    assert [data['stats'][key] for key in ('rate', 'streakText')] == expected
    assert data['signal'] == 'Between games'  # Passing the old cap never ends an open run.
    assert ' of ' not in data['completed']
    assert 'NaN' not in data['note']


def test_model_pulse_clears_search_data_and_handles_missing_measurements():
    node = shutil.which('node')
    if not node:
        pytest.skip('viewer checks require Node')
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const nodes = new Map();
const element = id => {
  if (!nodes.has(id)) {
    const classes = new Set();
    nodes.set(id, {classList:{toggle(name, enabled){enabled ? classes.add(name) : classes.delete(name);},
      contains(name){return classes.has(name);}}});
  }
  return nodes.get(id);
};
const context = vm.createContext({document:{getElementById:element, addEventListener(){}},
  window:{addEventListener(){}}, ResizeObserver:class {observe(){}}, Dex:{getSpriteData(){}},
  BattleSound:{setMute(){}}, EventSource:class {addEventListener(){} close(){}}});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
context.data = {room:'battle-test-1', model:{mode:'search', room:'battle-test-1', worlds:6, engines:3,
  revision:'abcdef123456', last_inference_ms:2313.3, mean_inference_ms:0, decisions:8}};
vm.runInContext('applyModelPulse(data)', context);
const search = {last:element('inference-last').textContent, mean:element('inference-mean').textContent,
  revision:element('model-revision').textContent, fullRevision:element('model-revision').title,
  worldsVisible:!element('search-worlds-row').classList.contains('hidden')};
context.data = {model:{mode:'policy', updates:0}, system:{learner:'training', evaluator:'stopped'}};
vm.runInContext('applyModelPulse(data)', context);
const policy = {worldsHidden:element('search-worlds-row').classList.contains('hidden'),
  updatesVisible:!element('policy-updates-row').classList.contains('hidden'),
  revision:element('model-revision').textContent, last:element('inference-last').textContent,
  updates:element('updates').textContent, learner:element('learner-state').textContent};
vm.runInContext('applyModelPulse({})', context);
process.stdout.write(JSON.stringify({search, policy, missing:element('learner-state').textContent}));
"""
    result = subprocess.run([node, '-e', script, str(VIEWER)], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data['search'] == {'last': '2.3 s', 'mean': '0 ms', 'revision': 'abcdef12',
                             'fullRevision': 'abcdef123456', 'worldsVisible': True}
    assert data['policy'] == {'worldsHidden': True, 'updatesVisible': True, 'revision': 'Not reported',
                             'last': '—', 'updates': '0', 'learner': 'Training'}
    assert data['missing'] == 'Not reported'


PREVIEW = [
    "|gametype|doubles", "|player|p1|One", "|player|p2|Two",
    "|poke|p1|Pikachu, L50", "|poke|p1|Eevee, L50",
    "|poke|p2|Pikachu, L50", "|poke|p2|Eevee, L50", "|teampreview|4",
]
SWITCHES = [
    "|switch|p1a: Pikachu|Pikachu, L50|100/100",
    "|switch|p2a: Pikachu|Pikachu, L50|100/100", "|turn|1",
]


def render_log(lines):
    node = shutil.which("node")
    if not node:
        pytest.skip("viewer compatibility checks require Node")
    # Load the actual viewer while keeping its automatic fetch loops dormant.
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({
  document: {getElementById: () => ({}), addEventListener() {}},
  window: {addEventListener() {}},
  ResizeObserver: class {observe() {}},
  Dex: {getSpriteData() {}}, BattleSound: {setMute() {}},
  EventSource: class {addEventListener() {} close() {}},
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
context.lines = input.lines;
process.stdout.write(JSON.stringify(vm.runInContext('rendererLog(lines)', context)));
"""
    result = subprocess.run(
        [node, "-e", script], input=json.dumps({"viewer": str(VIEWER), "lines": lines}),
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def test_filtered_relay_clears_team_preview_before_first_switch():
    lines = PREVIEW + SWITCHES
    assert render_log(lines) == PREVIEW + ["|start"] + SWITCHES
    assert lines == PREVIEW + SWITCHES


def test_team_preview_remains_until_battle_actually_starts():
    assert render_log(PREVIEW) == PREVIEW


@pytest.mark.parametrize("start", ["|start", "|start|"])
def test_full_protocol_recordings_keep_their_start_event(start):
    lines = PREVIEW + [start] + SWITCHES
    assert render_log(lines) == lines


def test_cumulative_polls_and_later_switches_keep_one_start_boundary():
    initial = render_log(PREVIEW)
    started = render_log(PREVIEW + SWITCHES)
    later = PREVIEW + SWITCHES + ["|switch|p1a: Eevee|Eevee, L50|100/100", "|turn|2"]
    final = render_log(later)
    assert started[:len(initial)] == initial
    assert final[:len(started)] == started
    assert final.count("|start") == 1
    assert render_log(final) == final


def test_polls_append_without_restarting_active_playback():
    node = shutil.which("node")
    if not node:
        pytest.skip("viewer compatibility checks require Node")
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const events = [];
const context = vm.createContext({
  document: {getElementById: () => ({clientWidth:640, style:{}, classList:{add() {}, remove() {}}}), addEventListener() {}},
  window: {addEventListener() {}},
  ResizeObserver: class {observe() {}},
  Dex: {getSpriteData() {}}, BattleSound: {setMute() {}},
  jQuery: () => ({empty() {}}),
  setTimeout() {},
  EventSource: class {addEventListener() {} close() {}},
  Battle: class {
    constructor(options) {this.stepQueue=[]; this.currentStep=0; this.scene={updateAcceleration() {}};
      events.push(['create', options.paused]);}
    setViewpoint(side) {events.push(['viewpoint', side]);}
    addBatch(lines) {events.push(['append', lines]);}
    seekTurn() {events.push(['seek']);}
    play() {events.push(['play']);}
    pause() {events.push(['pause']);}
    destroy() {events.push(['destroy']);}
  },
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
let data = {room:'battle-gen9doublesou-1', side:'p2', live:true, log:input.lines};
context.data = data;
vm.runInContext('shown = data.room', context);
(async () => {
  vm.runInContext('applyBattle(data)', context);
  vm.runInContext('applyBattle(data)', context);
  context.data = {...data, log:[...data.log, '|turn|2']};
  vm.runInContext('applyBattle(data)', context);
  context.data = {...data, room:'battle-gen9doublesou-2'};
  vm.runInContext('shown = data.room', context);
  vm.runInContext('applyBattle(data)', context);
  process.stdout.write(JSON.stringify(events));
})();
"""
    result = subprocess.run(
        [node, "-e", script], input=json.dumps({"viewer": str(VIEWER), "lines": PREVIEW + SWITCHES}),
        text=True, capture_output=True, check=True,
    )
    events = json.loads(result.stdout)
    assert events[:2] == [["create", True], ["viewpoint", "p2"]]
    assert events.count(["play"]) == 2  # Once for each room, never on repeated polls.
    appended = [event[1] for event in events if event[0] == "append"]
    assert appended[:2] == [PREVIEW + ["|start"] + SWITCHES, ["|turn|2"]]
    assert [] not in appended  # Score-only events must not restart the animation loop.
    assert events[events.index(["destroy"]) - 1] == ["pause"]


PACING_HARNESS = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const events = [];
const context = vm.createContext({
  document: {getElementById: () => ({clientWidth:640, style:{}, classList:{add() {}, remove() {}}}), addEventListener() {}},
  window: {addEventListener() {}}, ResizeObserver: class {observe() {}},
  Dex: {getSpriteData() {}}, BattleSound: {setMute() {}}, jQuery: () => ({empty() {}}),
  EventSource: class {addEventListener() {} close() {}},
  Battle: class {
    constructor() {this.turn=1; this.stepQueue=[]; this.currentStep=0; this.atQueueEnd=false;
      this.scene={updateAcceleration() {}}; events.push('create');}
    setViewpoint() {}
    addBatch(lines) {
      this.stepQueue.push(...lines);
      for (const line of lines) if (line.startsWith('|turn|')) this.lastTurn = Number(line.slice(6));
    }
    seekTurn(turn) {
      events.push(turn === Infinity ? 'seek' : 'seek:' + turn);
      this.turn = turn === Infinity ? this.lastTurn : turn;
    }
    play() {events.push('play');} pause() {} destroy() {}
  },
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
const fade = () => vm.runInContext('battle.messageFadeTime', context);
"""


def run_pacing(body):
    node = shutil.which('node')
    if not node:
        pytest.skip('viewer checks require Node')
    result = subprocess.run([node, '-e', PACING_HARNESS + body], input=json.dumps({'viewer': str(VIEWER)}),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_reconnect_catches_up_and_replays_ignore_other_rooms():
    data = run_pacing(r"""
context.data = {room:'battle-test-1', side:'p2', live:true, turn:1, log:['|turn|1']};
vm.runInContext('shown=data.room; applyBattle(data)', context);
vm.runInContext('applyBattle(data)', context); // Unchanged state does not start another animation loop.
vm.runInContext('connect(); applyBattle(data)', context); // Reconnecting to the shown game resumes in place.
context.data = {...context.data, turn:5, log:['|turn|1','|turn|5']};
vm.runInContext('applyBattle(data)', context); // A large backlog jumps to the newest turn's start.
context.data = {...context.data, log:['|turn|2','|turn|5']};
vm.runInContext('applyBattle(data)', context); // Same-length changed prefix rebuilds instead of duplicating.
const liveEvents = [...events];
vm.runInContext('replay=true; relayRoom=null; applyBattle(data)', context);
const replayEvents = events.slice(liveEvents.length);
context.data = {...context.data, room:'battle-test-2'};
vm.runInContext('applyBattle(data)', context);
process.stdout.write(JSON.stringify({liveEvents, replayEvents, events}));
""")
    assert data['liveEvents'] == ['create', 'seek', 'play', 'seek:4', 'create', 'seek', 'play']
    assert data['replayEvents'] == ['create', 'play']
    assert data['events'] == data['liveEvents'] + data['replayEvents']


def test_live_turns_play_out_and_backlog_speeds_up_before_skipping():
    data = run_pacing(r"""
const steps = [];
const frame = (turn, live=true) => {
  context.data = {room:'battle-test-1', side:'p1', live, turn,
    log:[...Array(turn).keys()].flatMap(n => ['|move|p1a: A|Tackle|p2a: B', '|turn|' + (n + 1)])};
  vm.runInContext('applyBattle(data)', context);
};
vm.runInContext("shown='battle-test-1'", context);
frame(1);
const entry = events.length;
frame(2); steps.push({events: events.slice(entry), fade: fade()});   // the next turn animates in full
frame(3); steps.push({events: events.slice(entry), fade: fade()});   // one more queued: hyperfast, no skip
vm.runInContext('battle.turn = 3', context);
vm.runInContext("playbackState('turn')", context);
steps.push({events: events.slice(entry), fade: fade()});             // caught up: normal speed again
frame(4, false); steps.push({events: events.slice(entry), fade: fade()}); // the final turn is not cut
frame(8); steps.push({events: events.slice(entry), fade: fade()});   // a long backlog jumps to turn 7
process.stdout.write(JSON.stringify(steps));
""")
    assert [step['events'] for step in data[:4]] == [[], [], [], []]
    assert [step['fade'] for step in data] == [100, 40, 100, 100, 100]
    assert data[4]['events'] == ['seek:7']  # ...where the newest turn animates at normal speed.


def test_animations_are_never_skipped_on_a_timer():
    source = VIEWER.read_text()
    timer = source[source.index('liveTimer = window.setInterval'):]
    timer = timer[:timer.index('}, 1000);')]
    assert 'stalled' in timer and '15000' in timer  # Only a renderer that stops advancing.
    assert source.count('battle.seekTurn(') == 1    # Every skip goes through logged catchUp().


def test_http_503_retries_one_stream_and_cancels_old_retry():
    node = shutil.which('node')
    if not node:
        pytest.skip('viewer checks require Node')
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const sources = [], timers = new Map();
let nextTimer = 1;
const context = vm.createContext({
  document: {getElementById: () => ({classList:{add(){},toggle(){}}}), addEventListener() {}},
  window: {addEventListener() {}}, ResizeObserver: class {observe() {}},
  Dex: {getSpriteData() {}}, BattleSound: {setMute() {}},
  EventSource: class {
    constructor() {this.readyState=0; sources.push(this);} addEventListener() {}
    close() {this.readyState=2; this.closed=true;}
  },
  setTimeout(fn) {const id=nextTimer++; timers.set(id,fn); return id;},
  clearTimeout(id) {timers.delete(id);},
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
sources[0].readyState=2; sources[0].onerror();
const scheduled = timers.size;
const retry = [...timers.values()][0]; timers.clear(); retry();
const afterRetry = sources.length;
sources[1].readyState=2; sources[1].onerror();
vm.runInContext('connect()', context); // Clicking reconnect cancels the pending retry.
sources[1].onerror(); // Events from the old stream must not schedule another retry.
process.stdout.write(JSON.stringify({scheduled, afterRetry, count:sources.length,
  pending:timers.size, closed:sources.slice(0,-1).every(s=>s.closed)}));
"""
    result = subprocess.run([node, '-e', script], input=json.dumps({'viewer': str(VIEWER)}),
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {'scheduled': 1, 'afterRetry': 2, 'count': 3, 'pending': 0, 'closed': True}


def test_page_loads_animated_sprite_metadata_before_starting_renderer():
    # The real Dex silently chooses still PNGs when the sprite metadata is absent.
    # Guard the HTML bootstrap dependency that caused static Pokemon in the UI.
    from html.parser import HTMLParser

    class ScriptSources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sources = []

        def handle_starttag(self, tag, attrs):
            if tag == 'script':
                source = dict(attrs).get('src')
                if source:
                    self.sources.append(source)

    parser = ScriptSources()
    parser.feed(VIEWER.with_name('index.html').read_text())
    metadata = 'https://play.pokemonshowdown.com/data/pokedex-mini.js'
    assert metadata in parser.sources, 'Missing animated sprite metadata: Dex falls back to still PNGs'
    assert parser.sources.index(metadata) < parser.sources.index('/viewer.js')


def test_related_artwork_keeps_animated_sprite_dimensions_and_actual_cry():
    node = shutil.which('node')
    if not node:
        pytest.skip('viewer checks require Node')
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({
  document: {getElementById: () => ({classList:{add(){},remove(){}}}), addEventListener() {}},
  window: {addEventListener() {}}, ResizeObserver: class {observe() {}},
  BattleSound: {setMute() {}}, EventSource: class {addEventListener() {} close() {}},
  Dex: {
    species: {get: id => ({name:id})},
    getSpriteData(pokemon, isFront, options) {
      const file = pokemon === 'Garchomp-Mega-Z' ? 'garchomp-megaz.png' : 'garchomp-mega.gif';
      return {url:'https://play.pokemonshowdown.com/sprites/'+file,
        w:file.endsWith('.gif')?131:96, h:file.endsWith('.gif')?92:96,
        cryurl:pokemon+'.mp3', facing:isFront, gen:options.gen};
    },
  },
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
process.stdout.write(JSON.stringify(context.Dex.getSpriteData('Garchomp-Mega-Z',false,{gen:9})));
"""
    result = subprocess.run([node, '-e', script], input=json.dumps({'viewer': str(VIEWER)}),
                            text=True, capture_output=True, check=True)
    sprite = json.loads(result.stdout)
    assert sprite['url'].endswith('/garchomp-mega.gif')
    assert (sprite['w'], sprite['h']) == (131, 92)
    assert sprite['cryurl'] == 'Garchomp-Mega-Z.mp3'
    assert sprite['facing'] is False
    assert sprite['gen'] == 9
