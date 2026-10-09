"""Renderer protocol compatibility for filtered live relays and recordings."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


VIEWER = Path(__file__).resolve().parents[1] / "web/live-watch/viewer.js"


@pytest.mark.parametrize("score,results,expected", [
    ([0, 0, 0], [], [None, None, '—']),
    ([202, 321, 0], ['Loss', 'Loss', 'Win', 'Loss', 'Loss', 'Loss', 'Loss', 'Win'],
     [202 / 523 * 100, 25, '2 losses']),
    ([1, 1, 2], ['Tie', 'Tie', 'Loss', 'Win'], [25, 25, '2 ties']),
    ([12, 0, 0], ['Win'] * 8, [100, 100, '≥8 wins']),
    ([8, 0, 0], ['Win'] * 8, [100, 100, '8 wins']),
    ([1, 0, 0], ['Win'], [100, 100, '1 win']),
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
    assert [data['stats'][key] for key in ('rate', 'recentRate', 'streakText')] == expected
    assert data['signal'] == 'Between games'  # Passing the old cap never ends an open run.
    assert ' of ' not in data['completed']
    assert 'NaN' not in data['note']


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
    constructor(options) {events.push(['create', options.paused]);}
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


def test_reconnect_catches_up_and_replays_ignore_other_rooms():
    node = shutil.which('node')
    if not node:
        pytest.skip('viewer checks require Node')
    script = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const events = [];
const context = vm.createContext({
  document: {getElementById: () => ({clientWidth:640, style:{}, classList:{add() {}, remove() {}}}), addEventListener() {}},
  window: {addEventListener() {}}, ResizeObserver: class {observe() {}},
  Dex: {getSpriteData() {}}, BattleSound: {setMute() {}}, jQuery: () => ({empty() {}}),
  EventSource: class {addEventListener() {} close() {}},
  Battle: class {
    constructor() {this.turn=1; this.stepQueue=[]; this.currentStep=0; events.push('create');}
    setViewpoint() {} addBatch() {} seekTurn() {events.push('seek');}
    play() {events.push('play');} pause() {} destroy() {}
  },
});
vm.runInContext(fs.readFileSync(input.viewer, 'utf8'), context);
context.data = {room:'battle-test-1', side:'p2', live:true, turn:1, log:['|turn|1']};
vm.runInContext('shown=data.room; applyBattle(data)', context);
vm.runInContext('applyBattle(data)', context); // Unchanged state does not start another animation loop.
vm.runInContext('connect(); applyBattle(data)', context); // Reconnecting catches up even with the same log.
context.data = {...context.data, turn:5, log:['|turn|1','|turn|5']};
vm.runInContext('applyBattle(data)', context); // A large turn backlog must be skipped.
context.data = {...context.data, log:['|turn|2','|turn|5']};
vm.runInContext('applyBattle(data)', context); // Same-length changed prefix rebuilds instead of duplicating.
const liveEvents = [...events];
vm.runInContext('replay=true; relayRoom=null; applyBattle(data)', context);
const replayEvents = events.slice(liveEvents.length);
context.data = {...context.data, room:'battle-test-2'};
vm.runInContext('applyBattle(data)', context);
process.stdout.write(JSON.stringify({liveEvents, replayEvents, events}));
"""
    result = subprocess.run([node, '-e', script], input=json.dumps({'viewer': str(VIEWER)}),
                            text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data['liveEvents'] == ['create', 'seek', 'play', 'seek', 'seek', 'create', 'seek', 'play']
    assert data['replayEvents'] == ['create', 'play']
    assert data['events'] == data['liveEvents'] + data['replayEvents']


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
  document: {getElementById: () => ({classList:{add(){}}}), addEventListener() {}},
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
