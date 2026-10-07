"""Renderer protocol compatibility for filtered live relays and recordings."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


VIEWER = Path(__file__).resolve().parents[1] / "web/live-watch/viewer.js"
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
  document: {getElementById: () => ({})},
  window: {addEventListener() {}},
  ResizeObserver: class {observe() {}},
  Dex: {}, BattleSound: {setMute() {}},
  fetch: () => new Promise(() => {}),
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
  document: {getElementById: () => ({clientWidth:640, style:{}})},
  window: {addEventListener() {}},
  ResizeObserver: class {observe() {}},
  Dex: {}, BattleSound: {setMute() {}},
  jQuery: () => ({empty() {}}),
  setTimeout() {},
  fetch: () => new Promise(() => {}),
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
context.fetch = async () => ({ok:true, json:async () => data});
(async () => {
  await vm.runInContext('connect()', context);
  await vm.runInContext('connect()', context);
  data = {...data, log:[...data.log, '|turn|2']};
  await vm.runInContext('connect()', context);
  data = {...data, room:'battle-gen9doublesou-2'};
  await vm.runInContext('connect()', context);
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
    assert appended[:3] == [PREVIEW + ["|start"] + SWITCHES, [], ["|turn|2"]]
    assert events[events.index(["destroy"]) - 1] == ["pause"]
