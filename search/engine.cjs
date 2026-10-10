/* Determinized one-turn search over the pinned official Showdown simulator.
 *
 * JSON lines on stdin -> JSON lines on stdout. Each "search" message carries one
 * or more determinized worlds (our exact team + sampled opponent sets) plus the
 * public field state. For each world the engine rebuilds the battle, simulates
 * every (our joint action, their joint action) pair for one turn, scores the
 * resulting positions, solves the simultaneous-move game and returns per-action
 * values for our candidate choices. We are always p1 inside the engine.
 */
'use strict';
const readline = require('node:readline');
const path = require('node:path');
const PS = path.resolve(__dirname, '..', 'node_modules', 'pokemon-showdown');
const {Battle, Teams, Dex} = require(PS);
const {State} = require(path.join(PS, 'dist', 'sim', 'state'));
const {PRNG} = require(path.join(PS, 'dist', 'sim', 'prng'));

const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const emit = v => process.stdout.write(JSON.stringify(v) + '\n');

// ------------------------------------------------------------------ building
function buildBattle(format, world, field, seed) {
  const battle = new Battle({formatid: format, seed: seed || [1, 2, 3, 4]});
  battle.setPlayer('p1', {name: 'Us', team: Teams.pack(world.p1.map(m => m.set))});
  battle.setPlayer('p2', {name: 'Them', team: Teams.pack(world.p2.map(m => m.set))});
  const order = n => Array.from({length: n}, (_, i) => i + 1).join('');
  if (battle.requestState === 'teampreview') {
    battle.choose('p1', 'team ' + order(Math.min(4, world.p1.length)));
    battle.choose('p2', 'team ' + order(Math.min(4, world.p2.length)));
  }
  for (const [i, sideId] of ['p1', 'p2'].entries()) {
    const side = battle.sides[i];
    const states = world[sideId];
    let megaUsed = false;
    side.pokemon.forEach((pokemon, j) => {
      const st = states[j] && states[j].state;
      if (!st) return;
      if (st.mega) megaUsed = true;
      applyMon(battle, pokemon, st);
    });
    if (megaUsed || (field.mega_used && field.mega_used[sideId])) {
      for (const pokemon of side.pokemon) pokemon.canMegaEvo = false;
    }
    side.pokemonLeft = side.pokemon.filter(p => !p.fainted).length;
  }
  applyField(battle, field);
  battle.turn = Math.max(1, field.turn || 1);
  battle.log = [];
  return battle;
}

function applyMon(battle, pokemon, st) {
  if (st.mega && !pokemon.species.isMega) {
    const mega = pokemon.canMegaEvo || battle.actions.canMegaEvo(pokemon);
    if (mega) {
      pokemon.formeChange(mega, pokemon.getItem(), true);
    }
  }
  if (st.ability) {
    const id = toID(st.ability);
    if (id && id !== pokemon.ability && !st.mega) {
      pokemon.ability = id; pokemon.baseAbility = id;
      pokemon.abilityState = battle.initEffectState({id, target: pokemon});
    }
  }
  if (st.item !== undefined && st.item !== null) {
    const id = toID(st.item);
    pokemon.item = id;
    pokemon.itemState = battle.initEffectState({id, target: pokemon});
  }
  // Volatiles and boosts: reset whatever the lead switch-in produced.
  pokemon.volatiles = {};
  pokemon.boosts = {atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0};
  for (const [k, v] of Object.entries(st.boosts || {})) if (k in pokemon.boosts) pokemon.boosts[k] = v;
  if (st.fainted) {
    pokemon.hp = 0; pokemon.fainted = true; pokemon.status = 'fnt';
    pokemon.isActive = pokemon.isActive && true;
    pokemon.switchFlag = false;
    return;
  }
  if (typeof st.hp_abs === 'number') pokemon.hp = Math.max(1, Math.min(pokemon.maxhp, st.hp_abs));
  else if (typeof st.hp === 'number') pokemon.hp = Math.max(1, Math.min(pokemon.maxhp, Math.round(st.hp * pokemon.maxhp)));
  pokemon.status = '';
  pokemon.statusState = battle.initEffectState({});
  if (st.status) {
    const id = toID(st.status);
    pokemon.status = id;
    pokemon.statusState = battle.initEffectState({id, target: pokemon});
    if (id === 'slp') {
      const start = [2, 3, 3][Math.floor(Math.random() * 3)];
      pokemon.statusState.startTime = start;
      pokemon.statusState.time = Math.max(1, start - (st.sleep_turns || 0));
    } else if (id === 'tox') {
      pokemon.statusState.stage = Math.max(0, st.toxic_turns || 0);
    } else if (id === 'frz') {
      pokemon.statusState.startTime = 3;
      pokemon.statusState.time = 3;
    }
  }
  if (st.item === '' && toID(pokemon.ability) === 'unburden' && st.item_gone) {
    try { pokemon.addVolatile('unburden'); } catch (e) {}
  }
  pokemon.activeMoveActions = st.actions || 0;
  pokemon.activeTurns = st.active_turns || 0;
  if (st.pp) {
    for (const slot of pokemon.moveSlots) {
      if (slot.id in st.pp) slot.pp = st.pp[slot.id];
    }
  }
  if (st.last_move) {
    const move = battle.dex.moves.get(st.last_move);
    if (move.exists) { pokemon.lastMove = move; pokemon.moveThisTurn = ''; }
  }
  if (!pokemon.isActive) return;
  const vol = st.volatiles || {};
  const add = (id, extra) => {
    try {
      battle.event = null;
      pokemon.volatiles[id] = battle.initEffectState({id, target: pokemon, ...(extra || {})});
    } catch (e) {}
  };
  if (st.protect_last_turn) add('stall', {counter: 3, duration: 2});
  if (vol.substitute) add('substitute', {hp: Math.floor(pokemon.maxhp / 4)});
  if (vol.confusion) add('confusion', {time: 2 + Math.floor(Math.random() * 2)});
  if (vol.taunt) add('taunt', {duration: Math.max(1, 3 - ((st.turn || 0) - (vol.taunt.turn || 0)))});
  if (vol.encore && vol.encore.move) add('encore', {move: vol.encore.move, duration: 2});
  if (vol.disable && vol.disable.move) add('disable', {move: vol.disable.move, duration: 3});
  if (vol.throatchop) add('throatchop', {duration: 2});
  if (vol.yawn) add('yawn', {duration: 1});
  if (vol.perishsong) add('perishsong', {duration: (vol.perishsong.count || 3) + 1});
  if (vol.focusenergy) add('focusenergy');
  if (vol.dragoncheer) add('dragoncheer');
  if (vol.saltcure) add('saltcure');
  if (vol.charge) add('charge');
  if (vol.healblock) add('healblock', {duration: 2});
  if (vol.leechseed) add('leechseed', {sourceSlot: pokemon.side.foe.active[0] ? pokemon.side.foe.active[0].getSlot() : 'p2a'});
  if (vol.stockpile) add('stockpile', {layers: vol.stockpile.layers || 1});
  if (st.locked_move) add('choicelock', {move: toID(st.locked_move)});
}

function applyField(battle, field) {
  const f = battle.field;
  f.weather = ''; f.weatherState = battle.initEffectState({id: ''});
  if (field.weather && field.weather.id) {
    f.weather = toID(field.weather.id);
    f.weatherState = battle.initEffectState({id: f.weather, duration: field.weather.duration || 0});
    if (!field.weather.duration) delete f.weatherState.duration;
  }
  f.terrain = ''; f.terrainState = battle.initEffectState({id: ''});
  if (field.terrain && field.terrain.id) {
    f.terrain = toID(field.terrain.id);
    f.terrainState = battle.initEffectState({id: f.terrain, duration: field.terrain.duration || 5});
  }
  f.pseudoWeather = {};
  for (const [id, duration] of Object.entries(field.pseudo || {})) {
    f.pseudoWeather[id] = battle.initEffectState({id, duration});
  }
  for (const [i, sideId] of ['p1', 'p2'].entries()) {
    const side = battle.sides[i];
    side.sideConditions = {};
    for (const [id, info] of Object.entries((field.sides || {})[sideId] || {})) {
      const st = {id};
      if (info.duration) st.duration = info.duration;
      if (info.layers) st.layers = info.layers;
      if (id === 'tailwind' || id === 'reflect' || id === 'lightscreen' || id === 'auroraveil' ||
          id === 'stealthrock' || id === 'spikes' || id === 'toxicspikes' || id === 'stickyweb' ||
          id === 'safeguard' || id === 'mist' || id === 'wideguard_never') {
        side.sideConditions[id] = battle.initEffectState(st);
      }
    }
  }
}

// ------------------------------------------------------------------ choices
function targets(target, slot, count) {
  if (count === 1) return [null];
  const allies = [];
  for (let i = 0; i < count; i++) if (i !== slot) allies.push(-(i + 1));
  if (target === 'normal' || target === 'any') return [1, 2, ...allies];
  if (target === 'adjacentFoe') return [1, 2];
  if (target === 'adjacentAlly') return allies;
  if (target === 'adjacentAllyOrSelf') return [-1, -2];
  return [null];
}

/** Legal joint choices for one side of a move request, lightly pruned. */
function sideChoices(battle, side, opts) {
  opts = opts || {};
  const choices = [];
  const perSlot = [];
  const bench = side.pokemon.map((p, i) => [p, i]).filter(([p, i]) => i >= side.active.length && !p.fainted);
  for (let s = 0; s < side.active.length; s++) {
    const pokemon = side.active[s];
    if (!pokemon || pokemon.fainted) { perSlot.push(['pass']); continue; }
    const data = pokemon.getMoveRequestData();
    const options = [];
    const mega = pokemon.canMegaEvo && !opts.noMega;
    (data.moves || []).forEach((m, j) => {
      if (m.disabled) return;
      if (m.pp === 0 && m.id !== 'struggle') return;
      const move = battle.dex.moves.get(m.id);
      const tgts = data.moves.length === 1 && m.id === 'struggle' ? [null] :
        (m.target ? targets(m.target, s, side.active.length) : [null]);
      for (const t of tgts) {
        // Prune attacking our own partner with a damaging move.
        if (t !== null && t < 0 && move.category !== 'Status' && !opts.allowAllyHits) continue;
        // Prune targeting an empty/fainted foe slot when the other foe is alive.
        if (t !== null && t > 0) {
          const foe = side.foe.active[t - 1];
          const other = side.foe.active[2 - t];
          if ((!foe || foe.fainted) && other && !other.fainted) continue;
        }
        let c = 'move ' + (j + 1) + (t !== null ? ' ' + t : '');
        if (mega) c += ' mega';
        options.push(c);
        if (mega && opts.bothMega) options.push('move ' + (j + 1) + (t !== null ? ' ' + t : ''));
      }
    });
    if (!data.trapped && !data.maybeTrapped && !opts.noSwitch) {
      for (const [p, i] of bench) options.push('switch ' + (i + 1));
    }
    perSlot.push(options.length ? options : ['default']);
  }
  const rec = (k, acc) => {
    if (k === perSlot.length) {
      const sw = acc.filter(c => c.startsWith('switch'));
      if (new Set(sw).size !== sw.length) return;
      if (acc.filter(c => c.endsWith(' mega')).length > 1) return;
      choices.push(acc.join(', '));
      return;
    }
    for (const o of perSlot[k]) rec(k + 1, acc.concat([o]));
  };
  rec(0, []);
  // If both slots can mega but only one may, keep variants where either megas.
  if (!choices.length && perSlot.length === 2) {
    for (const a of perSlot[0]) for (const b of perSlot[1]) {
      const a2 = a.replace(/ mega$/, ''), b2 = b.replace(/ mega$/, '');
      choices.push(a + ', ' + b2); if (b !== b2) choices.push(a2 + ', ' + b);
    }
  }
  return Array.from(new Set(choices));
}

function megaFix(choices, side) {
  // When both actives can Mega, sideChoices drops double-mega combos; add single-mega ones.
  if (side.active.length < 2) return choices;
  const [a, b] = side.active;
  if (!(a && b && a.canMegaEvo && b.canMegaEvo)) return choices;
  return choices;
}

// ------------------------------------------------------------------ evaluation
const W = {
  alive: 1.0, hp: 1.2, boost: 0.12, speedBoost: 0.10, status: {brn: 0.25, par: 0.3, slp: 0.45, frz: 0.5, psn: 0.12, tox: 0.25},
  tailwind: 0.10, trickroom: 0.10, screen: 0.06, win: 30,
};

function sideScore(battle, side, sideIndex) {
  let score = 0;
  for (const p of side.pokemon) {
    if (p.fainted || p.hp <= 0) continue;
    let v = W.alive + W.hp * (p.hp / p.maxhp);
    if (p.status && W.status[p.status]) {
      let pen = W.status[p.status];
      if (p.status === 'brn') {
        const atk = p.storedStats.atk, spa = p.storedStats.spa;
        pen = atk > spa ? 0.45 : 0.12;
      }
      if (p.status === 'slp' && p.statusState && p.statusState.time) pen *= Math.min(1, p.statusState.time / 2);
      v -= pen;
    }
    if (p.isActive) {
      const b = p.boosts;
      const atkStat = p.storedStats.atk >= p.storedStats.spa ? 'atk' : 'spa';
      const off = Math.max(-4, Math.min(4, b[atkStat]));
      v += W.boost * off * (0.6 + 0.4 * p.hp / p.maxhp);
      v += 0.05 * Math.max(-3, Math.min(3, b.def + b.spd));
      v += W.speedBoost * Math.max(-2, Math.min(2, b.spe));
      if (p.volatiles.substitute) v += 0.15;
      if (p.volatiles.perishsong) v -= 0.4 * (3 - Math.min(3, (p.volatiles.perishsong.duration || 4) - 1)) / 3 + 0.2;
      if (p.volatiles.confusion) v -= 0.1;
      if (p.volatiles.taunt) v -= 0.05;
      if (p.volatiles.encore) v -= 0.15;
      if (p.volatiles.yawn) v -= 0.25;
      if (p.volatiles.leechseed) v -= 0.1;
    }
    score += v;
  }
  const sc = side.sideConditions;
  if (sc.tailwind) score += W.tailwind * (sc.tailwind.duration || 1);
  for (const s of ['reflect', 'lightscreen', 'auroraveil']) if (sc[s]) score += W.screen * (sc[s].duration || 1);
  return score;
}

function speedEdge(battle) {
  // Positive when Trick Room favours p1 (p1 slower on average among living mons).
  const avg = side => {
    const alive = side.pokemon.filter(p => !p.fainted);
    if (!alive.length) return 0;
    return alive.reduce((a, p) => a + p.getStat('spe', false, true), 0) / alive.length;
  };
  const d = avg(battle.sides[1]) - avg(battle.sides[0]);
  return Math.max(-1, Math.min(1, d / 40));
}

const {features: valueFeatures} = require('./valuefeat.cjs');
const VNET = require('./valuenet.cjs');
const NETS = {};
let CURRENT = {net: null, beta: 0, scale: 8};
function setValue(pathArg, beta, scale) {
  if (!pathArg || !beta) { CURRENT = {net: null, beta: 0, scale: 8}; return; }
  if (!NETS[pathArg]) NETS[pathArg] = VNET.load(pathArg);
  CURRENT = {net: NETS[pathArg], beta, scale: scale || 8};
}

function evaluate(battle) {
  if (CURRENT.net && !battle.ended) {
    const x = valueFeatures(battle, 0);
    const learned = CURRENT.net.hand ? VNET.handUnits(CURRENT.net, x) : CURRENT.scale * VNET.forward(CURRENT.net, x);
    if (CURRENT.beta >= 1) return learned;
    return (1 - CURRENT.beta) * handEvaluate(battle) + CURRENT.beta * learned;
  }
  return handEvaluate(battle);
}

function handEvaluate(battle) {
  if (battle.ended) {
    if (!battle.winner) return 0;
    return battle.winner === battle.sides[0].name ? W.win : -W.win;
  }
  let v = sideScore(battle, battle.sides[0], 0) - sideScore(battle, battle.sides[1], 1);
  const tr = battle.field.pseudoWeather.trickroom;
  if (tr) v += W.trickroom * (tr.duration || 1) * speedEdge(battle) * 2;
  return v;
}

// ------------------------------------------------------------------ game solving
function solveZeroSum(M, iters) {
  // Regret matching for the row player (maximizer) and column player (minimizer).
  const n = M.length, m = M[0].length;
  const rr = new Float64Array(n), cr = new Float64Array(m);
  const rs = new Float64Array(n), cs = new Float64Array(m);
  const x = new Float64Array(n), y = new Float64Array(m);
  for (let t = 0; t < iters; t++) {
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { x[i] = Math.max(0, rr[i]); sx += x[i]; }
    for (let j = 0; j < m; j++) { y[j] = Math.max(0, cr[j]); sy += y[j]; }
    for (let i = 0; i < n; i++) x[i] = sx > 0 ? x[i] / sx : 1 / n;
    for (let j = 0; j < m; j++) y[j] = sy > 0 ? y[j] / sy : 1 / m;
    const rowVal = new Float64Array(n), colVal = new Float64Array(m);
    let v = 0;
    for (let i = 0; i < n; i++) {
      const Mi = M[i];
      let s = 0;
      for (let j = 0; j < m; j++) { s += Mi[j] * y[j]; colVal[j] += x[i] * Mi[j]; }
      rowVal[i] = s; v += x[i] * s;
    }
    for (let i = 0; i < n; i++) { rr[i] += rowVal[i] - v; rs[i] += x[i]; }
    for (let j = 0; j < m; j++) { cr[j] += v - colVal[j]; cs[j] += y[j]; }
  }
  const norm = a => { const s = a.reduce((p, q) => p + q, 0); return Array.from(a, q => q / s); };
  return {x: norm(rs), y: norm(cs)};
}

// ------------------------------------------------------------------ search
function searchWorld(msg, world, wi) {
  const t0 = Date.now();
  const battle = buildBattle(msg.format, world, msg.field, [wi + 1, 7, 13, 19]);
  const ours = (msg.our_choices || sideChoices(battle, battle.sides[0])).slice();
  let theirs = sideChoices(battle, battle.sides[1], {noSwitch: !!msg.opp_no_switch});
  if (msg.max_opp && theirs.length > msg.max_opp) {
    theirs = theirs.slice(0, msg.max_opp);
  }
  const base = State.serializeBattle(battle);
  const baseStr = JSON.stringify(base);
  const seeds = msg.seeds || 1;
  let screenSims = 0;
  const screenErr = {};
  // Common random numbers (msg.crn): a cell's seed depends on the world, the
  // seed index and the OPPONENT action only, so every one of our rows faces the
  // same chance stream for a given reply. Without it the seed also depends on
  // our row, and close actions are ranked partly by luck (paper, Prop. mc).
  const crn = !!msg.crn;
  const sim1 = (a, b, k, seed) => {
    const bb = State.deserializeBattle(baseStr);
    bb.prng = new PRNG(crn && seed ? seed : [wi * 131 + k * 17 + 1, a.length + 3, b.length + 5, 11]);
    if (!bb.choose('p1', a)) { screenErr.p1 = screenErr.p1 || (a + ' :: ' + bb.sides[0].choice.error); return null; }
    if (!bb.choose('p2', b)) { screenErr.p2 = screenErr.p2 || (b + ' :: ' + bb.sides[1].choice.error); return undefined; }
    screenSims++;
    return evaluate(bb);
  };
  if (msg.screen && (theirs.length > msg.screen.keep_opp || ours.length > msg.screen.keep_ours)) {
    // Screening: rank their options against a few probes of ours, then rank ours
    // against their strongest screened options. Full matrix only on survivors.
    const probes = [];
    const step = Math.max(1, Math.floor(ours.length / msg.screen.probe));
    for (let i = 0; i < ours.length && probes.length < msg.screen.probe; i += step) probes.push(ours[i]);
    if (theirs.length > msg.screen.keep_opp) {
      const sc = theirs.map((b, j) => {
        let s = 0, n = 0;
        for (const [pi, a] of probes.entries()) { const v = sim1(a, b, j, [wi * 131 + 1, pi + 3, 7, 13]); if (v === undefined) return [b, Infinity]; if (v !== null) { s += v; n++; } }
        return [b, n ? s / n : Infinity];
      });
      sc.sort((p, q) => p[1] - q[1]);
      theirs = sc.filter(p => p[1] !== Infinity).slice(0, msg.screen.keep_opp).map(p => p[0]);
    }
    if (ours.length > msg.screen.keep_ours) {
      const top = theirs.slice(0, msg.screen.probe);
      const sc = ours.map((a, i) => {
        let s = 0, n = 0;
        for (const [ti, b] of top.entries()) { const v = sim1(a, b, i, [wi * 131 + 1, 3, ti + 5, 13]); if (v === null) return [a, -Infinity]; if (v !== undefined) { s += v; n++; } }
        return [a, n ? s / n : -Infinity];
      });
      sc.sort((p, q) => q[1] - p[1]);
      ours.splice(0, ours.length, ...sc.filter(p => p[1] !== -Infinity).slice(0, msg.screen.keep_ours).map(p => p[0]));
    }
  }
  const M = ours.map(() => new Float64Array(theirs.length));
  const validOurs = new Array(ours.length).fill(true);
  const validTheirs = new Array(theirs.length).fill(true);
  let sims = 0;
  const firstErr = {};
  for (let i = 0; i < ours.length; i++) {
    if (!validOurs[i]) continue;
    for (let j = 0; j < theirs.length; j++) {
      if (!validTheirs[j]) continue;
      let total = 0;
      for (let k = 0; k < seeds; k++) {
        const b = State.deserializeBattle(baseStr);
        b.prng = new PRNG([wi * 131 + k * 17 + 1, crn ? 3 : i + 3, j + 5, 11]);
        if (!b.choose('p1', ours[i])) { validOurs[i] = false; if (!firstErr.p1) firstErr.p1 = ours[i] + ' :: ' + b.sides[0].choice.error; break; }
        if (!b.choose('p2', theirs[j])) { validTheirs[j] = false; if (!firstErr.p2) firstErr.p2 = theirs[j] + ' :: ' + b.sides[1].choice.error; break; }
        total += evaluate(b);
        sims++;
      }
      if (!validOurs[i]) break;
      M[i][j] = total / seeds;
    }
  }
  const rows = ours.map((c, i) => i).filter(i => validOurs[i]);
  const cols = theirs.map((c, j) => j).filter(j => validTheirs[j]);
  if (!rows.length || !cols.length) return {error: 'no valid actions ' + JSON.stringify([firstErr, screenErr, ours.length, theirs.length]).slice(0, 400), ours: ours.slice(0, 4), theirs: theirs.length};
  const sub = rows.map(i => cols.map(j => M[i][j]));
  const {x, y} = solveZeroSum(sub, msg.iters || 2000);
  // Opponent model: blend of equilibrium play and a softmax "greedy" prior that
  // assumes they pick actions that are good for them against our uniform play.
  const colMean = cols.map((j, c) => rows.reduce((a, i, r) => a + sub[r][c], 0) / rows.length);
  const tau = msg.opp_tau || 1.0;
  const mn = Math.min(...colMean);
  const pri = colMean.map(v => Math.exp(-(v - mn) / tau));
  const ps = pri.reduce((a, b) => a + b, 0);
  const alpha = msg.alpha === undefined ? 0.5 : msg.alpha;
  const q = y.map((yy, c) => alpha * yy + (1 - alpha) * pri[c] / ps);
  const values = {};
  const worst = {};
  const nashValue = {};
  rows.forEach((i, r) => {
    values[ours[i]] = sub[r].reduce((a, v, c) => a + v * q[c], 0);
    worst[ours[i]] = Math.min(...sub[r]);
    nashValue[ours[i]] = sub[r].reduce((a, v, c) => a + v * y[c], 0);
  });
  const ourMix = {};
  rows.forEach((i, r) => { ourMix[ours[i]] = x[r]; });
  const topOpp = cols.map((j, c) => [theirs[j], q[c]]).sort((a, b) => b[1] - a[1]).slice(0, 5);
  // Root statistics for learning: the one-turn game value under the averaged
  // strategies, and the static evaluation of the rebuilt (pre-turn) position.
  let eq = 0;
  rows.forEach((i, r) => { for (let c = 0; c < cols.length; c++) eq += x[r] * sub[r][c] * y[c]; });
  const root = {eq, hand: handEvaluate(battle)};
  if (msg.features) root.x = valueFeatures(battle, 0);
  return {values, worst, nash: nashValue, mix: ourMix, ms: Date.now() - t0, sims: sims + screenSims,
          n_ours: rows.length, n_theirs: cols.length, top_opp: topOpp, root};
}

function combos(n, k) {
  const out = [];
  const rec = (start, acc) => {
    if (acc.length === k) { out.push(acc.slice()); return; }
    for (let i = start; i < n; i++) { acc.push(i); rec(i + 1, acc); acc.pop(); }
  };
  rec(0, []);
  return out;
}

function previewPlans(msg) {
  // Score every (bring-4, lead pair) plan by the value of the turn-1 position,
  // averaged over sampled opponent sets and opponent selections.
  let s = msg.seed || 1;
  const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
  const plans = [];
  for (const four of combos(msg.ours.length, 4)) {
    for (const leads of combos(4, 2)) {
      const lead = leads.map(i => four[i]);
      const back = four.filter(i => !lead.includes(i));
      plans.push([...lead, ...back]);
    }
  }
  const oppPlans = [];
  for (const w of msg.worlds) {
    for (let k = 0; k < (msg.samples_per_world || 3); k++) {
      const idx = w.map((_, i) => i);
      const chosen = [];
      const weights = (msg.opp_weights || idx.map(() => 1)).slice();
      while (chosen.length < 4 && idx.length) {
        const tot = idx.reduce((a, i) => a + weights[i], 0);
        let r = rnd() * tot, pick = idx[idx.length - 1];
        for (const i of idx) { r -= weights[i]; if (r <= 0) { pick = i; break; } }
        chosen.push(pick); idx.splice(idx.indexOf(pick), 1);
      }
      for (let i = chosen.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [chosen[i], chosen[j]] = [chosen[j], chosen[i]]; }
      oppPlans.push(chosen.map(i => w[i]));
    }
  }
  const values = {};
  for (const plan of plans) {
    let total = 0, n = 0;
    for (const opp of oppPlans) {
      try {
        const battle = new Battle({formatid: msg.format, seed: [n + 1, 2, 3, 4]});
        battle.setPlayer('p1', {name: 'Us', team: Teams.pack(plan.map(i => msg.ours[i]))});
        battle.setPlayer('p2', {name: 'Them', team: Teams.pack(opp)});
        battle.choose('p1', 'team 1234'); battle.choose('p2', 'team 1234');
        total += evaluate(battle); n++;
      } catch (e) {}
    }
    values['team ' + plan.map(i => i + 1).join(',')] = n ? total / n : -1e9;
  }
  return values;
}

function handle(msg) {
  if (msg.type === 'search') {
    setValue(msg.value_path, msg.value_beta, msg.value_scale);
    const results = msg.worlds.map((w, i) => {
      try { return searchWorld(msg, w, (msg.world_offset || 0) + i); } catch (e) { return {error: String(e && e.stack || e).slice(0, 800)}; }
    });
    return {id: msg.id, results};
  }
  if (msg.type === 'preview') {
    setValue(msg.value_path, msg.value_beta, msg.value_scale);
    return {id: msg.id, values: previewPlans(msg)};
  }
  if (msg.type === 'features') {
    // Value-learning rows for rebuilt public positions: features from our side
    // (p1) and the hand evaluation, one entry per determinized world.
    return {id: msg.id, results: msg.worlds.map(w => {
      try {
        const battle = buildBattle(msg.format, w, msg.field);
        return {x: valueFeatures(battle, 0), hand: handEvaluate(battle)};
      } catch (e) { return {error: String(e && e.message || e).slice(0, 400)}; }
    })};
  }
  if (msg.type === 'choices') {
    const battle = buildBattle(msg.format, msg.worlds[0], msg.field);
    return {id: msg.id, p1: sideChoices(battle, battle.sides[0]), p2: sideChoices(battle, battle.sides[1])};
  }
  if (msg.type === 'debug') {
    const battle = buildBattle(msg.format, msg.worlds[0], msg.field);
    const info = s => s.pokemon.map(p => ({name: p.name, species: p.species.name, hp: p.hp, maxhp: p.maxhp, status: p.status,
      item: p.item, ability: p.ability, active: p.isActive, fainted: p.fainted, boosts: p.boosts, vol: Object.keys(p.volatiles),
      canMega: p.canMegaEvo, moves: p.moveSlots.map(m => m.id + ':' + m.pp)}));
    return {id: msg.id, p1: info(battle.sides[0]), p2: info(battle.sides[1]), field: {weather: battle.field.weather,
      wd: battle.field.weatherState.duration, terrain: battle.field.terrain, pseudo: Object.keys(battle.field.pseudoWeather),
      sides: battle.sides.map(s => Object.keys(s.sideConditions))}, turn: battle.turn, value: evaluate(battle),
      p1choices: sideChoices(battle, battle.sides[0]).length, p2choices: sideChoices(battle, battle.sides[1]).length};
  }
  if (msg.type === 'ping') return {id: msg.id, ok: true};
  throw new Error('unknown message type');
}

module.exports = {buildBattle, sideChoices, evaluate, handEvaluate, solveZeroSum, searchWorld, handle, State, PRNG, Battle, Teams, Dex};

if (require.main === module) {
const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  let msg;
  try { msg = JSON.parse(line); } catch (e) { emit({error: 'bad json'}); return; }
  try { emit(handle(msg)); } catch (e) { emit({id: msg.id, error: String(e && e.stack || e).slice(0, 1500)}); }
});
}
