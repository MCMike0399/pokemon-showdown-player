/* Fast omniscient self-play for value-function data.
 *
 * usage: node search/selfplay.cjs <teams.json> <focus-team.json> <games> <seed> <out.jsonl> [value.json]
 *        [--focus-prob P] [--value-beta B] [--tag NAME]
 *
 * With probability P (default 0.5) one side, chosen at random, plays the focus
 * (deployed) team and the other a pool team; otherwise both come from the pool.
 * Each line is one turn-start position from one side's perspective:
 *   {x: features, y: outcome (1 win, 0 loss, .5 tie), t: turn index, g: game id,
 *    s: side 0/1, f: 1 if that side is the focus team, h: hand evaluation,
 *    v: one-turn equilibrium value of the search that chose the turn}
 * h and v are from the same side's perspective, so learned values can be
 * compared with, or bootstrapped from, the hand evaluation the search uses.
 */
'use strict';
const fs = require('node:fs');
const E = require('./engine.cjs');
const {features} = require('./valuefeat.cjs');
const V = require('./valuenet.cjs');

const argv = process.argv.slice(2);
const flags = {};
const pos = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i].startsWith('--')) flags[argv[i].slice(2)] = argv[++i];
  else pos.push(argv[i]);
}
const [teamsPath, focusPath, gamesArg, seedArg, outPath, valuePath] = pos;
const teams = JSON.parse(fs.readFileSync(teamsPath, 'utf8')).map(t => t.sets || t);
const focusRaw = JSON.parse(fs.readFileSync(focusPath, 'utf8'));
const focus = focusRaw.sets || focusRaw;
const focusProb = flags['focus-prob'] === undefined ? 0.5 : +flags['focus-prob'];
const valueBeta = flags['value-beta'] === undefined ? 1 : +flags['value-beta'];
const tag = flags.tag || 'sp';
const games = +gamesArg;
let seed = +seedArg;
const seed0 = seed;

const net = valuePath ? V.load(valuePath) : null;
const FMT = 'gen9championsvgc2026regmc';

function rng() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }
const pick = a => a[Math.floor(rng() * a.length)];

function value(b) {
  if (b.ended) return b.winner === b.sides[0].name ? 30 : b.winner ? -30 : 0;
  const hand = E.handEvaluate(b);
  if (!net) return hand;
  const x = features(b, 0);
  const learned = net.hand ? V.handUnits(net, x) : 8 * V.forward(net, x);
  return valueBeta >= 1 ? learned : (1 - valueBeta) * hand + valueBeta * learned;
}

function sample(p) {
  let r = rng(), acc = 0;
  for (let i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
  return p.length - 1;
}

function chooseTurn(battle) {
  const base = JSON.stringify(E.State.serializeBattle(battle));
  let A = E.sideChoices(battle, battle.sides[0]);
  let B = E.sideChoices(battle, battle.sides[1]);
  const sim = (a, b) => {
    const c = E.State.deserializeBattle(base);
    c.prng = new E.PRNG([Math.floor(rng() * 65535), 1, 2, 3]);
    if (!c.choose('p1', a)) return null;
    if (!c.choose('p2', b)) return undefined;
    return value(c);
  };
  const K = 10;
  const screen = (mine, other, sign, isP1) => {
    if (mine.length <= K) return mine;
    const probes = [];
    for (let i = 0; i < 3; i++) probes.push(pick(other));
    const sc = mine.map(m => {
      let s = 0, n = 0;
      for (const o of probes) {
        const v = isP1 ? sim(m, o) : sim(o, m);
        if (v === null || v === undefined) { if ((isP1 && v === null) || (!isP1 && v === undefined)) return [m, -1e9]; continue; }
        s += sign * v; n++;
      }
      return [m, n ? s / n : -1e9];
    }).sort((x, y) => y[1] - x[1]);
    // Keep the best plus a couple of random ones for diversity.
    const keep = sc.slice(0, K - 2).map(x => x[0]);
    const rest = sc.slice(K - 2).filter(x => x[1] > -1e9);
    for (let i = 0; i < 2 && rest.length; i++) keep.push(rest.splice(Math.floor(rng() * rest.length), 1)[0][0]);
    return keep;
  };
  A = screen(A, B, 1, true);
  B = screen(B, A, -1, false);
  const M = A.map(a => B.map(b => { const v = sim(a, b); return v === null || v === undefined ? NaN : v; }));
  const rows = A.map((_, i) => i).filter(i => M[i].some(v => !Number.isNaN(v)));
  const cols = B.map((_, j) => j).filter(j => rows.every(i => !Number.isNaN(M[i][j])));
  if (!rows.length || !cols.length) return {a: pick(A), b: pick(B), v: null};
  const sub = rows.map(i => cols.map(j => Number.isNaN(M[i][j]) ? -30 : M[i][j]));
  const {x, y} = E.solveZeroSum(sub, 400);
  let v = 0;
  rows.forEach((_, r) => cols.forEach((_, c) => { v += x[r] * sub[r][c] * y[c]; }));
  const eps = 0.08;
  const xa = x.map(p => (1 - eps) * p + eps / x.length), yb = y.map(p => (1 - eps) * p + eps / y.length);
  return {a: A[rows[sample(xa)]], b: B[cols[sample(yb)]], v};
}

function forced(battle, side) {
  const sw = side.pokemon.map((p, i) => [p, i]).filter(([p, i]) => i >= side.active.length && !p.fainted);
  const parts = [];
  const used = new Set();
  side.active.forEach(p => {
    if (p && p.switchFlag) {
      const opts = sw.filter(([, i]) => !used.has(i));
      if (!opts.length) { parts.push('pass'); return; }
      const [, i] = pick(opts); used.add(i); parts.push('switch ' + (i + 1));
    } else parts.push('pass');
  });
  return parts.join(', ');
}

function playOne(gameIndex) {
  let t1 = pick(teams), t2 = pick(teams);
  const isFocus = [false, false];
  if (rng() < focusProb) {
    const side = rng() < 0.5 ? 0 : 1;
    if (side === 0) t1 = focus; else t2 = focus;
    isFocus[side] = true;
  }
  const g = `${tag}-${seed0}-${gameIndex}`;
  const battle = new E.Battle({formatid: FMT, seed: [Math.floor(rng() * 65535), 3, 5, 7]});
  battle.setPlayer('p1', {name: 'A', team: E.Teams.pack(t1)});
  battle.setPlayer('p2', {name: 'B', team: E.Teams.pack(t2)});
  const order = () => { const idx = [1, 2, 3, 4, 5, 6]; for (let i = 5; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } return 'team ' + idx.slice(0, 4).join(''); };
  battle.choose('p1', order()); battle.choose('p2', order());
  const states = [];
  let guard = 0;
  while (!battle.ended && guard++ < 60) {
    if (battle.requestState === 'move') {
      const hand = E.handEvaluate(battle);
      const turn = {x0: features(battle, 0), x1: features(battle, 1), h: hand, t: battle.turn};
      const {a, b, v} = chooseTurn(battle);
      turn.v = v;
      states.push(turn);
      if (!battle.choose('p1', a)) battle.choose('p1', 'default');
      if (!battle.choose('p2', b)) battle.choose('p2', 'default');
    } else if (battle.requestState === 'switch') {
      for (const [i, s] of battle.sides.entries()) {
        if (s.requestState === 'switch' && !s.isChoiceDone()) {
          if (!battle.choose('p' + (i + 1), forced(battle, s))) battle.choose('p' + (i + 1), 'default');
        }
      }
    } else break;
  }
  if (!battle.ended) return 0;
  const y = battle.winner === 'A' ? 1 : battle.winner === 'B' ? 0 : 0.5;
  const r = n => (n === null || n === undefined ? null : Math.round(n * 1e4) / 1e4);
  let buf = '';
  for (const st of states) {
    buf += JSON.stringify({x: st.x0, y, t: st.t, g, s: 0, f: isFocus[0] ? 1 : 0, h: r(st.h), v: r(st.v)}) + '\n';
    buf += JSON.stringify({x: st.x1, y: 1 - y, t: st.t, g, s: 1, f: isFocus[1] ? 1 : 0, h: r(-st.h), v: st.v === null ? null : r(-st.v)}) + '\n';
  }
  fs.appendFileSync(outPath, buf);
  return states.length * 2;
}

let n = 0;
const t0 = Date.now();
for (let gi = 0; gi < games; gi++) {
  try { n += playOne(gi); } catch (e) { process.stderr.write(String(e && e.stack || e).slice(0, 300) + '\n'); }
  if ((gi + 1) % 10 === 0) process.stderr.write(`games ${gi + 1} samples ${n} ${(Date.now() - t0) / 1000}s\n`);
}
process.stderr.write(`done games ${games} samples ${n} ${(Date.now() - t0) / 1000}s\n`);
