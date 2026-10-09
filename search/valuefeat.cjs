/* Position features for the learned value function, from one side's perspective.
 * Used identically by the self-play data generator and by the search engine. */
'use strict';

const STATUSES = ['brn', 'par', 'slp', 'frz', 'psn', 'tox'];
const WEATHERS = ['raindance', 'sunnyday', 'sandstorm', 'snowscape'];
const TERRAINS = ['grassyterrain', 'psychicterrain', 'electricterrain', 'mistyterrain'];

function damageProxy(battle, att, def) {
  // Rough best fraction of def's max HP att can deal with one move this turn.
  if (!att || !def || att.fainted || def.fainted) return 0;
  let best = 0;
  for (const slot of att.moveSlots) {
    const move = battle.dex.moves.get(slot.id);
    if (!move.exists || move.category === 'Status' || slot.pp <= 0) continue;
    let bp = move.basePower || 0;
    if (!bp) continue;
    let type = move.type;
    if (move.id === 'weatherball' && battle.field.weather) { bp = 100; type = {raindance: 'Water', sunnyday: 'Fire', sandstorm: 'Rock', snowscape: 'Ice'}[battle.field.weather] || type; }
    if (!battle.dex.getImmunity(type, def.getTypes())) continue;
    let eff = battle.dex.getEffectiveness(type, def.getTypes());
    eff = Math.pow(2, Math.max(-2, Math.min(2, eff)));
    const phys = move.category === 'Physical';
    const A = att.storedStats[phys ? 'atk' : 'spa'] * boostMul(att.boosts[phys ? 'atk' : 'spa']);
    const D = def.storedStats[phys ? 'def' : 'spd'] * boostMul(def.boosts[phys ? 'def' : 'spd']);
    const stab = att.getTypes().includes(type) ? 1.5 : 1;
    let w = 1;
    if (battle.field.weather === 'raindance') { if (type === 'Water') w = 1.5; else if (type === 'Fire') w = 0.5; }
    if (battle.field.weather === 'sunnyday') { if (type === 'Fire') w = 1.5; else if (type === 'Water') w = 0.5; }
    const spread = ['allAdjacentFoes', 'allAdjacent'].includes(move.target) ? 0.75 : 1;
    const acc = move.accuracy === true ? 1 : (move.accuracy || 100) / 100;
    const dmg = ((22 * bp * A / Math.max(1, D)) / 50 + 2) * stab * eff * w * spread * 0.925 * acc;
    best = Math.max(best, dmg / def.maxhp);
  }
  return Math.min(1.5, best);
}

function boostMul(b) {
  b = b || 0;
  return b >= 0 ? (2 + b) / 2 : 2 / (2 - b);
}

function effSpeed(battle, p) {
  let s = p.storedStats.spe * boostMul(p.boosts.spe);
  if (p.item === 'choicescarf') s *= 1.5;
  if (p.status === 'par') s *= 0.5;
  if (p.side.sideConditions.tailwind) s *= 2;
  return s;
}

function monFeat(battle, p) {
  if (!p || p.fainted || p.hp <= 0) return new Array(14).fill(0);
  const st = p.storedStats;
  const off = Math.max(st.atk, st.spa);
  const offBoost = st.atk >= st.spa ? p.boosts.atk : p.boosts.spa;
  return [1, p.hp / p.maxhp, p.isActive ? 1 : 0,
    ...STATUSES.map(s => (p.status === s ? 1 : 0)),
    offBoost / 2, p.boosts.spe / 2, (p.boosts.def + p.boosts.spd) / 4,
    effSpeed(battle, p) / 200, off / 200];
}

function sideFeat(battle, side) {
  const actives = side.active.slice(0, 2);
  const bench = side.pokemon.filter(p => !actives.includes(p))
    .sort((a, b) => (b.fainted ? -1 : b.hp / b.maxhp) - (a.fainted ? -1 : a.hp / a.maxhp));
  const mons = [...actives, ...bench].slice(0, 4);
  while (mons.length < 4) mons.push(null);
  const f = [];
  for (const p of mons) f.push(...monFeat(battle, p));
  const sc = side.sideConditions;
  f.push(sc.tailwind ? (sc.tailwind.duration || 1) / 4 : 0,
    sc.reflect ? (sc.reflect.duration || 1) / 5 : 0,
    sc.lightscreen ? (sc.lightscreen.duration || 1) / 5 : 0,
    sc.auroraveil ? (sc.auroraveil.duration || 1) / 5 : 0);
  f.push(side.pokemon.filter(p => !p.fainted).length / 4);
  return f;
}

function features(battle, sideIdx) {
  const us = battle.sides[sideIdx], them = battle.sides[1 - sideIdx];
  const f = [...sideFeat(battle, us), ...sideFeat(battle, them)];
  const fl = battle.field;
  const tr = fl.pseudoWeather.trickroom;
  f.push(tr ? (tr.duration || 1) / 5 : 0);
  for (const w of WEATHERS) f.push(fl.weather === w ? (fl.weatherState.duration || 3) / 5 : 0);
  for (const t of TERRAINS) f.push(fl.terrain === t ? (fl.terrainState.duration || 3) / 5 : 0);
  // Pairwise active matchups: damage pressure and who moves first.
  const ua = us.active.slice(0, 2), ta = them.active.slice(0, 2);
  for (const a of [0, 1]) for (const b of [0, 1]) {
    const x = ua[a], y = ta[b];
    f.push(damageProxy(battle, x, y), damageProxy(battle, y, x));
    let faster = 0;
    if (x && y && !x.fainted && !y.fainted) {
      const d = effSpeed(battle, x) - effSpeed(battle, y);
      faster = Math.sign(tr ? -d : d);
    }
    f.push(faster);
  }
  // Best bench pressure (switch-in threat) both ways.
  let ub = 0, tb = 0;
  for (const p of us.pokemon) if (!p.isActive && !p.fainted) for (const y of ta) ub = Math.max(ub, damageProxy(battle, p, y));
  for (const p of them.pokemon) if (!p.isActive && !p.fainted) for (const x of ua) tb = Math.max(tb, damageProxy(battle, p, x));
  f.push(ub, tb, Math.min(1, (battle.turn || 0) / 20));
  return f;
}

module.exports = {features, damageProxy};
