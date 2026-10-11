/* Features of one joint action, for the learned screening ranker.
 *
 * The probe screening in engine.cjs simulates every candidate action against a
 * few replies only to rank them. The ranker (search/train_screen.py, trained on
 * MPS from those probe results) predicts the same ranking from the position and
 * these action features, so the engine can skip most probe simulations.
 *
 * A joint action is one choice string for a doubles side, e.g.
 * "move icebeam 1 mega, switch 3" (moves by id, as the agent sends ours) or
 * "move 2 1, move 1" (moves by request slot, as the engine enumerates theirs).
 * Features are from the choosing side's perspective: target 1/2 are foe slots,
 * -1/-2 our own slots.
 */
'use strict';
const {features: stateFeatures, effSpeed, boostMul} = require('./valuefeat.cjs');

const VERSION = 1;
const SLOT = 31;
const JOINT = 8;
const ACTION_DIM = 2 * SLOT + JOINT;
const SUPPORT = new Set(['followme', 'ragepowder', 'helpinghand', 'allyswitch', 'spotlight']);
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');

function moveDamage(battle, att, def, move) {
  // Fraction of def's max HP one hit of `move` deals (same rough model as valuefeat.damageProxy).
  if (!att || !def || att.fainted || def.fainted || move.category === 'Status') return 0;
  let bp = move.basePower || 0;
  let type = move.type;
  if (move.id === 'weatherball' && battle.field.weather) {
    bp = 100;
    type = {raindance: 'Water', sunnyday: 'Fire', sandstorm: 'Rock', snowscape: 'Ice'}[battle.field.weather] || type;
  }
  if (!bp || !battle.dex.getImmunity(type, def.getTypes())) return 0;
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
  return Math.min(1.5, dmg / def.maxhp);
}

function threat(battle, attackers, def) {
  // Best single-move damage any of `attackers` deals to `def`.
  let best = 0;
  for (const a of attackers) {
    if (!a || a.fainted) continue;
    for (const slot of a.moveSlots) {
      const move = battle.dex.moves.get(slot.id);
      if (move.exists) best = Math.max(best, moveDamage(battle, a, def, move));
    }
  }
  return best;
}

const alive = p => p && !p.fainted && p.hp > 0;
const hpFrac = p => (alive(p) ? p.hp / p.maxhp : 0);

/** Parse one slot's part of a choice into {kind, move, target, mega, switchTo}. */
function parsePart(part, pokemon) {
  const words = part.trim().split(/\s+/);
  const out = {kind: words[0], mega: words.includes('mega'), target: null, move: null, switchTo: null};
  if (out.kind === 'move' && pokemon) {
    const ref = words[1];
    if (/^\d+$/.test(ref)) {
      // Engine enumeration: the slot in the move request (a locked request lists one move).
      const moves = pokemon.getMoveRequestData().moves || [];
      const m = moves[+ref - 1];
      out.move = m ? m.id : null;
    } else {
      out.move = toID(ref);
    }
    const t = words.slice(2).find(w => /^-?\d+$/.test(w));
    if (t !== undefined) out.target = +t;
  } else if (out.kind === 'switch') {
    out.switchTo = +words[1] - 1;
  }
  return out;
}

function slotFeatures(battle, side, slot, part, acc) {
  const f = new Array(SLOT).fill(0);
  const me = side.active[slot];
  const foes = side.foe.active.slice(0, 2);
  const p = parsePart(part, alive(me) ? me : null);
  if (p.kind === 'switch') {
    const incoming = side.pokemon[p.switchTo];
    f[1] = 1;
    f[26] = hpFrac(incoming);
    let best = 0;
    for (const foe of foes) best = Math.max(best, threat(battle, [incoming], foe));
    f[27] = best;
    f[28] = threat(battle, foes, incoming);
    f[29] = alive(me) ? threat(battle, foes, me) : 0;
    f[30] = hpFrac(me);
    return f;
  }
  if (p.kind !== 'move' || !alive(me) || !p.move) {
    f[2] = 1;
    return f;
  }
  const move = battle.dex.moves.get(p.move);
  f[0] = 1;
  f[3] = p.mega ? 1 : 0;
  if (!move.exists) return f;
  f[4] = move.category === 'Physical' ? 1 : 0;
  f[5] = move.category === 'Special' ? 1 : 0;
  f[6] = move.category === 'Status' ? 1 : 0;
  f[7] = (move.basePower || 0) / 150;
  f[8] = Math.max(-1, Math.min(1, (move.priority || 0) / 3));
  f[9] = move.accuracy === true ? 1 : (move.accuracy || 100) / 100;
  const spread = ['allAdjacentFoes', 'allAdjacent'].includes(move.target);
  f[10] = spread ? 1 : 0;
  f[11] = p.target !== null && p.target < 0 ? 1 : 0;
  f[12] = move.stallingMove ? 1 : 0;
  f[13] = move.id === 'fakeout' ? 1 : 0;
  f[14] = move.weather || move.terrain || move.pseudoWeather || move.sideCondition ? 1 : 0;
  f[15] = move.boosts && ['self', 'adjacentAllyOrSelf', 'allies'].includes(move.target) ? 1 : 0;
  f[16] = move.category === 'Status' && (move.status || move.volatileStatus) && !move.stallingMove &&
    !['self', 'adjacentAlly', 'adjacentAllyOrSelf', 'allySide', 'allies'].includes(move.target) ? 1 : 0;
  f[17] = move.heal || (move.flags && move.flags.heal) ? 1 : 0;
  f[18] = move.selfSwitch ? 1 : 0;
  f[19] = SUPPORT.has(move.id) ? 1 : 0;
  f[24] = hpFrac(me);
  if (move.category === 'Status') return f;
  // Damage: spread moves hit both foes; a single-target move hits its target,
  // or the first living foe when the choice names none.
  let targets = [];
  if (spread) targets = [0, 1];
  else if (p.target !== null && p.target > 0) targets = [p.target - 1];
  else if (p.target === null) targets = [alive(foes[0]) ? 0 : 1];
  const dmg = targets.map(i => moveDamage(battle, me, foes[i], move));
  if (targets.length) {
    f[20] = dmg[0];
    f[21] = alive(foes[targets[0]]) && dmg[0] >= hpFrac(foes[targets[0]]) ? 1 : 0;
    f[22] = dmg.length > 1 ? dmg[1] : 0;
    const foe = foes[targets[0]];
    if (alive(foe)) {
      const tr = battle.field.pseudoWeather.trickroom;
      const d = effSpeed(battle, me) - effSpeed(battle, foe);
      f[23] = Math.sign(tr ? -d : d);
    }
    f[25] = hpFrac(foe);
  }
  targets.forEach((t, k) => { acc.damage[t] += dmg[k]; acc.targeted[t] += 1; });
  return f;
}

/** Feature vector of one joint choice for side `sideIdx` (0 = p1). */
function actionFeatures(battle, sideIdx, choice) {
  const side = battle.sides[sideIdx];
  const parts = String(choice).split(',');
  const acc = {damage: [0, 0], targeted: [0, 0]};
  const out = [];
  for (let s = 0; s < 2; s++) out.push(...slotFeatures(battle, side, s, parts[s] || 'pass', acc));
  const foes = side.foe.active.slice(0, 2);
  const a = out.slice(0, SLOT), b = out.slice(SLOT);
  out.push(
    acc.targeted[0] > 1 || acc.targeted[1] > 1 ? 1 : 0,
    acc.damage[0], acc.damage[1],
    alive(foes[0]) && acc.damage[0] >= hpFrac(foes[0]) ? 1 : 0,
    alive(foes[1]) && acc.damage[1] >= hpFrac(foes[1]) ? 1 : 0,
    a[12] && b[12] ? 1 : 0,
    a[1] && b[1] ? 1 : 0,
    a[3] + b[3]);
  return out;
}

module.exports = {VERSION, ACTION_DIM, actionFeatures, stateFeatures, moveDamage};
