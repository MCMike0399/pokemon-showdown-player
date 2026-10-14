/* Features of one joint action for the human opponent policy (search/human).
 *
 * The screening features (search/screenfeat.cjs: 70 per joint action, 146 for the
 * position) describe what an action does; ladder players also pick by what the
 * move IS (Tailwind vs Trick Room are both "field setters" there) and by what
 * they are using. So the policy adds one-hot move and species identities per
 * slot from a fixed vocabulary (search/human/vocab.json, from the format's usage
 * statistics). The engine computes these for the opponent's enumerated actions
 * (search/engine.cjs `opp_model`), the extractor for logged human actions.
 */
'use strict';
const path = require('node:path');
const SF = require('../screenfeat.cjs');
const VOCAB = require(path.join(__dirname, 'vocab.json'));
const VERSION = 2;
const MOVE_IDX = new Map(VOCAB.moves.map((m, i) => [m, i]));
const SPECIES_IDX = new Map(VOCAB.species.map((s, i) => [s.toLowerCase().replace(/[^a-z0-9]/g, ''), i]));
const NM = VOCAB.moves.length + 1, NS = VOCAB.species.length + 1;
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const DIM = SF.ACTION_DIM + 2 * (NM + NS);

function partMove(part, pokemon) {
  const words = String(part).trim().split(/\s+/);
  if (words[0] !== 'move' || !pokemon) return null;
  const ref = words[1];
  if (/^\d+$/.test(ref)) {
    const moves = pokemon.getMoveRequestData().moves || [];
    const m = moves[+ref - 1];
    return m ? m.id : null;
  }
  return toID(ref);
}

/** Per slot: one-hot of the move used (or switch/pass bucket) and of the acting species. */
function identity(battle, side, slot, part) {
  const out = new Array(NM + NS).fill(0);
  const me = side.active[slot];
  const alive = me && !me.fainted;
  const mid = alive ? partMove(part, me) : null;
  out[mid !== null && MOVE_IDX.has(mid) ? MOVE_IDX.get(mid) : NM - 1] = 1;
  const sid = alive ? toID(me.species.baseSpecies || me.species.name) : '';
  const sidx = SPECIES_IDX.has(toID(me && alive ? me.species.name : '')) ? SPECIES_IDX.get(toID(me.species.name)) :
    (SPECIES_IDX.has(sid) ? SPECIES_IDX.get(sid) : NS - 1);
  out[NM + sidx] = 1;
  return out;
}

function actionFeatures(battle, sideIdx, choice) {
  const side = battle.sides[sideIdx];
  const parts = String(choice).split(',');
  const base = SF.actionFeatures(battle, sideIdx, choice);
  return [...base, ...identity(battle, side, 0, parts[0] || 'pass'), ...identity(battle, side, 1, parts[1] || 'pass')];
}

module.exports = {VERSION, DIM, actionFeatures, stateFeatures: SF.stateFeatures};
