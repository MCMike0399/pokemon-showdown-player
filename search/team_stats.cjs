/* Exact level-50 stats of saved teams from the pinned simulator (stdin: {name: team}). */
'use strict';
const path = require('node:path');
const PS = path.resolve(__dirname, '..', 'node_modules', 'pokemon-showdown');
const {Battle, Teams} = require(PS);
const teams = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const out = {};
for (const [name, t] of Object.entries(teams)) {
  try {
    const b = new Battle({formatid: t.format, seed: [1, 2, 3, 4]});
    b.setPlayer('p1', {name: 'a', team: Teams.pack(t.sets)});
    b.setPlayer('p2', {name: 'b', team: Teams.pack(t.sets)});
    out[name] = b.sides[0].pokemon.map(p => [p.baseSpecies.baseSpecies.toLowerCase().replace(/[^a-z0-9]/g, ''),
                                             {...p.storedStats, hp: p.maxhp}]);
  } catch (e) { out[name] = null; }
}
process.stdout.write(JSON.stringify(out));
