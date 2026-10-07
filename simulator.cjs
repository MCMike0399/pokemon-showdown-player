/* JSON-lines bridge to a pinned official Showdown simulator. No network/accounts.
 * Only getPlayerStreams().p1/p2 output crosses into Python observations.
 */
const readline = require('node:readline');
const {BattleStream, getPlayerStreams, Dex, Teams, TeamValidator} = require('pokemon-showdown');
const rl = readline.createInterface({input: process.stdin});
const emit = value => process.stdout.write(JSON.stringify(value) + '\n');
let battle;
let streams;
let flushScheduled = false;
const pending = {p1: [], p2: []};
const sequence = {p1: 0, p2: 0};

function flush() {
  flushScheduled = false;
  for (const side of ['p1', 'p2']) {
    const lines = pending[side].splice(0);
    if (!lines.length) continue;
    let request;
    const publicLines = [];
    for (const line of lines) {
      if (line.startsWith('|request|')) request = JSON.parse(line.slice(9));
      else publicLines.push(line);
    }
    if (request) request.rqid = ++sequence[side];
    emit({side, lines: publicLines, request});
  }
}

async function run(input) {
  if (input.type === 'validate') {
    const format = Dex.formats.get(input.format);
    if (!format.exists) throw new Error('Unknown simulator format: ' + input.format);
    const sets = structuredClone(input.sets);
    const errors = new TeamValidator(input.format).validateTeam(sets) || [];
    emit({errors, packed: errors.length ? null : Teams.pack(sets)});
  } else if (input.type === 'import') {
    const format = Dex.formats.get(input.format);
    if (!format.exists) throw new Error('Unknown simulator format: ' + input.format);
    const sets = Teams.import(input.text);
    if (!sets) throw new Error('No team sets in paste');
    const errors = new TeamValidator(input.format).validateTeam(sets) || [];
    emit({errors, sets: errors.length ? null : sets});
  } else if (input.type === 'dex') {
    const format = Dex.formats.get(input.format);
    if (!format.exists) throw new Error('Unknown simulator format: ' + input.format);
    const dex = Dex.forFormat(input.format);
    const pokedex = Object.fromEntries(dex.species.all().map(p => [p.id, {name: p.name, types: p.types, baseStats: p.baseStats}]));
    const moves = Object.fromEntries(dex.moves.all().map(m => [m.id, {name: m.name, basePower: m.basePower,
      accuracy: m.accuracy, type: m.type, category: m.category, priority: m.priority, target: m.target}]));
    const typechart = Object.fromEntries(dex.types.all().map(t => [t.name, {damageTaken: t.damageTaken}]));
    emit({format: input.format, mod: format.mod, dex: {pokedex, moves, typechart}});
  } else if (input.type === 'start') {
    if (battle) throw new Error('A simulator process runs exactly one game');
    const format = Dex.formats.get(input.format);
    if (!format.exists) throw new Error('Unknown simulator format: ' + input.format);
    const options = {formatid: input.format, seed: input.seed};
    const players = [];
    for (const [i, sets] of [input.team1, input.team2].entries()) {
      const name = (i === (input.learnerSide === 'p2' ? 1 : 0)) ? 'LocalBrain' : 'LocalOpponent';
      if (sets) {
        const copy = structuredClone(sets);
        const errors = new TeamValidator(input.format).validateTeam(copy);
        if (errors) throw new Error('Team ' + (i + 1) + ': ' + errors.join('; '));
        players.push({name, team: Teams.pack(copy)});
      } else {
        if (!format.team) throw new Error('This format requires two teams');
        players.push({name});
      }
    }
    battle = new BattleStream();
    streams = getPlayerStreams(battle);
    for (const side of ['p1', 'p2']) {
      (async () => {
        for await (const chunk of streams[side]) {
          pending[side].push(...chunk.split('\n').filter(Boolean));
          if (!flushScheduled) {flushScheduled = true; setImmediate(flush);}
        }
      })().catch(error => emit({error: error.message}));
    }
    await streams.omniscient.write(`>start ${JSON.stringify(options)}\n>player p1 ${JSON.stringify(players[0])}\n>player p2 ${JSON.stringify(players[1])}`);
    // Consent to OTS locally, just as both human players accepting on ladder.
    if (battle.battle.ruleTable.has('openteamsheets')) {
      battle.battle.showOpenTeamSheets();
      battle.battle.sendUpdates();
    }
  } else if (input.type === 'choose') {
    if (!streams || !['p1', 'p2'].includes(input.side)) throw new Error('Invalid player stream');
    await streams[input.side].write(input.choice);
  } else if (input.type === 'stop') {
    process.exit(0);
  } else {
    throw new Error('Unknown bridge operation');
  }
}

let chain = Promise.resolve();
rl.on('line', line => {
  chain = chain.then(() => run(JSON.parse(line))).catch(error => {
    emit({error: error.message});
    process.exitCode = 1;
    rl.close();
  });
});
