// Validate many teams at once: stdin JSON {format, teams:[[sets]]} -> stdout [errors|null]
const path = require('node:path');
const {TeamValidator} = require(path.resolve(__dirname, '..', 'node_modules', 'pokemon-showdown'));
let input = '';
process.stdin.on('data', d => input += d);
process.stdin.on('end', () => {
  const msg = JSON.parse(input);
  const v = new TeamValidator(msg.format);
  const out = msg.teams.map(t => { try { return v.validateTeam(structuredClone(t)) || null; } catch (e) { return [String(e)]; } });
  process.stdout.write(JSON.stringify(out));
});
