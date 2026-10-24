/* A shared root policy against an opponent who knows its sampled hidden set.
 * This solves a finite, one-sided Bayesian game, NOT the full Pokemon game.
 * Every row must have been simulated in every world. Missing cells are errors.
 * There are no simulator calls or trained parameters in this solver.
 */
'use strict';

function distribution(regrets) {
  const p = Array.from(regrets, v => Math.max(0, v));
  const sum = p.reduce((a, b) => a + b, 0);
  return p.map(v => sum ? v / sum : 1 / p.length);
}

function solveBelief(matrices, options = {}) {
  const iterations = options.iterations ?? 4000;
  const tolerance = options.tolerance ?? 0.01;
  const risk = options.risk ?? 0.1;
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 100000 ||
      !Number.isFinite(tolerance) || tolerance < 0 || !Number.isFinite(risk) || risk < 0)
    throw new Error('invalid belief solver options');
  if (!matrices.length || !matrices[0].ours.length) throw new Error('empty belief game');
  const actions = matrices[0].ours;
  const n = actions.length, worlds = matrices.length;
  if (new Set(actions).size !== n) throw new Error('duplicate row labels');
  const games = matrices.map(g => {
    const m = g.theirs.length;
    if (!m || g.ours.length !== n || new Set(g.ours).size !== n || g.payoff.length !== n)
      throw new Error('incomplete belief game');
    const M = actions.map(a => {
      const i = g.ours.indexOf(a), row = g.payoff[i];
      if (!row || row.length !== m || !row.every(Number.isFinite)) throw new Error('missing or invalid payoff');
      return row;
    });
    if (!g.prior || g.prior.length !== m || !g.prior.every(v => Number.isFinite(v) && v >= 0))
      throw new Error('invalid opponent prior');
    const sum = g.prior.reduce((a, b) => a + b, 0);
    if (!sum) throw new Error('empty opponent prior');
    return {M, m, prior: g.prior.map(v => v / sum),
            regret: new Float64Array(m), sum: new Float64Array(m)};
  });
  const security = x => games.reduce((s, {M, m}) => {
    let worst = Infinity;
    for (let j = 0; j < m; j++) {
      let v = 0;
      for (let i = 0; i < n; i++) v += x[i] * M[i][j];
      worst = Math.min(worst, v);
    }
    return s + worst / worlds;
  }, 0);
  const modelRows = actions.map((_, i) => games.reduce((s, g) =>
    s + g.M[i].reduce((v, z, j) => v + z * g.prior[j], 0) / worlds, 0));
  const modelValue = x => x.reduce((s, p, i) => s + p * modelRows[i], 0);
  const regret = new Float64Array(n), sum = new Float64Array(n);
  let avg, lower, upper, used;
  for (let t = 1; t <= iterations; t++) {
    const x = distribution(regret), row = new Float64Array(n);
    for (const g of games) {
      const y = distribution(g.regret), col = new Float64Array(g.m);
      let value = 0;
      for (let i = 0; i < n; i++) {
        let v = 0;
        for (let j = 0; j < g.m; j++) {
          v += g.M[i][j] * y[j];
          col[j] += x[i] * g.M[i][j];
        }
        row[i] += v / worlds;
        value += x[i] * v;
      }
      for (let j = 0; j < g.m; j++) {
        g.regret[j] = Math.max(0, g.regret[j] + value - col[j]); g.sum[j] += t * y[j];
      }
    }
    let value = 0;
    for (let i = 0; i < n; i++) value += x[i] * row[i];
    for (let i = 0; i < n; i++) {
      regret[i] = Math.max(0, regret[i] + row[i] - value); sum[i] += t * x[i];
    }
    if (t % 64 === 0 || t === iterations) {
      const mass = t * (t + 1) / 2;
      avg = Array.from(sum, v => v / mass);
      lower = security(avg);
      upper = Math.max(...actions.map((_, i) => games.reduce((s, g) =>
        s + g.M[i].reduce((v, z, j) => v + z * g.sum[j] / mass, 0) / worlds, 0)));
      used = t;
      if (upper - lower <= tolerance) break;
    }
  }
  // Exploit the supplied opponent prior only within an explicit security budget.
  // On this line segment, security is concave, so its feasible interval contains
  // zero and bisection finds the largest permitted movement toward the pure BR.
  const br = modelRows.indexOf(Math.max(...modelRows));
  const blend = t => avg.map((v, i) => (1 - t) * v + (i === br ? t : 0));
  let lo = 0, hi = 1;
  if (security(blend(1)) >= lower - risk) lo = 1;
  else for (let k = 0; k < 45; k++) {
    const mid = (lo + hi) / 2;
    if (security(blend(mid)) >= lower - risk) lo = mid; else hi = mid;
  }
  const x = blend(lo);
  return {policy: Object.fromEntries(actions.map((a, i) => [a, x[i]])),
          security: security(x), equilibrium_security: lower, upper_bound: upper,
          gap: Math.max(0, upper - lower), iterations: used, exploit_weight: lo,
          model_value: modelValue(x), equilibrium_model_value: modelValue(avg),
          model_rows: Object.fromEntries(actions.map((a, i) => [a, modelRows[i]])),
          worlds, actions: n, risk};
}

module.exports = {solveBelief};
