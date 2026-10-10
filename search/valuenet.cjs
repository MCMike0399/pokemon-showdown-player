/* Tiny MLP forward pass (weights exported from search/train_value.py). */
'use strict';
const fs = require('node:fs');
function load(p) {
  const w = JSON.parse(fs.readFileSync(p, 'utf8'));
  return {layers: w.layers.map(l => ({W: l.W.map(r => Float64Array.from(r)), b: Float64Array.from(l.b)})),
          mean: w.mean, std: w.std, hand: w.hand || null};
}
/** Raw output: the logit of the win probability. */
function logit(net, x) {
  let h = x.map((v, i) => (v - net.mean[i]) / net.std[i]);
  net.layers.forEach((l, k) => {
    const o = new Array(l.b.length);
    for (let j = 0; j < l.b.length; j++) {
      let s = l.b[j]; const row = l.W[j];
      for (let i = 0; i < h.length; i++) s += row[i] * h[i];
      o[j] = k < net.layers.length - 1 ? Math.max(0, s) : s;
    }
    h = o;
  });
  return h[0];
}
/** Output in (-1, 1) = 2*sigmoid(logit)-1 (legacy scale, multiplied by value_scale). */
function forward(net, x) {
  return Math.tanh(logit(net, x) / 2);
}
/** Output in hand-evaluation units when the net carries the hand calibration
 * p = sigmoid(a*h + b): u = (logit/temp - b) / a, so sigmoid(a*u + b) is the net's
 * live-calibrated probability and blending with the hand evaluation is unit-consistent.
 * Clipped below the +-30 terminal score so a real win always beats an estimate. */
function handUnits(net, x) {
  const u = (logit(net, x) / (net.hand.temp || 1) - net.hand.b) / net.hand.a;
  return Math.max(-20, Math.min(20, u));
}
module.exports = {load, logit, forward, handUnits};
