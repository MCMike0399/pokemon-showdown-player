/* Tiny MLP forward pass (weights exported from search/train_value.py). Output in (-1, 1). */
'use strict';
const fs = require('node:fs');
function load(p) {
  const w = JSON.parse(fs.readFileSync(p, 'utf8'));
  return {layers: w.layers.map(l => ({W: l.W.map(r => Float64Array.from(r)), b: Float64Array.from(l.b)})), mean: w.mean, std: w.std};
}
function forward(net, x) {
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
  return Math.tanh(h[0] / 2); // = 2*sigmoid(logit)-1
}
module.exports = {load, forward};
