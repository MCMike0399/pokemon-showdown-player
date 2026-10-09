"""Train the value MLP on self-play samples (MPS if available); export JSON weights."""
import json, sys, random
import numpy as np, torch, torch.nn as nn

def main(out, *paths, epochs=8, hidden=128):
    X, Y = [], []
    for p in paths:
        for line in open(p):
            d = json.loads(line); X.append(d['x']); Y.append(d['y'])
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    n = len(X); idx = np.random.RandomState(0).permutation(n); cut = int(n * 0.9)
    mean = X[idx[:cut]].mean(0); std = X[idx[:cut]].std(0) + 1e-3
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    Xt = torch.tensor((X - mean) / std, device=dev); Yt = torch.tensor(Y, device=dev)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    tr, va = torch.tensor(idx[:cut], device=dev), torch.tensor(idx[cut:], device=dev)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            loss = lossf(net(Xt[b]).squeeze(-1), Yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lv = net(Xt[va]).squeeze(-1); vl = lossf(lv, Yt[va]).item()
            acc = ((lv > 0).float() == (Yt[va] > .5).float()).float().mean().item()
        print(f'epoch {ep} val_loss {vl:.4f} val_acc {acc:.3f} n={n}', flush=True)
    layers = [m for m in net if isinstance(m, nn.Linear)]
    json.dump({'layers': [{'W': l.weight.detach().cpu().tolist(), 'b': l.bias.detach().cpu().tolist()} for l in layers],
               'mean': mean.tolist(), 'std': std.tolist(), 'val_loss': vl, 'val_acc': acc, 'n': n}, open(out, 'w'))

if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
