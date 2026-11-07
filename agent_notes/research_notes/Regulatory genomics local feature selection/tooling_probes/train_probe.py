# EXPLORATORY feasibility probe (design-deciding: time-to-train + learnability on this laptop). NOT a thesis result.
import sys, time, h5py, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
p = sys.argv[1]; dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(0)
with h5py.File(p, "r") as f: Xtr, ytr, Xva, yva, Xte, yte = (torch.tensor(f[k][:], dtype=torch.float32) for k in ["X_train", "Y_train", "X_valid", "Y_valid", "X_test", "Y_test"])
def blk(i, o, k, pool, drop): return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()] + ([nn.MaxPool1d(pool)] if pool else []) + [nn.Dropout(drop)]
net = nn.Sequential(*blk(4, 24, 19, 0, .1), *blk(24, 32, 7, 4, .2), *blk(32, 48, 7, 4, .3), *blk(48, 64, 3, 3, .4), nn.Flatten(), nn.Linear(64 * 4, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(.5), nn.Linear(96, 1)).to(dev)
opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
def auc(X, y):
    net.eval()
    with torch.no_grad(): s = torch.cat([net(X[i:i+1000].to(dev)).cpu() for i in range(0, len(X), 1000)])
    net.train(); return roc_auc_score(y.numpy().ravel(), s.numpy().ravel())
best, t0 = 0, time.perf_counter()
for ep in range(30):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 100):
        b = perm[i:i+100]; opt.zero_grad(); lf(net(Xtr[b].to(dev)), ytr[b].to(dev)).backward(); opt.step()
    v = auc(Xva, yva)
    if v > best: best, state, bep = v, {k: x.clone() for k, x in net.state_dict().items()}, ep
    if ep - bep >= 5: break
net.load_state_dict(state)
print(f"[{dev}] CNN-deep-like ({sum(q.numel() for q in net.parameters())} params) on synthetic_code_dataset: {ep+1} epochs in {time.perf_counter()-t0:.1f}s; best val AUROC {best:.3f} @ep{bep+1}; test AUROC {auc(Xte, yte):.3f}")
