# EXPLORATORY feasibility probe (design-deciding only; NOT a thesis result). Random-init models, timing only.
import time, torch, torch.nn as nn, captum
from captum.attr import IntegratedGradients, DeepLiftShap
from tangermeme.utils import random_one_hot
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.saliency import saliency
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
from bpnetlite.bpnet import BPNet
print("captum", captum.__version__, "| torch", torch.__version__)
try: torch.zeros(2, dtype=torch.float64, device="mps"); print("float64 tensor on MPS: OK")
except Exception as e: print("float64 tensor on MPS:", type(e).__name__, str(e)[:110])
def sync(d):
    if d == "mps": torch.mps.synchronize()
def T(fn, d):
    sync(d); t = time.perf_counter(); fn(); sync(d); return time.perf_counter() - t
def blk(i, o, k, pool, drop): return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()] + ([nn.MaxPool1d(pool)] if pool else []) + [nn.Dropout(drop)]
def cnn(): return nn.Sequential(*blk(4, 24, 19, 0, .1), *blk(24, 32, 7, 4, .2), *blk(32, 48, 7, 4, .3), *blk(48, 64, 3, 3, .4), nn.Flatten(), nn.Linear(256, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(.5), nn.Linear(96, 1))
X = random_one_hot((256, 4, 200), random_state=0).float()
for dev in ["cpu", "mps"]:
    torch.manual_seed(0); net = cnn().eval().to(dev); x = X.to(dev); refs = X[torch.randperm(256)[:20]].to(dev)
    ig = IntegratedGradients(net); dls = DeepLiftShap(net)
    ig.attribute(x[:4], baselines=torch.zeros_like(x[:4]), target=0, n_steps=50)
    r = {"captum IG (50 steps, zero baseline)": T(lambda: ig.attribute(x, baselines=torch.zeros_like(x), target=0, n_steps=50, internal_batch_size=4096), dev),
         "captum DeepLiftShap (20 shared baselines)": T(lambda: dls.attribute(x, baselines=refs, target=0), dev),
         "tangermeme deep_lift_shap (20 per-seq dinuc shuffles)": T(lambda: deep_lift_shap(net, X, n_shuffles=20, device=dev, random_state=0, batch_size=256), dev),
         "tangermeme saliency (grad-corrected)": T(lambda: saliency(net, X, device=dev, batch_size=256), dev),
         "tangermeme ISM (600 variants/seq)": T(lambda: saturation_mutagenesis(net, X, device=dev, batch_size=1024), dev)}
    for k, v in r.items(): print(f"[{dev}] CNN-deep-like L=200, 256 seqs: {k}: {v:.2f}s  -> {v/256*4000:.0f}s per 4,000 seqs (extrap.)")
for dev, n in (("mps", 1024), ("cpu", 256)):
    torch.manual_seed(0); m = BPNet(n_filters=64, n_layers=8, n_outputs=1, n_control_tracks=0, verbose=False).to(dev)
    Xb = random_one_hot((n, 4, 2114), random_state=1).float(); yc = torch.randn(n, 1); opt = torch.optim.Adam(m.parameters(), 1e-3)
    def ep():
        for i in range(0, n, 64):
            xb = Xb[i:i+64].to(dev); opt.zero_grad(); prof, cnt = m(xb)
            loss = ((cnt - yc[i:i+64].to(dev)) ** 2).mean() - torch.log_softmax(prof.reshape(len(xb), -1), -1).mean(); loss.backward(); opt.step()
    t = T(ep, dev); print(f"[{dev}] BPNet-64x8 training, {n} seqs x 2114bp, bs=64: {t:.1f}s -> ~{t/n*20000/60:.1f} min/epoch at 20k seqs (extrap.)")
