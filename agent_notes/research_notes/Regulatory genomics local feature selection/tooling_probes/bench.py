# NOTE: the final captum block in this file has a bug (Xt is reused from the L=1000 loop, so shapes mismatch); recheck.py supersedes it.
# EXPLORATORY feasibility probe (design-deciding only; NOT a thesis result). Random-init models, timing only.
import time, sys, inspect, traceback, platform
import numpy as np, torch, torch.nn as nn
print("python", sys.version.split()[0], "| torch", torch.__version__, "| threads", torch.get_num_threads(), "|", platform.machine(), "| mps", torch.backends.mps.is_available())
torch.manual_seed(0); np.random.seed(0)
def sync(d):
    if d == "mps": torch.mps.synchronize()
def timeit(fn, d):
    sync(d); t = time.perf_counter(); out = fn(); sync(d); return time.perf_counter() - t, out
def kw(f, **k):
    p = inspect.signature(f).parameters; return {a: b for a, b in k.items() if a in p}

print("\n=== A. simdna on py3.10 ===")
try:
    import simdna
    from simdna import synthetic as sn
    lm = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH, pseudocountProb=0.001)
    names = sorted(lm.loadedMotifs.keys())
    print("ENCODE motifs:", len(names), [n for n in names if n.startswith(("CTCF", "GATA1", "SPI1", "TAL1"))][:6])
    m1 = "CTCF_known1" if "CTCF_known1" in names else names[0]
    m2 = next((n for n in names if n.startswith("GATA")), names[1])
    emb = [sn.SubstringEmbedder(sn.PwmSamplerFromLoadedMotifs(lm, m)) for m in (m1, m2)]
    gen = sn.EmbedInABackground(sn.ZeroOrderBackgroundGenerator(seqLength=200), emb)
    t = time.perf_counter(); seqs = list(sn.GenerateSequenceNTimes(gen, 1000).generateSequences()); dt = time.perf_counter() - t
    s = seqs[0]
    print(f"generated 1000 x 200bp, 2 PWM-sampled motifs each, in {dt:.2f}s")
    print("seq[:50]", s.seq[:50]); print("embeddings (what, startPos):", [(str(e.what)[:45], e.startPos) for e in s.embeddings])
except Exception: traceback.print_exc()
try:
    from simdna import dinuc_shuffle as ds
    fns = [n for n in dir(ds) if "shuffle" in n.lower() and callable(getattr(ds, n))]
    print("simdna.dinuc_shuffle fns:", fns)
    print("call:", getattr(ds, fns[0])("ACGTACGTTTGACCAGGT"))
except Exception as e: print("simdna dinuc shuffle FAILED:", type(e).__name__, e)

print("\n=== B. tangermeme / bpnet-lite API ===")
import tangermeme; print("tangermeme", getattr(tangermeme, "__version__", "?"))
from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute, insert, dinucleotide_shuffle
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.saliency import saliency
try: from tangermeme.saturation_mutagenesis import saturation_mutagenesis
except ImportError: from tangermeme.ism import saturation_mutagenesis
for f in (deep_lift_shap, saliency, saturation_mutagenesis, substitute, insert, dinucleotide_shuffle, random_one_hot):
    sig = inspect.signature(f)
    print(f"{f.__module__}.{f.__name__}{tuple(sig.parameters)}\n    defaults:", {k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty})
try:
    import bpnetlite; from bpnetlite.bpnet import BPNet, CountWrapper
    print("bpnetlite", getattr(bpnetlite, "__version__", "?"), "imports OK without macs3; BPNet.__init__", inspect.signature(BPNet.__init__))
    print("BPNet loaders:", [n for n in dir(BPNet) if n.startswith("from_")])
    from bpnetlite.chrombpnet import ChromBPNet; print("ChromBPNet loaders:", [n for n in dir(ChromBPNet) if n.startswith("from_")])
except Exception: traceback.print_exc()

print("\n=== C. timings: BPNet-64x8 count head, L=2114 ===")
bp = BPNet(n_filters=64, n_layers=8, n_outputs=1, n_control_tracks=0, verbose=False)
print("BPNet(64 filters, 8 layers) params:", sum(p.numel() for p in bp.parameters()))
X = random_one_hot((16, 4, 2114), random_state=0).float()
for dev in ["cpu", "mps"]:
    try:
        m = CountWrapper(bp).to(dev).eval()
        f = lambda n, ns: deep_lift_shap(m, X[:n], **kw(deep_lift_shap, n_shuffles=ns, device=dev, random_state=0, verbose=False, batch_size=64))
        timeit(lambda: f(2, 20), dev)  # warmup
        t, a = timeit(lambda: f(16, 20), dev); print(f"[{dev}] deep_lift_shap 16 seqs x 20 dinuc-shuffle refs: {t:.2f}s  ({t/16:.3f}s/seq) attr shape {tuple(a.shape)}")
        t, _ = timeit(lambda: saliency(m, X, **kw(saliency, device=dev, batch_size=64)), dev); print(f"[{dev}] saliency (grad-corrected default) 16 seqs: {t:.3f}s")
        t, _ = timeit(lambda: saturation_mutagenesis(m, X[:2], **kw(saturation_mutagenesis, device=dev, batch_size=256, verbose=False)), dev); print(f"[{dev}] ISM (fast default) 2 seqs x 2114bp (~{3*2114} variants/seq): {t:.2f}s ({t/2:.2f}s/seq)")
    except Exception: print(f"[{dev}] FAILED"); traceback.print_exc(limit=3)
big = BPNet(n_filters=512, n_layers=8, n_outputs=1, n_control_tracks=0, verbose=False)
print("ChromBPNet-sized (512x8) params:", sum(p.numel() for p in big.parameters()))
for dev in ["cpu", "mps"]:
    try:
        m = CountWrapper(big).to(dev).eval(); xb = X[:8].to(dev)
        with torch.no_grad():
            m(xb); t, _ = timeit(lambda: m(xb), dev)
        print(f"[{dev}] 512x8 forward, 8 seqs x 2114bp: {t:.3f}s ({t/8*1000:.1f} ms/seq)")
    except Exception: print(f"[{dev}] FAILED"); traceback.print_exc(limit=3)

print("\n=== D. small CNN (Koo/DeepBind-like) training, 20k seqs ===")
class SmallCNN(nn.Module):
    def __init__(s, L):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(4, 32, 19, padding=9), nn.ReLU(), nn.MaxPool1d(4), nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                              nn.Flatten(), nn.Linear(64 * (L // 16), 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(s, x): return s.net(x)
for L in (200, 1000):
    Xt = random_one_hot((20000, 4, L), random_state=1).float(); y = torch.randint(0, 2, (20000, 1)).float()
    for dev in ["cpu", "mps"]:
        try:
            torch.manual_seed(0); net = SmallCNN(L).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
            def epoch():
                for i in range(0, 20000, 128):
                    xb, yb = Xt[i:i+128].to(dev), y[i:i+128].to(dev); opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
            t, _ = timeit(epoch, dev)
            print(f"[{dev}] L={L} SmallCNN ({sum(p.numel() for p in net.parameters())} params) 1 epoch / 20k seqs, bs=128: {t:.2f}s")
        except Exception: print(f"[{dev}] FAILED"); traceback.print_exc(limit=3)
    if L == 200:
        net = SmallCNN(200).eval(); Xs = Xt[:64]
        for dev in ["cpu", "mps"]:
            try:
                n2 = net.to(dev)
                t, _ = timeit(lambda: deep_lift_shap(n2, Xs, **kw(deep_lift_shap, n_shuffles=20, device=dev, random_state=0, verbose=False)), dev); print(f"[{dev}] SmallCNN L=200 deep_lift_shap 64 seqs x 20 refs: {t:.2f}s")
                t, _ = timeit(lambda: saturation_mutagenesis(n2, Xs, **kw(saturation_mutagenesis, device=dev, verbose=False)), dev); print(f"[{dev}] SmallCNN L=200 ISM 64 seqs: {t:.2f}s")
            except Exception: print(f"[{dev}] FAILED"); traceback.print_exc(limit=3)
try:
    from captum.attr import DeepLiftShap, IntegratedGradients
    import captum; n2 = SmallCNN(200).eval().to("mps"); xb = Xt[:8].to("mps"); bl = torch.zeros(8, 4, 200, device="mps")
    a = IntegratedGradients(n2).attribute(xb, baselines=bl, n_steps=50); print("captum", captum.__version__, "IntegratedGradients on MPS OK", tuple(a.shape))
except Exception: print("captum on MPS FAILED"); traceback.print_exc(limit=3)
