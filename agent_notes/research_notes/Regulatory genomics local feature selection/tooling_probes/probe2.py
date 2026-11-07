# EXPLORATORY feasibility probe (design-deciding only; NOT a thesis result).
import sys, os, json, time, urllib.request, traceback, inspect
import numpy as np, h5py, torch
SP = sys.argv[1]
def get(u, js=True, method="GET"):
    req = urllib.request.Request(u, headers={"User-Agent": "research-probe"}, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        return (json.loads(r.read()) if js else r.read()) if method == "GET" else (r.status, r.headers.get("Content-Length"))
def dl(u, dest):
    if not os.path.exists(dest):
        t = time.perf_counter(); d = get(u, False); open(dest, "wb").write(d); print(f"  downloaded {os.path.basename(dest)} {len(d)/1e6:.1f}MB in {time.perf_counter()-t:.1f}s")
    return dest
def sync(d):
    if d == "mps": torch.mps.synchronize()

print("=== 1. Koo lab synthetic h5 contents ===")
lst = {x["name"]: x["download_url"] for x in get("https://api.github.com/repos/p-koo/exponential_activations/contents/data")}
for fn in ["synthetic_code_dataset.h5", "synthetic_dataset.h5"]:
    try:
        p = dl(lst[fn], f"{SP}/dl_koo/{fn}")
        with h5py.File(p, "r") as f:
            f.visititems(lambda n, o: print(f"  {fn}:{n} shape={o.shape} dtype={o.dtype}") if isinstance(o, h5py.Dataset) else None)
            for k in f.keys():
                if k.lower().startswith("y"): y = f[k][:]; print(f"   {k}: label mean per task = {np.round(y.mean(0), 3)[:12]}")
            for k in [k for k in f.keys() if "model" in k.lower()][:1]:
                M = f[k][:2].astype(float); ax = int(np.argmin(M.shape[1:])) + 1
                ic = np.log2(4) + np.sum(np.where(M > 0, M * np.log2(np.clip(M, 1e-12, 1)), 0), axis=ax)
                print(f"   {k}: per-pos IC seq0 >0.1 at {int((ic[0] > 0.1).sum())} of {ic.shape[1]} positions; colsum range {M.sum(ax).min():.2f}-{M.sum(ax).max():.2f}")
    except Exception: traceback.print_exc(limit=2)
try: print("GradientCorrection/data:", [(x["name"], x["size"]) for x in get("https://api.github.com/repos/p-koo/GradientCorrection/contents/data")])
except Exception as e: print("GradientCorrection/data ERR", e)

print("\n=== 2. JASPAR downloads + planted-motif generator (memelite + numpy) ===")
for yr in ["2024", "2026"]:
    u = f"https://jaspar.elixir.no/download/data/{yr}/CORE/JASPAR{yr}_CORE_vertebrates_non-redundant_pfms_meme.txt"
    try: print(f"  {u} -> HTTP {get(u, method='HEAD')}")
    except Exception as e: print(f"  {u} -> {e}")
try:
    meme = dl(lst["JASPAR_CORE_2016_vertebrates.meme"], f"{SP}/dl_jaspar/JASPAR_CORE_2016_vertebrates.meme")
    from memelite.io import read_meme
    motifs = read_meme(meme); names = list(motifs); m0 = np.asarray(motifs[names[0]])
    print(f"  memelite.read_meme: {len(names)} motifs; '{names[0]}' array shape {m0.shape} type {type(motifs[names[0]]).__name__}")
    rng = np.random.default_rng(0); N, L = 20000, 200; pick = names[:5]
    t = time.perf_counter(); S = rng.integers(0, 4, (N, L)); mask = np.zeros((N, L), bool); y = rng.integers(0, 2, N)
    for i in np.flatnonzero(y):
        for _ in range(rng.integers(1, 4)):
            P = np.asarray(motifs[pick[rng.integers(5)]], float); P = P.T if P.shape[0] == 4 else P; P = P / P.sum(1, keepdims=True)
            w = P.shape[0]; s = rng.integers(0, L - w + 1)
            S[i, s:s+w] = (P.cumsum(1) > rng.random((w, 1))).argmax(1); mask[i, s:s+w] = True
    X = np.eye(4, dtype=np.int8)[S].transpose(0, 2, 1); dt = time.perf_counter() - t
    print(f"  planted-motif generator: {N} x {L}bp, 1-3 PWM samples in positives, with per-position mask: {dt:.2f}s; X {X.shape} mask frac(pos)={mask[y==1].mean():.3f}")
    from tangermeme.ersatz import substitute; from tangermeme.utils import random_one_hot, characters
    Xb = random_one_hot((2, 4, 60), random_state=0); Xs = substitute(Xb, "CACGTG", start=20)
    print("  tangermeme substitute -> ", characters(Xs[0]) if not isinstance(Xs, tuple) else Xs)
except Exception: traceback.print_exc(limit=3)

print("\n=== 3. Real pretrained ENCODE models -> PyTorch via bpnet-lite ===")
from bpnetlite.bpnet import BPNet, CountWrapper
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.utils import random_one_hot
print("  BPNet.from_chrombpnet", inspect.signature(BPNet.from_chrombpnet), "| from_chrombpnet_lite", inspect.signature(BPNet.from_chrombpnet_lite))
try:
    p = dl("https://huggingface.co/kundajelab/encode-bpnet-REST-ChIP-seq-Ishikawa-ENCSR000BUU-ENCSR174BCS/resolve/main/fold_0/model.h5", f"{SP}/dl_hf/bpnet_atlas_REST_fold0.h5")
    with h5py.File(p, "r") as f: print("  BPNet-atlas h5 layers:", list(f["model_weights"].keys())[:30] if "model_weights" in f else list(f.keys()))
    for loader in (BPNet.from_chrombpnet, BPNet.from_chrombpnet_lite):
        try: m = loader(p); print(f"  BPNet atlas via {loader.__name__}: OK params={sum(q.numel() for q in m.parameters())}"); break
        except Exception as e: print(f"  BPNet atlas via {loader.__name__}: FAILED {type(e).__name__}: {str(e)[:150]}")
except Exception: traceback.print_exc(limit=2)
try:
    tree = get("https://huggingface.co/api/models/kundajelab/encode-chrombpnet-ATAC-ENCSR839IAD-ENCSR608VHS/tree/main?recursive=true")
    big = [(x["path"], x.get("size", 0)) for x in tree if x.get("type") == "file" and x["path"].startswith("fold_0/") and not "/logs" in x["path"]]
    for pth, sz in big: print(f"  chrombpnet atlas fold_0 file: {pth} {sz/1e6:.1f}MB")
    nob = [pth for pth, sz in big if pth.endswith(".h5") and "nobias" in pth.lower() and sz < 250e6] or [pth for pth, sz in big if pth.endswith(".h5") and "wo_bias" in pth.lower() and sz < 250e6]
    if nob:
        p = dl(f"https://huggingface.co/kundajelab/encode-chrombpnet-ATAC-ENCSR839IAD-ENCSR608VHS/resolve/main/{nob[0]}", f"{SP}/dl_hf/chrombpnet_nobias_fold0.h5")
        m = BPNet.from_chrombpnet(p); print(f"  ChromBPNet-nobias -> PyTorch BPNet OK: params={sum(q.numel() for q in m.parameters())}, n_filters={getattr(m,'n_filters','?')}, n_layers={getattr(m,'n_layers','?')}")
        X = random_one_hot((4, 4, 2114), random_state=0).float()
        for dev, n in (("mps", 4), ("cpu", 2)):
            cw = CountWrapper(m).to(dev).eval(); deep_lift_shap(cw, X[:1], n_shuffles=2, device=dev, random_state=0)
            sync(dev); t = time.perf_counter(); a = deep_lift_shap(cw, X[:n], n_shuffles=20, device=dev, random_state=0); sync(dev); dt = time.perf_counter() - t
            print(f"  [{dev}] real ChromBPNet-nobias count head: deep_lift_shap {n} seqs x 20 refs = {dt:.2f}s ({dt/n:.2f}s/seq)")
    else: print("  no nobias h5 found under 250MB")
except Exception: traceback.print_exc(limit=3)

print("\n=== 4. Hugging Face: Genentech (gReLU zoo) + multimolecule/bpnet ===")
try:
    ms = get("https://huggingface.co/api/models?author=Genentech&limit=100&sort=lastModified")
    print(f"  Genentech HF models: {len(ms)}:", ", ".join(f"{x['id'].split('/')[1]}" for x in ms))
except Exception as e: print("  Genentech ERR", e)
try:
    r = get("https://huggingface.co/multimolecule/bpnet/raw/main/README.md", js=False).decode()
    keep = [l.strip() for l in r.splitlines() if any(k in l for k in ("OSKN", "Oct4", "Kipoi", "zenodo", "Zenodo", "pip install", "convert", "Convert", "original", "weights", "unofficial", "Disclaimer", "1000", "1 kb", "license"))]
    print("  multimolecule/bpnet README key lines:\n   " + "\n   ".join(keep[:22]))
except Exception as e: print("  multimolecule ERR", e)
try:
    d = get("https://zenodo.org/api/records/4294904"); print("  zenodo 4294904:", d["metadata"]["title"][:80], [(f["key"], f"{f['size']/1e6:.1f}MB") for f in d["files"]])
except Exception as e: print("  zenodo 4294904 ERR", e)
