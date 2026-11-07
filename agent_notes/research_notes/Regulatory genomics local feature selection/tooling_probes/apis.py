import json, urllib.request, sys
def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": "research-probe", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
def human(n): 
    for unit in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
for rid in ["7011631", "3371164", "5502060", "6639247", "5874507"]:
    try:
        d = get(f"https://zenodo.org/api/records/{rid}")
        fs = d.get("files", [])
        print(f"### zenodo {rid}: '{d['metadata']['title'][:90]}' pub {d['metadata'].get('publication_date')} | {len(fs)} files, total {human(sum(f['size'] for f in fs))} | latest-version? {d.get('links',{}).get('latest','')[-30:]}")
        for f in sorted(fs, key=lambda f: -f["size"])[:8]: print(f"    {f['key'][:100]}  {human(f['size'])}")
    except Exception as e: print(f"### zenodo {rid}: ERR {e}")
for repo, path in [("p-koo/GradientCorrection", ""), ("p-koo/exponential_activations", "data"), ("p-koo/exponential_activations", "code/generate_data")]:
    try:
        d = get(f"https://api.github.com/repos/{repo}/contents/{path}")
        print(f"### github {repo}/{path}: " + ", ".join(f"{x['name']}({human(x['size'])})" if x['type']=='file' else x['name']+'/' for x in d))
    except Exception as e: print(f"### github {repo}/{path}: ERR {e}")
try:
    r = urllib.request.urlopen("https://raw.githubusercontent.com/p-koo/GradientCorrection/main/README.md", timeout=30).read().decode()
except Exception:
    r = urllib.request.urlopen("https://raw.githubusercontent.com/p-koo/GradientCorrection/master/README.md", timeout=30).read().decode()
print("### GradientCorrection README:\n", r[:1800])
for m in ["multimolecule/bpnet", "kundajelab/encode-chrombpnet-ATAC-ENCSR839IAD-ENCSR608VHS", "kundajelab/encode-bpnet-REST-ChIP-seq-Ishikawa-ENCSR000BUU-ENCSR174BCS"]:
    try:
        t = get(f"https://huggingface.co/api/models/{m}/tree/main?recursive=true")
        files = [x for x in t if x.get("type") == "file"]
        print(f"### HF {m}: {len(files)} files, total {human(sum(x.get('size',0) for x in files))}")
        for x in files[:14]: print(f"    {x['path']}  {human(x.get('size',0))}")
    except Exception as e: print(f"### HF {m}: ERR {e}")
try:
    y = urllib.request.urlopen("https://raw.githubusercontent.com/kipoi/models/master/BPNet-OSKN/model.yaml", timeout=30).read().decode()
    print("### kipoi BPNet-OSKN model.yaml:\n", y[:900])
except Exception as e: print("### kipoi BPNet-OSKN: ERR", e)
d = get("https://pypi.org/pypi/tensorflow/2.8.0/json"); print("### tensorflow 2.8.0 macOS-arm64 wheels:", [f['filename'] for f in d['urls'] if 'arm64' in f['filename']] or "NONE")
