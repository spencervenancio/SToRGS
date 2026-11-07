import json, sys, urllib.request
pkgs = sys.argv[1:]
for p in pkgs:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{p}/json", timeout=20) as r:
            d = json.load(r)
    except Exception as e:
        print(f"### {p}: NOT FOUND ({e})"); continue
    info = d["info"]; rel = d["releases"]
    def first_upload(v):
        f = rel.get(v) or []
        return min((x["upload_time"] for x in f), default="?")
    vers = [v for v in rel if rel[v]]
    vers_sorted = sorted(vers, key=first_upload)
    latest = info["version"]
    print(f"### {p}  latest={latest}  uploaded={first_upload(latest)}  requires_python={info.get('requires_python')}")
    print(f"    first_release={vers_sorted[0] if vers_sorted else '?'} @ {first_upload(vers_sorted[0]) if vers_sorted else '?'} ; n_releases={len(vers)}")
    print(f"    last5: " + ", ".join(f"{v}({first_upload(v)[:10]})" for v in vers_sorted[-5:]))
    files = rel.get(latest) or []
    print(f"    latest files: " + ", ".join(f"{x['filename']}[{x['size']//1024}KB]" for x in files[:4]))
    print(f"    home: {info.get('home_page') or (info.get('project_urls') or {})}")
    print(f"    requires_dist: {info.get('requires_dist')}")
