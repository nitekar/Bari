#!/usr/bin/env python3
"""Lightweight static checks for model artifacts and code references.

Run this without heavy ML runtimes to get quick diagnostics.

Outputs: scripts/static_checks_report.json
"""
import os
import io
import sys
import json
import glob
import pickletools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_bytes(path, n=2048):
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception as e:
        return None

def inspect_tflite(path):
    b = read_bytes(path, 8)
    if not b:
        return {"error": "could not read"}
    try:
        hdr = b.decode('ascii', errors='replace')
    except Exception:
        hdr = str(b[:8])
    return {"header": hdr, "looks_like_tflite": hdr.startswith('TFL3')}

def inspect_hdf5(path):
    b = read_bytes(path, 8)
    if not b:
        return {"error": "could not read"}
    return {"header_bytes": list(b), "is_hdf5": b.startswith(b"\x89HDF\r\n\x1a\n")}

def inspect_pickle(path):
    b = read_bytes(path, 4096)
    if not b:
        return {"error": "could not read"}
    out = io.StringIO()
    try:
        # pickletools.dis can accept a file-like
        pickletools.dis(io.BytesIO(b), out)
        snippet = out.getvalue()
        return {"disasm_snippet": snippet[:1000].replace('\n', '\\n')}
    except Exception as e:
        return {"error": f"disasm failed: {e}"}

def find_feature_defs():
    patterns = ["FEAT_WITH_HB", "FEAT_NO_HB", "build_wh_vector", "build_nh_vector"]
    hits = {}
    for p in ROOT.rglob('*.py'):
        try:
            txt = p.read_text(encoding='utf-8')
        except Exception:
            continue
        for pat in patterns:
            if pat in txt:
                hits.setdefault(pat, []).append(str(p.relative_to(ROOT)))
    return hits

def find_scalers():
    files = []
    for ext in ('*.pkl','*.joblib','*.sav'):
        files.extend([str(p.relative_to(ROOT)) for p in ROOT.rglob(ext)])
    return files

def main():
    report = {"root": str(ROOT), "models": [], "code_hits": {}, "scalers": []}

    model_dir = ROOT / 'models' / 'saved_models'
    if not model_dir.exists():
        report['models_error'] = 'models/saved_models not found'
    else:
        for p in sorted(model_dir.iterdir()):
            info = {"path": str(p.relative_to(ROOT)), "size": p.stat().st_size}
            if p.suffix.lower() == '.tflite':
                info.update(inspect_tflite(p))
            elif p.suffix.lower() in ('.pkl', '.joblib', '.sav'):
                info.update(inspect_pickle(p))
            elif p.suffix.lower() in ('.h5', '.keras'):
                info.update(inspect_hdf5(p))
            else:
                info.update({"note": "unknown extension - raw size reported"})
            report['models'].append(info)

    report['code_hits'] = find_feature_defs()
    report['scalers'] = find_scalers()

    out_path = ROOT / 'scripts' / 'static_checks_report.json'
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report to: {out_path}")

if __name__ == '__main__':
    main()
