#!/usr/bin/env python3
"""Verify original scientific file hashes, using Git-only or complete data scope."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_INDEX=ROOT/'results/2026-11-03-qwen35-9b-gdn-state-c6r-c7/RELEASE_ASSETS.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,default=DEFAULT_INDEX)
    p.add_argument('--artifact-directory',type=Path)
    p.add_argument('--git-only',action='store_true',help='Check intentionally committed originals, without downloading raw data')
    a=p.parse_args();meta=json.loads(a.metadata.read_text());index_root=a.metadata.resolve().parent
    dest=a.artifact_directory or ROOT/meta['results_directory']
    entries=[]
    for line in (index_root/'PUBLICATION_MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);rel=PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        entries.append((digest,name))
    assert len(entries)==len({n for _,n in entries})==meta['original_total_files']
    release_files={r['path'] for asset in meta['assets'] for r in asset['files']}
    assert len(release_files)==meta['release_raw_files']
    selected=[(h,n) for h,n in entries if not a.git_only or n not in release_files]
    assert len(selected)==(meta['git_artifact_files'] if a.git_only else meta['original_total_files'])
    missing=[];bad=[];total=0
    for digest,name in selected:
        path=dest/name
        if not path.is_file():missing.append(name)
        elif path.is_symlink() or sha(path)!=digest:bad.append(name)
        else:total+=path.stat().st_size
    result={'status':'PASS' if not missing and not bad else 'INCOMPLETE_OR_MISMATCH',
            'scope':'git_only' if a.git_only else 'complete_original_delivery','expected_files':len(selected),
            'verified_bytes':total,'missing_count':len(missing),'mismatched_paths':bad}
    print(json.dumps(result,indent=2))
    if missing:print('Missing data: run scripts/fetch_release_data.py; first missing path:',missing[0])
    if missing or bad:raise SystemExit(1)
    assert sha(dest/'MANIFEST.sha256')==meta['original_main_manifest_sha256']
    assert sha(dest/'DELIVERY_READY.json')==meta['original_ready_sha256']

if __name__=='__main__':main()
