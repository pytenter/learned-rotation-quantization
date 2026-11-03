#!/usr/bin/env python3
"""Download explicitly requested Release data; verify archives and original members."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_INDEX=ROOT/'results/2026-11-03-qwen35-9b-gdn-state-c6r-c7/RELEASE_ASSETS.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def matches(path,spec):
    return path.is_file() and not path.is_symlink() and path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256']

def safe_target(dest,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise RuntimeError('Unsafe metadata path')
    target=dest/Path(*rel.parts)
    if not target.resolve().is_relative_to(dest.resolve()):raise RuntimeError('Destination symlink escapes data directory')
    return target

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,default=DEFAULT_INDEX)
    p.add_argument('--destination',type=Path)
    p.add_argument('--cache-directory',type=Path)
    p.add_argument('--asset',action='append',help='Exact asset name; repeat to select a subset')
    p.add_argument('--remove-archives',action='store_true')
    args=p.parse_args();meta=json.loads(args.metadata.read_text())
    dest=args.destination or ROOT/meta['results_directory']
    cache=args.cache_directory or ROOT/'.downloads'/meta['release_tag']
    dest.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    selected=set(args.asset or [a['name'] for a in meta['assets']])
    if selected-{a['name'] for a in meta['assets']}:raise RuntimeError('Unknown selected asset name')
    for a in meta['assets']:
        if a['name'] not in selected:continue
        specs={x['path']:x for x in a['files']}
        if len(specs)!=len(a['files']):raise RuntimeError('Duplicate metadata member')
        if all(matches(safe_target(dest,n),s) for n,s in specs.items()):
            print('Already verified:',a['name'],flush=True);continue
        archive=cache/a['name']
        if Path(a['name']).name!=a['name']:raise RuntimeError('Unsafe asset name')
        if not matches(archive,a):
            if archive.exists():raise RuntimeError('Different existing archive; retained: '+str(archive))
            partial=archive.with_suffix(archive.suffix+'.part')
            if partial.exists():raise RuntimeError('Previous partial retained; inspect before retry: '+str(partial))
            print('Downloading:',a['name'],a['bytes'],'bytes',flush=True)
            req=urllib.request.Request(a['url'],headers={'User-Agent':'gdn-state-results-downloader'})
            with urllib.request.urlopen(req,timeout=120) as src,partial.open('xb') as out:
                for block in iter(lambda:src.read(8<<20),b''):out.write(block)
            if not matches(partial,a):raise RuntimeError('Downloaded archive failed SHA256: '+str(partial))
            os.replace(partial,archive)
        with tarfile.open(archive,'r:') as tf:
            members=tf.getmembers()
            if len(members)!=len(specs) or {m.name for m in members}!=set(specs):raise RuntimeError('Archive member list mismatch')
            for member in members:
                if not member.isfile():raise RuntimeError('Non-regular archive member')
                expected=specs[member.name]
                if member.size!=expected['bytes']:raise RuntimeError('Archive member size mismatch')
                target=safe_target(dest,member.name)
                if matches(target,expected):continue
                if target.exists() or target.is_symlink():raise RuntimeError('Different existing data retained: '+str(target))
                target.parent.mkdir(parents=True,exist_ok=True)
                partial=target.with_suffix(target.suffix+'.part')
                with tf.extractfile(member) as src,partial.open('xb') as out:
                    for block in iter(lambda:src.read(8<<20),b''):out.write(block)
                if not matches(partial,expected):raise RuntimeError('Member SHA256 mismatch: '+member.name)
                os.replace(partial,target)
        print('Verified and extracted:',a['name'],flush=True)
        if args.remove_archives:archive.unlink()
    print('Selected data complete. Full verification: python3 scripts/verify_release_data.py')

if __name__=='__main__':main()
