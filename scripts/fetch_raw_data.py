#!/usr/bin/env python3
"""Fetch this release's raw groups and verify each byte; no model/GPU needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def matches(path, spec):
    return path.is_file() and path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remove-archives', action='store_true')
    args = parser.parse_args()
    meta = json.loads((ROOT / 'RELEASE_ASSETS.json').read_text())
    dest = ROOT / meta['results_directory']
    cache = ROOT / '.downloads'
    cache.mkdir(exist_ok=True)
    for a in meta['assets']:
        specs = {x['path']: x for x in a['files']}
        if all(matches(dest / n, s) for n, s in specs.items()):
            print('Already verified:', a['name'], flush=True)
            continue
        archive = cache / a['name']
        if not matches(archive, a):
            if archive.exists():
                raise RuntimeError('Existing archive failed checksum: ' + str(archive))
            partial = archive.with_suffix(archive.suffix + '.part')
            print('Downloading:', a['name'], a['bytes'], 'bytes', flush=True)
            req = urllib.request.Request(a['url'], headers={'User-Agent': 'gdn-state-results-downloader'})
            with urllib.request.urlopen(req, timeout=120) as src, partial.open('wb') as out:
                for b in iter(lambda: src.read(8 * 1024 * 1024), b''):
                    out.write(b)
            if not matches(partial, a):
                raise RuntimeError('Downloaded archive failed SHA256: ' + str(partial))
            os.replace(partial, archive)
        with tarfile.open(archive, 'r:') as tf:
            members = tf.getmembers()
            if len(members) != len(specs) or {m.name for m in members} != set(specs):
                raise RuntimeError('Archive member list mismatch')
            for member in members:
                rel = Path(member.name)
                if not member.isfile() or rel.is_absolute() or '..' in rel.parts:
                    raise RuntimeError('Unsafe archive member')
                expected = specs[member.name]
                if member.size != expected['bytes']:
                    raise RuntimeError('Archive member size mismatch')
                target = dest / rel
                if matches(target, expected):
                    continue
                if target.exists():
                    raise RuntimeError('Refusing to replace different existing data: ' + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + '.part')
                with tf.extractfile(member) as src, partial.open('wb') as out:
                    for b in iter(lambda: src.read(8 * 1024 * 1024), b''):
                        out.write(b)
                if not matches(partial, expected):
                    raise RuntimeError('Member SHA256 mismatch: ' + member.name)
                os.replace(partial, target)
        print('Verified and extracted:', a['name'], flush=True)
        if args.remove_archives:
            archive.unlink()
    print('Raw data complete. Run: python3 scripts/verify_delivery.py')

if __name__ == '__main__':
    main()
