#!/usr/bin/env python3
"""Verify the complete 1,677-file original delivery, including Release data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-directory', type=Path, help='Optional existing complete local delivery')
    args = parser.parse_args()
    meta = json.loads((ROOT / 'RELEASE_ASSETS.json').read_text())
    dest = args.artifact_directory or ROOT / meta['results_directory']
    missing, bad = [], []
    entries = (ROOT / 'PUBLICATION_MANIFEST.sha256').read_text().splitlines()
    for line in entries:
        digest, name = line.split('  ', 1)
        path = dest / name
        if not path.is_file():
            missing.append(name)
        elif sha(path) != digest:
            bad.append(name)
    print(json.dumps({'expected_files': len(entries), 'missing': len(missing), 'mismatched': bad,
                      'status': 'PASS' if not missing and not bad else 'INCOMPLETE_OR_MISMATCH'}, indent=2))
    if missing:
        print('Missing files: download raw data with python3 scripts/fetch_raw_data.py', file=sys.stderr)
    if missing or bad:
        sys.exit(1)
    assert len(entries) == meta['original_total_files']
    assert sha(dest / 'MANIFEST.sha256') == meta['original_main_manifest_sha256']
    assert sha(dest / 'DOCUMENTATION_MANIFEST.sha256') == meta['original_documentation_manifest_sha256']

if __name__ == '__main__':
    main()
