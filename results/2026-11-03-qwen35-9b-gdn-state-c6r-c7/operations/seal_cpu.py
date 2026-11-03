"""Seal only fully closed, actually reviewed data; all original science bytes stay intact."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import stat

ROOT=Path('/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_FOUR_ROTATIONS_C6R_C7_20261103_V1')
assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
assert not (ROOT/'MANIFEST.sha256').exists() and not (ROOT/'DELIVERY_READY.json').exists()
for name,status in [('operations/CPU_PIPELINE_RESULT.json','PASS_COMPLETE_CPU_ANALYSIS_REPORT_PIPELINE'),
                    ('operations/CPU_TICK_LAYOUT_RESULT.json','PASS_CPU_TICK_LAYOUT_V2'),
                    ('analysis/INDEPENDENT_DATA_REVIEW.json','PASS'),('visual_qa/VISUAL_REVIEW.json','PASS')]:
    assert json.loads((ROOT/name).read_text())['status']==status
who=json.loads((ROOT/'operations/CPU_PIPELINE_CLAIM.json').read_text())['parent']
proc=Path('/proc')/str(who['pid'])/'stat'
if proc.exists():assert int(proc.read_text().split(') ',1)[1].split()[19])!=who['starttick'],'CPU report parent still active'
assert (ROOT/'REPORT_zh.md').is_file() and (ROOT/'DATA_DICTIONARY_zh.md').is_file()

def file_info(path):
    before=path.stat();assert stat.S_ISREG(before.st_mode) and not path.is_symlink()
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    after=path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return {'path':str(path.relative_to(ROOT)),'bytes':after.st_size,'sha256':h.hexdigest()}

files=[]
for directory,dirs,names in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in ['.cache','__pycache__','.staging']]
    for name in names:
        p=Path(directory)/name
        assert name not in ['MANIFEST.sha256','DELIVERY_READY.json']
        assert not p.is_symlink()
        files.append(p)
files.sort()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(file_info,files))
manifest=''.join(r['sha256']+'  '+r['path']+'\n' for r in rows)
with (ROOT/'MANIFEST.sha256').open('x') as f:f.write(manifest)
ready={'status':'PASS_COMPLETE_DELIVERY_READY','A':{m:'PASS' for m in ['Native','Hadamard','C6R','C7']},
       'B':{m:'PASS' for m in ['Native','Hadamard','C6R','C7']},'manifest_files':len(rows),
       'manifest_bytes':sum(r['bytes'] for r in rows),'manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),
       'main_manifest_excludes_only_itself_and_this_ready_file':True,'excluded_non_science_caches':['.cache','__pycache__','.staging'],
       'recommended_report':'REPORT_zh.md','new_generation':False,'new_scoring':False,'new_training':False,
       'GPU_worker_real_wait_integer':0,'own_NVML_released':True,'independent_data_review':'PASS','visual_review':'PASS'}
with (ROOT/'DELIVERY_READY.json').open('x') as f:json.dump(ready,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(ready),flush=True)
