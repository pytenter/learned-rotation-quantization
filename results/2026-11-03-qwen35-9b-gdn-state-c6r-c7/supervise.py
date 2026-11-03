"""One CPU owning parent, one admitted GPU4 worker, real integer wait/release."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
PY='/data/zypan/envs/qwen-gdn-group64-historical-port-v1/bin/python'
OVERLAY='/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/runtime/transformers-qwen35/src:/data/zypan/envs/qwen-gdn-group64-historical-user-site-v1'

def save(name,obj):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def identity(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'PPID':int(s[1]),'starttick':int(s[19]),
            'argv':[x.decode() for x in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}

def epoch_live(who):
    try:return identity(who['pid'])['starttick']==who['starttick']
    except FileNotFoundError:return False

def nvml():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    cfg=json.loads((ROOT/'CONFIG.json').read_text())
    for name in ['CPU_AUDIT.json','CPU_COLLECTOR_BANK_GATE.json']:
        assert json.loads((ROOT/'gates'/name).read_text())['status']=='PASS'
    sources=json.loads((ROOT/'provenance/SOURCE_MANIFEST.json').read_text())
    for item in sources:assert sha(ROOT/item['path'])==item['sha256']
    before=nvml();assert not any(cfg['gpu_uuid'] in row for row in before.splitlines()),'GPU4 is busy; no worker launched'
    assert not (ROOT/'worker.log').exists() and not (ROOT/'formal').exists()
    parent=identity(os.getpid())
    save('operations/RUN_CLAIM.json',{'parent':parent,'physical_gpu_index':cfg['gpu_index'],'GPU_uuid':cfg['gpu_uuid'],
         'SOURCE_manifest_sha256':sha(ROOT/'provenance/SOURCE_MANIFEST.json'),'NVML_before':before})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':cfg['gpu_uuid'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':OVERLAY,
         'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false',
         'TRITON_CACHE_DIR':str(ROOT/'.cache/triton'),'TORCH_EXTENSIONS_DIR':str(ROOT/'.cache/torch_extensions')}
    command=[PY,'-B',str(ROOT/'run_gpu.py')]
    begin=time.monotonic()
    with (ROOT/'worker.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        worker=identity(child.pid)
        save('operations/WORKER_STARTED.json',{'parent':parent,'worker':worker,'CVD':cfg['gpu_uuid'],'command':command})
        print('WORKER_STARTED',json.dumps(worker),flush=True)
        code=child.wait()
    save('operations/WORKER_WAIT.json',{'worker':worker,'parent':parent,'owning_Popen_wait_integer':code,'elapsed_seconds':time.monotonic()-begin})
    deadline=time.monotonic()+120
    while True:
        after=nvml();held=[s for s in after.splitlines() if s.split(',')[0].strip()==str(worker['pid']) and cfg['gpu_uuid'] in s]
        if not held:break
        if time.monotonic()>deadline:raise RuntimeError('Own worker allocation not released after its real exit')
        time.sleep(1)
    assert not epoch_live(worker)
    save('operations/OWNING_CLOSED.json',{'worker':worker,'wait_integer':code,'worker_epoch_gone':True,'own_NVML_released':True,
                                        'foreign_processes_not_signalled':True,'NVML_after':after})
    for item in sources:assert sha(ROOT/item['path'])==item['sha256']
    if code:
        save('operations/PARENT_COMPLETE.json',{'status':'FAIL_GPU_CAPTURE_PRESERVED','worker_wait':code,'own_NVML_released':True})
        raise SystemExit(code)
    assert json.loads((ROOT/'formal/RESULT.json').read_text())['status']=='PASS'
    rows=[]
    for item in json.loads((ROOT/'provenance/MODEL_TOKENIZER_SHA256.json').read_text()):
        p=Path(item['path']);s=p.stat();digest=sha(p)
        assert s.st_size==item['bytes'] and digest==item['sha256']
        rows.append({'path':str(p),'bytes':s.st_size,'sha256':digest,'inode':s.st_ino,'device':s.st_dev})
    save('provenance/POST_RUN_MODEL_SOURCE_VERIFIED.json',{'status':'PASS','source_files':len(sources),'model_files':rows})
    save('operations/PARENT_COMPLETE.json',{'status':'PASS_GPU_CAPTURE_READY_FOR_CPU_ANALYSIS','worker_wait':code,
         'own_NVML_released':True,'worker_epoch_gone':True,'elapsed_seconds':time.monotonic()-begin})
    print('PARENT_COMPLETE_GPU_CAPTURE_PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        if not isinstance(error,SystemExit) and not (ROOT/'operations/SUPERVISOR_FIRST_FAILURE.json').exists():
            save('operations/SUPERVISOR_FIRST_FAILURE.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise
