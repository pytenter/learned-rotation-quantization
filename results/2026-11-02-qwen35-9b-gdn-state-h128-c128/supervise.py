"""One foreground CPU owner, one GPU4 worker, real integer wait and release."""
import os,sys,json,subprocess,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY='/data/zypan/envs/qwen-gdn-group64-historical-port-v1/bin/python'
OVERLAY='/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/runtime/transformers-qwen35/src:/data/zypan/envs/qwen-gdn-group64-historical-user-site-v1'
UUID='GPU-de1f9152-32e5-a033-d2bf-ea7abecb606f'

def save(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def identity(pid):
    p=Path('/proc')/str(pid)
    s=(p/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'ppid':int(s[1]),'starttick':int(s[19]),
            'argv':[a.decode() for a in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}
def apps():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return p.stdout
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    assert json.loads((ROOT/'gates/CPU_GATE_RESULT.json').read_text())['status']=='PASS'
    assert (ROOT/'operations/INPUT_PREPARATION_COMPLETE.json').exists()
    for name in ['provenance/RUNTIME.json','worker.log','formal','gates/GATE_RESULT.json','gates/real_collector']:
        assert not (ROOT/name).exists(), 'Refusing inherited runtime output: '+name
    before=apps();assert not any(UUID in line for line in before.splitlines()),'GPU4 has another compute process; refusing to launch'
    freeze=json.loads((ROOT/'provenance/SOURCE_MANIFEST.json').read_text())
    for f in freeze:
        assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256'],f
    parent=identity(os.getpid())
    save('operations/RUN_CLAIM.json',{'parent':parent,'gpu_uuid':UUID,'physical_gpu_index':4,'source_manifest_sha256':hashlib.sha256((ROOT/'provenance/SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'other_gpu_processes_before':before})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=OVERLAY,
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
        TRITON_CACHE_DIR=str(ROOT/'.cache/triton'),TORCH_EXTENSIONS_DIR=str(ROOT/'.cache/torch_extensions'))
    cmd=[PY,'-B',str(ROOT/'run_experiment.py')]
    with (ROOT/'worker.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        who=identity(child.pid)
        save('operations/WORKER_STARTED.json',{'parent':parent,'worker':who,'command':cmd,'CVD':UUID})
        print('WORKER_STARTED',json.dumps(who),flush=True)
        code=child.wait()
    save('operations/WORKER_WAIT.json',{'worker':who,'owning_parent':parent,'Popen_wait_integer':code,'monotonic':time.monotonic()})
    deadline=time.monotonic()+120
    while True:
        a=apps();held=[x for x in a.splitlines() if x.split(',')[0].strip()==str(who['pid'])]
        if not held:break
        if time.monotonic()>deadline:raise RuntimeError('Own worker NVML allocation remained after exit')
        time.sleep(1)
    for f in freeze:assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256'],f
    save('operations/OWNING_CLOSED.json',{'worker':who,'owning_parent':parent,'worker_wait':code,'own_NVML_released':True,
        'worker_proc_gone':not Path('/proc',str(who['pid'])).exists(),'gpu_processes_after':a})
    print('WORKER_EXIT',code,'OWN_NVML_RELEASED',flush=True)
    if code:sys.exit(code)
    assert json.loads((ROOT/'formal/RESULT.json').read_text())['status']=='PASS'
    save('operations/PARENT_COMPLETE.json',{'status':'PASS','worker_wait':code,'own_NVML_released':True})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not isinstance(e,SystemExit):
            p=ROOT/'operations/SUPERVISOR_FAILURE.json'
            if not p.exists():save('operations/SUPERVISOR_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()})
        raise
