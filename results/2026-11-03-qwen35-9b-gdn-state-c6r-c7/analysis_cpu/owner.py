"""Single CPU report pipeline, admitted only after real GPU exit/release; no GPU retries."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PY='/data/zypan/envs/qwen-gdn-group64-historical-port-v1/bin/python'
OVERLAY='/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/runtime/transformers-qwen35/src:/data/zypan/envs/qwen-gdn-group64-historical-user-site-v1'

def save(name,obj):
    with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')

def identity(pid):
    proc=Path('/proc')/str(pid);s=(proc/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'starttick':int(s[19]),'PPID':int(s[1]),
            'argv':[v.decode() for v in (proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    assert json.loads((ROOT/'operations/PARENT_COMPLETE.json').read_text())['status']=='PASS_GPU_CAPTURE_READY_FOR_CPU_ANALYSIS'
    assert json.loads((ROOT/'operations/OWNING_CLOSED.json').read_text())['own_NVML_released']
    assert json.loads((ROOT/'gates/CPU_ANALYSIS_QUALIFICATION.json').read_text())['status']=='PASS'
    parent=identity(os.getpid())
    sources=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(Path(__file__).parent.glob('*.py'))]
    save('operations/CPU_PIPELINE_CLAIM.json',{'parent':parent,'sources':sources,'GPU_work_authorized_here':False})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':OVERLAY,
         'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MPLCONFIGDIR':str(ROOT/'.cache/matplotlib')}
    results=[];begin=time.monotonic()
    for script,marker in [('aggregate.py','analysis/ANALYSIS_COMPLETE.json'),
                          ('independent_review.py','analysis/INDEPENDENT_DATA_REVIEW.json'),
                          ('bitmap_qa.py','visual_qa/BITMAP_CHECKS.json'),('report.py','analysis/REPORT_CREATED.json')]:
        assert not (ROOT/marker).exists(),'Step already completed; inspect claims instead of duplicating it'
        command=[PY,'-B',str(ROOT/'analysis_cpu'/script)]
        with (ROOT/('cpu_'+Path(script).stem+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            who=identity(child.pid)
            print('CPU_STEP_STARTED',script,json.dumps(who),flush=True)
            code=child.wait()
        row={'script':script,'child':who,'owner':parent,'real_Popen_wait_integer':code}
        save('operations/CPU_WAIT_'+Path(script).stem+'.json',row);results.append(row)
        if code:
            save('operations/CPU_PIPELINE_RESULT.json',{'status':'FAIL_CPU_ONLY_PRESERVED','failed_step':script,'steps':results})
            raise SystemExit(code)
        assert json.loads((ROOT/marker).read_text())['status']=='PASS'
    for row in sources:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
    save('operations/CPU_PIPELINE_RESULT.json',{'status':'PASS_COMPLETE_CPU_ANALYSIS_REPORT_PIPELINE','steps':results,
        'elapsed_seconds':time.monotonic()-begin,'GPU_restarted':False,'manual_visual_review_and_delivery_seal_pending':True})
    print('CPU_PIPELINE_PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not isinstance(e,SystemExit) and not (ROOT/'operations/CPU_PIPELINE_FIRST_FAILURE.json').exists():
            save('operations/CPU_PIPELINE_FIRST_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()})
        raise
