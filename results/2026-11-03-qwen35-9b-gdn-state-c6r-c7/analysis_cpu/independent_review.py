"""Independent coverage/count/identity review from actual files and real wait receipts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv
import json
import math
import subprocess
import sys
import time
import numpy as np
from common import ROOT,LAYERS,METHODS,SAMPLES,NAMES,data_path,phase_root,histograms,sha,jd

def epoch_exists(who):
    path=Path('/proc')/str(who['pid'])/'stat'
    if not path.exists():return False
    return int(path.read_text().split(') ',1)[1].split()[19])==who['starttick']

def main():
    begin=time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    jd(ROOT/'operations/INDEPENDENT_REVIEW_CLAIM.json',{'pid':os.getpid(),'source_sha256':sha(__file__),
        'scope':'independent actual file coverage, integer counts, physical bank lifecycle, checkpoint and source identity'})
    assert json.loads((ROOT/'analysis/ANALYSIS_COMPLETE.json').read_text())['status']=='PASS'
    wait=json.loads((ROOT/'operations/WORKER_WAIT.json').read_text())
    closed=json.loads((ROOT/'operations/OWNING_CLOSED.json').read_text())
    parent=json.loads((ROOT/'operations/PARENT_COMPLETE.json').read_text())
    run=json.loads((ROOT/'operations/RUN_CLAIM.json').read_text())
    started=json.loads((ROOT/'operations/WORKER_STARTED.json').read_text())
    runtime=json.loads((ROOT/'provenance/ACTUAL_RUNTIME.json').read_text())
    cfg=json.loads((ROOT/'CONFIG.json').read_text())
    assert type(wait['owning_Popen_wait_integer']) is int and wait['owning_Popen_wait_integer']==0
    assert closed['worker_epoch_gone'] and closed['own_NVML_released']
    assert parent['status']=='PASS_GPU_CAPTURE_READY_FOR_CPU_ANALYSIS'
    assert not epoch_exists(wait['worker']) and not epoch_exists(run['parent'])
    assert runtime['pid']==started['worker']['pid'] and runtime['starttick']==started['worker']['starttick']
    assert runtime['argv']==started['worker']['argv'] and runtime['PPID']==started['worker']['PPID']
    assert runtime['CVD']==cfg['gpu_uuid'] and cfg['gpu_index']==4
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    numeric_pid_rows=[line for line in apps.splitlines() if line.split(',')[0].strip()==str(runtime['pid'])]
    # A new epoch reusing the same numeric PID is foreign, never the old worker.
    worker_pid_reused=(Path('/proc')/str(runtime['pid'])).exists()
    if not worker_pid_reused:assert not numeric_pid_rows
    assert sha(ROOT/'provenance/SOURCE_MANIFEST.json')==run['SOURCE_manifest_sha256']
    sources=json.loads((ROOT/'provenance/SOURCE_MANIFEST.json').read_text())
    for item in sources:assert sha(ROOT/item['path'])==item['sha256']
    pre=json.loads((ROOT/'provenance/MODEL_TOKENIZER_SHA256.json').read_text())
    post=json.loads((ROOT/'provenance/POST_RUN_MODEL_SOURCE_VERIFIED.json').read_text())
    assert post['status']=='PASS' and len(pre)==len(post['model_files'])==17
    for a,b in zip(pre,post['model_files']):
        assert a['path']==b['path'] and a['sha256']==b['sha256'] and a['bytes']==b['bytes']
        stat=Path(a['path']).stat();assert stat.st_size==a['bytes'] and stat.st_ino==a['inode'] and stat.st_dev==a['device']
    formal=json.loads((ROOT/'formal/RESULT.json').read_text());assert formal['status']=='PASS' and len(formal['results'])==8
    total_steps={m:0 for m in METHODS};total_writes={m:0 for m in METHODS};transform_checks=0
    for record in formal['results']:
        plan=record['plan'];sample=record['sample_id'];n=record['input_tokens']
        inp=json.loads((ROOT/'inputs'/f'{sample}.json').read_text());assert n==len(inp['prompt_ids'])+2048
        bank=record['physical_INT8_cache'];assert bank['persistent_B_dense_State'] is False
        assert bank['FP_reference_present']==plan['fp_reference']
        assert bank['FP_reference_bytes']==(50331648 if plan['fp_reference'] else 0)
        assert bank['B_codes_bytes']==2*12582912 and bank['B_scales_bytes']==2*393216
        assert bank['core_read_pointer_checks']==24*(n-1)
        assert set(int(k) for k in bank['per_layer'])==set(LAYERS)
        for layer in LAYERS:
            field=bank['per_layer'][str(layer)]
            assert field['codes']['shape']==[2,32,128,128] and field['codes']['dtype']=='torch.int8'
            assert field['scales']['shape']==[2,32,1,128] and field['scales']['dtype']=='torch.float32'
            assert field['codes']['bytes']==1048576 and field['scales']['bytes']==32768
            assert field['writes']==n and field['reads']==n-1
            assert record['core_calls_by_layer'][str(layer)]==n
        for method in plan['B_methods']:total_steps[method]+=n;total_writes[method]+=n*24
        transform_checks+=len(record['original_transform_runtime_checks'])
    assert all(n==8602 for n in total_steps.values()) and all(n==206448 for n in total_writes.values())
    assert transform_checks==192
    expected_groups={data_path(e,m,l,s) for e in ['A','B'] for m in METHODS for l in LAYERS for s in SAMPLES}
    expected_keys={data_path(e,m,l,s,'key_summaries') for e in ['A','B'] for m in METHODS for l in LAYERS for s in SAMPLES}
    assert set((ROOT/'formal').glob('*/[AB]/groups/*.npz'))==expected_groups
    assert set((ROOT/'formal').glob('*/[AB]/key_summaries/*.npz'))==expected_keys
    assert not list((ROOT/'formal').glob('*/.staging/**/*.npy'))
    summaries=list(csv.DictReader((ROOT/'analysis/per_layer_summary.csv').open()))
    index={(r['experiment'],r['method'],int(r['model_layer_id'])):r for r in summaries};assert len(index)==192
    reviewed=0
    for exp in ['A','B']:
        hist=histograms(exp)
        for layer in LAYERS:
            print(json.dumps({'phase':'INDEPENDENT_DATA_LAYER','experiment':exp,'layer':layer}),flush=True)
            for method in METHODS:
                totals=np.zeros(6,dtype=np.float64);n_groups=0
                for sample in SAMPLES:
                    with np.load(data_path(exp,method,layer,sample)) as g:
                        m=g['metrics'];ix=g['key_absmax_index'];positions=g['token_idx']
                        assert tuple(g['metric_names'])==NAMES and int(g['model_layer_id'])==layer
                        assert str(g['sample_id'])==sample and str(g['condition'])==f'{exp}_{method}'
                        assert m.dtype==np.float64 and m.shape==(129,32,128,16)
                        assert ix.dtype==np.uint8 and ix.shape==(129,32,128) and ix.max()<128
                        assert np.array_equal(positions,np.r_[0,np.arange(15,2048,16)])
                        assert not np.isinf(m).any() and np.isnan(m).sum()==np.isnan(m[...,5]).sum()
                        assert np.array_equal(np.isnan(m[...,5]),m[...,14].astype(bool))
                        assert (m[...,6]>0).all() and (m[...,7]>=0).all() and (m[...,15]==0).all()
                        assert np.isin(m[...,13:15],[0.,1.]).all()
                        zero=m[...,13].astype(bool)
                        assert (m[...,2][zero]==0).all() and (m[...,7][zero]==0).all()
                        assert (m[...,9:11]>=0).all() and (m[...,9:11]<=128).all()
                        with np.load(data_path(exp,method,layer,sample,'key_summaries')) as k:
                            km=k['key_absmax_over_value'];kc=k['key_out3_value_count']
                            assert km.dtype==np.float32 and kc.dtype==np.uint16
                            assert km.shape==kc.shape==(129,32,128) and np.isfinite(km).all() and kc.max()<=128
                            assert np.array_equal(kc.astype(np.int64).sum(axis=2),m[...,9].sum(axis=2).astype(np.int64))
                            assert np.array_equal(km.max(axis=2),m[...,2].max(axis=2))
                            if exp=='B':
                                assert k['cache_key_absmax'].shape==km.shape and np.isfinite(k['cache_key_absmax']).all()
                                assert k['cache_key_out3'].dtype==np.uint16 and k['cache_key_out3'].max()<=128
                        totals += np.array([m[...,9].sum(),m[...,10].sum(),m[...,11].sum(),m[...,12].sum(),m[...,13].sum(),m[...,14].sum()])
                        n_groups+=m.shape[0]*m.shape[1]*m.shape[2];reviewed+=1
                row=index[exp,method,layer];assert n_groups==int(row['group_observations'])==2113536
                for i,name in [(0,'out3_count'),(1,'out6_count'),(4,'zero_groups'),(5,'zero_variance_groups')]:assert int(totals[i])==int(row[name])
                pooled=math.sqrt(totals[2])/(math.sqrt(totals[3])+1e-12)
                assert abs(pooled-float(row['pooled_qdq_relative_L2']))<1e-12
                assert int(hist[f'{exp}_{method}__L{layer:02d}__peak_rms'].sum())==n_groups
                assert int(hist[f'{exp}_{method}__L{layer:02d}__absolute'].sum())==n_groups*128
    import torch
    sys.path.insert(0,str(ROOT))
    from rotation_adapter import Rotations,tensor_sha
    torch.set_num_threads(8);rotations=Rotations(ROOT,'cpu')
    matrix_rows=list(csv.DictReader((ROOT/'provenance/MATRIX_IDENTITY.csv').open()))
    assert len(matrix_rows)==96
    for row in matrix_rows:
        method=row['method'];layer=int(row['model_layer_id'])
        assert tensor_sha(rotations.matrix(method,layer))==row['effective_reference_sha256']
        if method in ('C6R','C7'):assert tensor_sha(rotations.delta(method,layer))==row['delta_sha256']
    assert not torch.cuda.is_initialized()
    gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True)
    jd(ROOT/'analysis/INDEPENDENT_DATA_REVIEW.json',{'status':'PASS','group_files_reviewed':reviewed,'key_files_reviewed':len(expected_keys),
        'all_group_fields_dimensions_finiteness_and_axis_checks':'PASS','integer_tail_counts_and_pooled_L2_independent_reduction':'PASS',
        'all_96_effective_matrix_hashes_recomputed':'PASS','source_files_fresh_SHA256':len(sources),
        'model_pre_and_post_SHA256_identity_files':17,'model_current_inode_device_bytes':'PASS',
        'input_steps_per_B_config':total_steps,'layer_writes_per_B_config':total_writes,
        'actual_learned_qk_transform_checks_in_formal':transform_checks,'worker_real_wait_integer':0,
        'worker_and_parent_epochs_fresh_gone':True,'own_NVML_fresh_released':True,'worker_numeric_pid_reused':worker_pid_reused,
        'numeric_pid_NVML_rows_if_new_epoch':numeric_pid_rows,'NVML_foreign_allowed':apps,
        'GPU_driver_inventory_postrun':gpu_info,'source_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-begin})
    print('INDEPENDENT_DATA_REVIEW_PASS',flush=True)

if __name__=='__main__':main()
