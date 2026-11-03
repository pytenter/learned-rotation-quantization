"""One GPU4 model, qualified full-prompt teacher forcing; no generation or scoring."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from pathlib import Path
import gc
import json
import math
import resource
import subprocess
import sys
import time
import traceback
import torch
from rotation_adapter import Rotations, METHODS, file_sha, tensor_sha
from collector import Collector
from state_bank import PhysicalStateBank, make_cache
from vendor.prior_core import LAYERS, dump
from vendor.canonical_h128 import normalize_qk

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'CONFIG.json').read_text())

def progress(phase,**data):
    row={'phase':phase,'monotonic':time.monotonic(),**data}
    path=ROOT/'operations/PROGRESS.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row));os.replace(tmp,path)
    print(json.dumps(row),flush=True)

def relative(a,b):return float((a.double()-b.double()).norm()/(b.double().norm()+1e-12))

def operator_checks(fn,q,k,v,g,beta,initial,rotations,layer):
    q=normalize_qk(q[0:1]).float();k=normalize_qk(k[0:1]).float()
    v=v[0:1];g=g[0:1];beta=beta[0:1]
    initial=initial[0:1].clone() if initial is not None else torch.zeros((1,32,128,128),device=q.device)
    native_out,native_state=fn(q,k,v,g,beta,initial,True,False)
    qd=q[:,0].double()/math.sqrt(128);kd=k[:,0].double()
    sd=initial.double()*g[:,0].double().exp()[...,None,None]
    delta=(v[:,0].double()-(sd*kd[...,None]).sum(-2))*beta[:,0].double()[...,None]
    nd=sd+kd[...,None]*delta[...,None,:]
    od=(nd*qd[...,None]).sum(-2)[:,None]
    base={'FP32_vs_FP64_state':relative(native_state,nd),'FP32_vs_FP64_readout':relative(native_out,od)}
    assert max(base.values())<2e-5,base
    records=[]
    for method in METHODS:
        qr=rotations.qk(method,q,layer,False);kr=rotations.qk(method,k,layer,False)
        sr=rotations.to_basis(method,initial,layer)
        out,state=fn(qr,kr,v,g,beta,sr,True,False)
        record={'method':method,'model_layer_id':layer,'readout_relative_L2':relative(out,native_out),
                'state_recovery_relative_L2':relative(rotations.from_basis(method,state,layer),native_state),**base}
        assert max(record[n] for n in ['readout_relative_L2','state_recovery_relative_L2'])<2e-5,record
        records.append(record)
    return records

class CorePatch:
    def __init__(self,model,bank,rotations,real_checks=False):
        import transformers.models.qwen3_5.modeling_qwen3_5 as Q
        self.Q=Q;self.bank=bank;self.rot=rotations;self.real_checks=real_checks
        self.native_fn=Q.torch_recurrent_gated_delta_rule;self.chunk_fn=Q.torch_chunk_gated_delta_rule
        self.layers=model.model.language_model.layers;self.saved=[];self.counts={l:0 for l in LAYERS}
        self.transform_checks=[];self.operator_checks=[]

    def __enter__(self):
        for layer in LAYERS:
            module=self.layers[layer].linear_attn;self.saved.append((module,module.forward))
            def core(query,key,value,g,beta,initial_state=None,output_final_state=True,
                     use_qk_l2norm_in_kernel=False,_layer=layer,**kwargs):
                assert query.shape==key.shape==(self.bank.rows,1,32,128)
                assert query.dtype==key.dtype==torch.bfloat16 and output_final_state
                self.bank.check_consumed(_layer,initial_state)
                if self.real_checks and self.bank.fp_reference and self.bank.token==1:
                    self.operator_checks.extend(operator_checks(self.native_fn,query,key,value,g,beta,initial_state,self.rot,_layer))
                q=normalize_qk(query) if use_qk_l2norm_in_kernel else query
                k=normalize_qk(key) if use_qk_l2norm_in_kernel else key
                q=q.float();k=k.float();qs=[];ks=[]
                names=(['Native'] if self.bank.fp_reference else [])+list(self.bank.b_methods)
                for row,method in enumerate(names):
                    qr=self.rot.qk(method,q[row:row+1],_layer,False)
                    kr=self.rot.qk(method,k[row:row+1],_layer,False)
                    qs.append(qr);ks.append(kr)
                    if self.bank.token==0 and method in ('C6R','C7'):
                        oq=self.rot.original_qk(method,q[row:row+1],_layer,False)
                        ok=self.rot.original_qk(method,k[row:row+1],_layer,False)
                        assert torch.equal(qr,oq) and torch.equal(kr,ok)
                        fullq=self.rot.original_qk(method,query[row:row+1],_layer,use_qk_l2norm_in_kernel)
                        fullk=self.rot.original_qk(method,key[row:row+1],_layer,use_qk_l2norm_in_kernel)
                        error=max(relative(qr,fullq),relative(kr,fullk));assert error<2e-5
                        self.transform_checks.append({'method':method,'model_layer_id':_layer,
                            'FP32_H_then_Delta_original_evaluation_exact':True,
                            'original_full_normalize_then_H_Delta_exact':bool(torch.equal(qr,fullq) and torch.equal(kr,fullk)),
                            'original_full_transform_relative_L2':error,'input_dtype':str(query.dtype),
                            'normalization_before_rotation':True})
                out,state=self.native_fn(torch.cat(qs,0),torch.cat(ks,0),value,g,beta,initial_state,True,False)
                self.counts[_layer]+=1
                return out.to(value.dtype),state
            def forward(*args,_original=module.forward,_core=core,**kwargs):
                self.Q.torch_recurrent_gated_delta_rule=_core;self.Q.torch_chunk_gated_delta_rule=_core
                try:return _original(*args,**kwargs)
                finally:
                    self.Q.torch_recurrent_gated_delta_rule=self.native_fn;self.Q.torch_chunk_gated_delta_rule=self.chunk_fn
            module.forward=forward
        return self

    def __exit__(self,*args):
        for module,forward in self.saved:module.forward=forward

def run_tokens(model,rotations,plan,tokens,collector,real_checks=False,formal_sample=None):
    bank=PhysicalStateBank(plan['fp_reference'],plan['B_methods'],collector)
    cache=make_cache(model.config.text_config,bank)
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),CorePatch(model,bank,rotations,real_checks) as patch:
        for t,token in enumerate(tokens):
            bank.token=collector.token=t
            out=model.model.language_model(input_ids=torch.full((bank.rows,1),int(token),device='cuda:0',dtype=torch.long),
                    past_key_values=cache,cache_position=torch.tensor([t],device='cuda:0'),use_cache=True)
            if not bool(torch.isfinite(out.last_hidden_state).all()):raise FloatingPointError('Nonfinite model hidden state')
            if t%32==0 or t==len(tokens)-1:
                progress('FORMAL_TOKEN' if formal_sample else 'GPU_SMOKE_TOKEN',plan=plan['name'],sample_id=formal_sample,
                         input_tokens_committed=t+1,total_input_tokens=len(tokens),elapsed_seconds=time.monotonic()-begin)
        assert all(v==len(tokens) for v in bank.write_counts.values())
        assert all(v==len(tokens)-1 for v in bank.read_counts.values())
        assert bank.read_checks==24*(len(tokens)-1)
        assert all(v==len(tokens) for v in patch.counts.values())
        assert cache.get_seq_length()==len(tokens)
    torch.cuda.synchronize()
    runtime=time.monotonic()-begin
    result={'status':'PASS','sample_id':formal_sample,'plan':plan,'input_tokens':len(tokens),
            'physical_INT8_cache':bank.audit(cache),'core_calls_by_layer':patch.counts,
            'original_transform_runtime_checks':patch.transform_checks,'common_state_operator_checks':patch.operator_checks,
            'model_forward_and_observation_seconds':runtime,'allocated_bytes_at_end':torch.cuda.memory_allocated(),
            'reserved_bytes_at_end':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'full_attention_KV_bytes':sum(x.numel()*x.element_size() for l in cache.layers for x in [getattr(l,'keys',None),getattr(l,'values',None)] if isinstance(x,torch.Tensor)),
            'GDN_conv_bytes':sum(x.numel()*x.element_size() for l in cache.layers for x in getattr(l,'conv_states',{}).values() if isinstance(x,torch.Tensor))}
    del out,cache,bank,patch;gc.collect();torch.cuda.empty_cache()
    return result

def check_owned_gpu():
    command=['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader,nounits']
    output=subprocess.check_output(command,text=True)
    rows=[x for x in output.splitlines() if CFG['gpu_uuid'] in x]
    assert all(int(x.split(',')[0])==os.getpid() for x in rows),'GPU4 has a foreign process; refusing next sample'
    return output

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==CFG['gpu_uuid']
    assert json.loads((ROOT/'gates/CPU_AUDIT.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'gates/CPU_COLLECTOR_BANK_GATE.json').read_text())['status']=='PASS'
    for f in json.loads((ROOT/'provenance/SOURCE_MANIFEST.json').read_text()):assert file_sha(ROOT/f['path'])==f['sha256']
    assert torch.cuda.device_count()==1
    torch.set_num_threads(8);torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1);torch.cuda.manual_seed(1)
    progress('MODEL_LOADING')
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
    model,info=Qwen3_5ForConditionalGeneration.from_pretrained(CFG['model'],torch_dtype=torch.bfloat16,
             device_map={'':0},local_files_only=True,attn_implementation='sdpa',output_loading_info=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert not info['missing_keys'] and not info['mismatched_keys'] and all('mtp' in x for x in info.get('unexpected_keys',[]))
    assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
    assert [i for i,l in enumerate(model.model.language_model.layers) if hasattr(l,'linear_attn')]==list(LAYERS)
    prior=json.loads((ROOT/'provenance/PRIOR_RUNTIME.json').read_text())
    assert file_sha(Q.__file__)==prior['modeling_sha256']
    rotations=Rotations(ROOT,'cuda:0')
    for method in ('C6R','C7'):
        manifest=json.loads((ROOT/'provenance'/f'{method.lower()}_rotation_manifest.json').read_text())
        for layer in LAYERS:assert tensor_sha(rotations.delta(method,layer))==manifest['per_layer_sha256'][str(layer)]
    stat=(Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()
    runtime={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,
             'modeling_file':Q.__file__,'modeling_sha256':file_sha(Q.__file__),'model_loading_info':info,
             'gpu_name':torch.cuda.get_device_name(0),'physical_gpu_index':CFG['gpu_index'],'CVD':os.environ['CUDA_VISIBLE_DEVICES'],
             'pid':os.getpid(),'PPID':os.getppid(),'starttick':int(stat[19]),
             'argv':[x.decode() for x in (Path('/proc')/str(os.getpid())/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')],
             'NVML':check_owned_gpu(),'rotation_load_identities':rotations.load_identity,'batch_plan':CFG['batch_plan'],
             'normalization':'original BF16 normalization before FP32 H then FP32 frozen Delta',
             'FP32_effective_matrix_used_as_fused_qk_shortcut':False,'model_weights_modified':False}
    dump(ROOT/'provenance/ACTUAL_RUNTIME.json',runtime)
    tokenizer=AutoTokenizer.from_pretrained(CFG['model'],local_files_only=True)
    smoke_ids=tokenizer.encode('This is a short synthetic recurrent state rotation check.',add_special_tokens=False)[:8]
    assert len(smoke_ids)==8
    dump(ROOT/'gates/GPU_SMOKE_INPUT.json',{'text':'This is a short synthetic recurrent state rotation check.',
                                         'token_ids':smoke_ids,'AIME':False})
    gates=[]
    for plan in CFG['batch_plan']:
        progress('GPU_SMOKE',plan=plan['name'])
        collector=Collector(ROOT/'gates/gpu'/plan['name'],rotations,plan['fp_reference'],plan['B_methods'],True)
        collector.start('non_AIME_smoke',[0,1,7],0)
        result=run_tokens(model,rotations,plan,smoke_ids,collector,True)
        result['coverage']=collector.finish();collector.finish_all();gates.append(result)
        del collector;gc.collect();torch.cuda.empty_cache()
    assert len(gates[0]['common_state_operator_checks'])==24*4
    assert len(gates[1]['original_transform_runtime_checks'])==24*2
    dump(ROOT/'gates/GPU_SMOKE_RESULT.json',{'status':'PASS','plans':gates,'all_four_basis_common_state_real_layer_checks':96})
    progress('GPU_SMOKE_PASS_FORMAL_START')
    (ROOT/'formal').mkdir(exist_ok=False)
    all_results=[];formal_start=time.monotonic()
    for plan in CFG['batch_plan']:
        collector=Collector(ROOT/'formal'/plan['name'],rotations,plan['fp_reference'],plan['B_methods'])
        for sample in CFG['samples']:
            check_owned_gpu()
            data=json.loads((ROOT/'inputs'/f'{sample}.json').read_text())
            assert len(data['continuation_ids'])==2048
            torch.manual_seed(1);torch.cuda.manual_seed(1)
            collector.start(sample,data['sampling_positions_zero_based_continuation'],len(data['prompt_ids']))
            begin=time.monotonic()
            result=run_tokens(model,rotations,plan,data['prompt_ids']+data['continuation_ids'],collector,False,sample)
            progress('COMPRESSING_COLLECTED_SAMPLE',plan=plan['name'],sample_id=sample)
            result['coverage']=collector.finish();result['prompt_tokens']=len(data['prompt_ids']);result['continuation_tokens']=2048
            result['with_compression_seconds']=time.monotonic()-begin
            dump(ROOT/'formal'/plan['name']/f'{sample}_RESULT.json',result);all_results.append(result)
            progress('SAMPLE_COMPLETE',plan=plan['name'],sample_id=sample,seconds=result['with_compression_seconds'])
        collector.finish_all();del collector;gc.collect();torch.cuda.empty_cache()
    assert len(all_results)==8
    dump(ROOT/'formal/RESULT.json',{'status':'PASS','A':{m:'PASS' for m in METHODS},'B':{m:'PASS' for m in METHODS},
         'results':all_results,'seconds':time.monotonic()-formal_start,'new_AIME_generation':0,'new_scoring':0,
         'B_total_input_tokens_per_configuration':8602,'B_total_layer_writes_per_configuration':206448})
    progress('GPU_WORK_COMPLETE',completed_phase_samples=len(all_results))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        path=ROOT/'operations/FIRST_FAILURE.json'
        if not path.exists():dump(path,{'error':str(error),'traceback':traceback.format_exc(),'GPU4_only':True})
        traceback.print_exc();sys.exit(1)
