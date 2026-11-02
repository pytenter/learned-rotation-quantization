import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import sys,json,time,gc,traceback,hashlib,subprocess,contextlib
from pathlib import Path
import numpy as np
import torch
from experiment_core import *
from vendor.canonical_h128 import hadamard,normalize_qk

ROOT=Path(__file__).resolve().parent
MODEL='/data/zypan/modelscope_models/Qwen3.5-9B'
GPU_UUID='GPU-de1f9152-32e5-a033-d2bf-ea7abecb606f'

def progress(phase,**kw):
    row={'phase':phase,'monotonic':time.monotonic(),**kw}
    p=ROOT/'operations/PROGRESS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(row));os.replace(tmp,p)
    print(json.dumps(row),flush=True)

def relative(a,b):return float((a.double()-b.double()).norm()/(b.double().norm()+EPS))

def operator_check(Q,q,k,v,g,beta,initial,h,layer,fn=None):
    # Independent real-layer common-state comparison, normalization BEFORE H.
    q=normalize_qk(q[0:1]).float();k=normalize_qk(k[0:1]).float()
    v=v[0:1];g=g[0:1];beta=beta[0:1]
    s=initial[0:1].clone() if initial is not None else torch.zeros((1,32,128,128),device=q.device)
    fn=Q.torch_recurrent_gated_delta_rule if fn is None else fn
    o,n=fn(q,k,v,g,beta,s,True,False)
    oh,nh=fn(q@h,k@h,v,g,beta,rotate_state(s,h),True,False)
    # Literal single-step FP64 equation; axis semantics verified by contraction.
    qd=q[:,0].double()/math.sqrt(128);kd=k[:,0].double()
    sd=s.double()*g[:,0].double().exp()[...,None,None]
    delta=(v[:,0].double()-(sd*kd[...,None]).sum(-2))*beta[:,0].double()[...,None]
    nd=sd+kd[...,None]*delta[...,None,:]
    od=(nd*qd[...,None]).sum(-2)[:,None]
    r={'model_layer_id':layer,'common_state_rotation_output_relative_L2':relative(oh,o),
       'recovered_state_relative_L2':relative(rotate_state(nh,h),n),
       'FP32_vs_independent_FP64_state_relative_L2':relative(n,nd),
       'FP32_vs_independent_FP64_readout_relative_L2':relative(o,od),
       'native_state_shape':list(n.shape),'state_stride':list(n.stride()),
       'key_contraction_axis':-2,'qk_transform':'normalize BF16 then float32 @ normalized Sylvester H128'}
    assert max(r[k] for k in r if k.endswith('relative_L2'))<2e-5,r
    r['status']='PASS';return r

class CorePatch:
    def __init__(self,model,bank,collector=None,real_checks=False):
        import transformers.models.qwen3_5.modeling_qwen3_5 as Q
        self.Q=Q;self.bank=bank;self.collector=collector;self.real_checks=real_checks
        self.h=hadamard(device='cuda:0');self.saved=[];self.real=[]
        self.native_fn=Q.torch_recurrent_gated_delta_rule
        self.chunk_fn=Q.torch_chunk_gated_delta_rule
        self.counts={i:0 for i in LAYERS}
        self.layers=model.model.language_model.layers

    def __enter__(self):
        for i in LAYERS:
            mod=self.layers[i].linear_attn
            self.saved.append((mod,mod.forward))
            # Even token0 uses the exact one-step recurrent equation, rather
            # than a padded multi-token chunk. The original one-token chunk
            # equivalence is a separate gate. No model weights are patched.
            def call(query,key,value,g,beta,initial_state=None,output_final_state=True,
                     use_qk_l2norm_in_kernel=False,_layer=i,**kwargs):
                assert query.shape==(3,1,32,128)
                self.bank.check_consumed(_layer,initial_state)
                if self.real_checks and self.bank.token==1 and _layer in (0,30):
                    self.real.append(operator_check(self.Q,query,key,value,g,beta,initial_state,self.h,_layer,self.native_fn))
                q=normalize_qk(query) if use_qk_l2norm_in_kernel else query
                k=normalize_qk(key) if use_qk_l2norm_in_kernel else key
                q=q.float();k=k.float()
                # Rotate only B-Hadamard row. A's rotated snapshot is observational.
                q=torch.cat((q[:2],q[2:3]@self.h),0)
                k=torch.cat((k[:2],k[2:3]@self.h),0)
                out,state=self.native_fn(q,k,value,g,beta,initial_state,output_final_state,False)
                self.counts[_layer]+=1
                return out.to(value.dtype),state
            def forward(*args,_original=mod.forward,_core=call,**kwargs):
                # Scope each core callback to this module's explicit original
                # layer ID. There is no traced loop-local layer counter.
                self.Q.torch_recurrent_gated_delta_rule=_core
                self.Q.torch_chunk_gated_delta_rule=_core
                try:return _original(*args,**kwargs)
                finally:
                    self.Q.torch_recurrent_gated_delta_rule=self.native_fn
                    self.Q.torch_chunk_gated_delta_rule=self.chunk_fn
            mod.forward=forward
        return self

    def __exit__(self,*args):
        for mod,forward in self.saved:mod.forward=forward

def make_cache(config,bank):
    from transformers import DynamicCache
    import types
    cache=DynamicCache(config=config)
    class Slot(dict):
        def __init__(self,layer):super().__init__({0:None});self.layer=layer
        def __getitem__(self,state_idx):
            assert state_idx==0
            return bank[self.layer]
    for layer in LAYERS:cache.layers[layer].recurrent_states=Slot(layer)
    def update(self,state,layer_idx,state_idx=0,**kwargs):
        assert state_idx==0
        bank[layer_idx]=state
        self.layers[layer_idx].is_recurrent_states_initialized[0]=True
        # Native forward ignores the return. Returning no dense tensor avoids
        # retaining a decoded INT8 state in any cache slot.
        return None
    cache.update_recurrent_state=types.MethodType(update,cache)
    return cache

def synthetic_gate(h):
    torch.manual_seed(20261102)
    orth=(h.double().T@h.double()-torch.eye(128,device=h.device,dtype=torch.float64)).abs().max().item()
    assert orth<2e-6
    snap={}
    bank=PhysicalStateBank(lambda layer,s:snap.update({layer:s.detach().clone()}))
    checks=[]
    for i,scale in [(0,1.),(30,13.)]:
        x=torch.randn((3,32,128,128),device=h.device)*scale
        x[:,0,7,23]=57*scale
        initial=x.clone();bank[i]=x
        m=group_metrics(x[0:1])[0]
        checks.append(independent_metrics_check(x[0:1],m,i,'synthetic',0,'Native'))
        x.zero_();assert torch.equal(snap[i],initial)
        assert torch.equal(bank.fp[i],initial[0:1])
        c,s=encode(initial[1:]);assert torch.equal(c,bank.codes[i]) and torch.equal(s,bank.scales[i])
    assert snap[0].abs().max()<snap[30].abs().max()
    zero=torch.zeros((1,32,128,128),device=h.device)
    z=group_metrics(zero)
    assert torch.isnan(z[...,5]).all() and (z[...,13]==1).all()
    c,s=encode(zero);assert (c==0).all() and (s>0).all()
    for v in (float('nan'),float('inf'),-float('inf')):
        bad=zero.clone();bad[0,0,3,9]=v
        try:encode(bad)
        except FloatingPointError:pass
        else:raise AssertionError('Nonfinite not rejected')
    x=torch.randn_like(zero)
    rt=relative(rotate_state(rotate_state(x,h),h),x);assert rt<2e-6
    ties=torch.tensor([-2.5,-1.5,-.5,.5,1.5,2.5],device=h.device)
    assert torch.equal(torch.round(ties),torch.tensor([-2.,-2.,-0.,0.,2.,2.],device=h.device))
    return {'status':'PASS','orthogonality_max_abs':orth,'roundtrip_relative_L2':rt,
            'synthetic_explicit_layer_checks':checks,'snapshot_mutation':'PASS',
            'zero_groups':'zero codes, positive scale eps/127, undefined Pearson kurtosis',
            'nonfinite':'NaN/+Inf/-Inf rejected before integer conversion',
            'rounding':'torch.round ties-to-even; -127..127; boundary occupancy is not clipping'}

def run_gate(model,Q,h):
    synthetic=synthetic_gate(h)
    out=ROOT/'gates/real_collector';out.mkdir(parents=True,exist_ok=False)
    collector=Collector(out,h,diagnostic=True);collector.start('real_gate',[0,1],0)
    bank=PhysicalStateBank(collector.observe)
    cfg=model.config.text_config
    cache=make_cache(cfg,bank)
    sample=json.loads(next((ROOT/'inputs').glob('aime26_*.json')).read_text())
    tokens=sample['prompt_ids'][:2]
    with torch.inference_mode(),CorePatch(model,bank,collector,real_checks=True) as patch:
        for t,tok in enumerate(tokens):
            bank.token=t;collector.token=t
            model.model.language_model(input_ids=torch.full((3,1),tok,device='cuda:0',dtype=torch.long),past_key_values=cache,
                                      cache_position=torch.tensor([t],device='cuda:0'),use_cache=True)
    assert len(patch.real)==2
    assert all(c==2 for c in bank.write_counts.values())
    assert all(c==1 for c in bank.read_counts.values())
    assert bank.read_checks==24
    # Original chunk and recurrent implementations compared at one-token initial state.
    q=torch.randn((1,1,32,128),device='cuda:0',dtype=torch.bfloat16)
    k=torch.randn_like(q);v=torch.randn_like(q);g=-torch.rand((1,1,32),device='cuda:0');b=torch.rand_like(g)
    a,s=Q.torch_recurrent_gated_delta_rule(q,k,v,g,b,None,True,True)
    aa,ss=Q.torch_chunk_gated_delta_rule(q,k,v,g,b,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
    chunk={'output_relative_L2':relative(aa,a),'state_relative_L2':relative(ss,s)}
    assert max(chunk.values())<2e-5,chunk
    coverage=collector.finish();collector.finish_all()
    result={'status':'PASS','synthetic':synthetic,'real_layer_operator_checks':patch.real,
            'one_token_chunk_vs_recurrent':chunk,'collector_coverage':coverage,'physical_cache':bank.audit(),
            'batch_row_semantics':'independent rows; row0 FP reference, row1 Native INT8, row2 KeyH128 INT8',
            'real_group_direct_checks':collector.direct_checks}
    dump(ROOT/'gates/GATE_RESULT.json',result)
    del collector,bank,cache;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
    assert torch.cuda.device_count()==1
    torch.set_num_threads(8);torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1);torch.cuda.manual_seed(1)
    progress('MODEL_LOADING')
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    from transformers import Qwen3_5ForConditionalGeneration
    model,info=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
        device_map={'':0},local_files_only=True,attn_implementation='sdpa',output_loading_info=True)
    model.eval()
    assert not info['missing_keys'] and not info['mismatched_keys'],info
    # The checkpoint may contain dedicated MTP weights unused by the causal decoder.
    assert all('mtp' in x for x in info.get('unexpected_keys',[])),info
    layers=model.model.language_model.layers
    actual=[i for i,l in enumerate(layers) if hasattr(l,'linear_attn')]
    assert actual==list(LAYERS)
    assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
    runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,
             'python':sys.version,'device_name':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
             'model_loading_info':info,'layers':actual,'full_attention_layers':[i for i in range(32) if i not in LAYERS],
             'modeling_file':Q.__file__,'modeling_sha256':hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest(),
             'weights':'unchanged BF16 full checkpoint; only language decoder executed, LM head/vision not needed for teacher-forced state collection',
             'attention_backend':'sdpa','batch_size':3,'persistent_state_design':'FP32 row0 separate clone; rows1/2 INT8 codes + FP32 scales'}
    dump(ROOT/'provenance/RUNTIME.json',runtime)
    h=hadamard(device='cuda:0')
    progress('COLLECTOR_AND_PHYSICAL_INT8_GATES')
    run_gate(model,Q,h)
    progress('GATES_PASS_FORMAL_START')
    (ROOT/'formal').mkdir(exist_ok=False)
    collector=Collector(ROOT/'formal',h)
    results=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for path in sorted((ROOT/'inputs').glob('aime26_*.json')):
        sample=json.loads(path.read_text());sid=sample['sample_id'];prompt=sample['prompt_tokens']
        positions=sample['sampling_positions_zero_based_continuation']
        collector.start(sid,positions,prompt)
        bank=PhysicalStateBank(collector.observe)
        cache=make_cache(model.config.text_config,bank)
        tokens=sample['prompt_ids']+sample['continuation_ids'];begin=time.monotonic()
        with torch.inference_mode(),CorePatch(model,bank,collector) as patch:
            for t,tok in enumerate(tokens):
                bank.token=t;collector.token=t
                out=model.model.language_model(input_ids=torch.full((3,1),tok,device='cuda:0',dtype=torch.long),
                    past_key_values=cache,cache_position=torch.tensor([t],device='cuda:0'),use_cache=True)
                if not bool(torch.isfinite(out.last_hidden_state).all()):raise FloatingPointError('Nonfinite model output')
                if t%32==0 or t==len(tokens)-1:
                    progress('FORMAL_TOKEN',sample_id=sid,input_tokens_committed=t+1,total=len(tokens),
                             continuation_committed=max(0,t+1-prompt),sampled_events=len(collector.seen),
                             elapsed_seconds=time.monotonic()-begin)
            assert all(v==len(tokens) for v in bank.write_counts.values())
            assert all(v==len(tokens)-1 for v in bank.read_counts.values())
            assert bank.read_checks==24*(len(tokens)-1)
            assert all(v==len(tokens) for v in patch.counts.values())
            assert cache.get_seq_length()==len(tokens)
        coverage=collector.finish()
        result={'status':'PASS','sample_id':sid,'prompt_tokens':prompt,'continuation_tokens':len(sample['continuation_ids']),
                'forwards':len(tokens),'batch_rows':3,'collection':coverage,'physical_INT8_cache':bank.audit(),
                'core_calls_by_layer':patch.counts,'seconds':time.monotonic()-begin,
                'allocated_bytes_at_end':torch.cuda.memory_allocated(),'reserved_bytes_at_end':torch.cuda.memory_reserved(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'full_attention_KV_bytes':sum(x.numel()*x.element_size() for l in cache.layers for x in [getattr(l,'keys',None),getattr(l,'values',None)] if isinstance(x,torch.Tensor)),
                'GDN_conv_bytes':sum(x.numel()*x.element_size() for l in cache.layers for x in getattr(l,'conv_states',{}).values() if isinstance(x,torch.Tensor))}
        dump(ROOT/'formal'/f'{sid}_RESULT.json',result);results.append(result)
        del out,bank,cache,patch;gc.collect();torch.cuda.empty_cache()
    collector.finish_all()
    dump(ROOT/'formal/RESULT.json',{'status':'PASS','A':'PASS' if len(results)==4 else 'PARTIAL',
         'B':'PASS' if len(results)==4 else 'PARTIAL','samples':results,'seconds':time.monotonic()-start,
         'total_continuation_tokens_per_trajectory':sum(r['continuation_tokens'] for r in results),
         'total_prompt_tokens_per_trajectory':sum(r['prompt_tokens'] for r in results)})
    progress('GPU_WORK_COMPLETE',samples=len(results))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        p=ROOT/'operations/FIRST_FAILURE.json'
        if not p.exists():dump(p,{'error':str(e),'traceback':traceback.format_exc()})
        traceback.print_exc();sys.exit(1)
