"""Explicit layer-local collector and physical INT8 recurrent-state bank.

Batch rows are independent experiments: 0=A FP reference, 1=B Native INT8,
2=B Key-H128 INT8. Only row 0 may have a persistent dense state.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

LAYERS = tuple(i for i in range(32) if i % 4 != 3)
METRICS = ('min','max','absmax','rms','peak_rms','kurtosis','scale',
           'qdq_rel_l2','boundary_count','out3_count','out6_count','squared_error',
           'squared_norm','zero_group','zero_variance','nonfinite_count')
CONDITIONS = ('A_Native','A_KeyH128','B_Native','B_KeyH128')
EPS = 1e-12

def json_default(obj):
    if isinstance(obj, (set, frozenset)): return sorted(obj)
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(f"Unsupported metadata type: {type(obj).__name__}")

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
        f.write('\n')

def encode(x):
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError('Nonfinite recurrent state rejected before INT8 conversion')
    scale = x.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return codes, scale

def decode(codes, scale):
    assert codes.dtype == torch.int8 and scale.dtype == torch.float32
    return codes.float() * scale

def rotate_state(s, h):
    # q_h=q@H; S_h=H^T@S. Sylvester H is symmetric, but the transpose is explicit.
    return torch.matmul(h.T, s)

def group_metrics(s):
    """One row per (batch,head,value); all 128 Key elements participate.

    Metric arithmetic is FP64; QDQ itself retains the frozen FP32 formula.
    Pearson kurtosis uses centered moments, population denominator K=128.
    """
    assert s.dtype == torch.float32 and s.shape[-2] == 128
    x = s.double()
    finite = torch.isfinite(x)
    nbad = (~finite).sum(dim=-2)
    good = nbad == 0
    safe = torch.where(finite, x, torch.zeros_like(x))
    sqnorm = safe.square().sum(dim=-2)
    rms = (sqnorm / 128).sqrt()
    absmax = safe.abs().amax(dim=-2)
    mean = safe.mean(dim=-2, keepdim=True)
    center = safe - mean
    variance = center.square().mean(dim=-2)
    kurt = center.pow(4).mean(dim=-2) / variance.square()
    kurt = torch.where((variance > 0) & good, kurt, torch.full_like(kurt, float('nan')))
    # Invalid inputs are counted above; caller fails instead of silently quantizing them.
    if not bool(good.all()):
        raise FloatingPointError(f'Collector saw {int(nbad.sum())} nonfinite elements')
    codes, scales = encode(s)
    dequant = decode(codes, scales)
    err = (dequant.double() - x).square().sum(dim=-2)
    metrics = [safe.amin(dim=-2), safe.amax(dim=-2), absmax, rms,
               absmax / (rms + EPS), kurt, scales.squeeze(-2).double(),
               err.sqrt() / (sqnorm.sqrt() + EPS),
               (codes.abs() == 127).sum(dim=-2),
               (safe.abs() > 3*rms.unsqueeze(-2)).sum(dim=-2),
               (safe.abs() > 6*rms.unsqueeze(-2)).sum(dim=-2),
               err, sqnorm, (absmax == 0), (variance == 0), nbad]
    return torch.stack([v.double() for v in metrics], dim=-1)

def histogram_edges():
    positive = np.exp2(np.linspace(-40, 20, 481, dtype=np.float64))
    return {'signed': np.r_[-positive[::-1], 0., positive],
            'absolute': np.r_[0., positive],
            'peak_rms': np.linspace(0., 12., 241, dtype=np.float64)}

class PhysicalStateBank:
    """INT8 storage is the source of every subsequent B read.

    fp[layer] stores a CLONE of batch row 0 ONLY, avoiding a view which would
    retain the other two FP32 rows. codes/scales contain rows 1 and 2 only.
    read_ptrs contain integers, never tensors or FP shadow copies.
    """
    def __init__(self, observer=None):
        self.fp = {}; self.codes = {}; self.scales = {}
        self.read_ptrs = {}; self.read_counts = {i:0 for i in LAYERS}
        self.write_counts = {i:0 for i in LAYERS}
        self.observer = observer; self.token = -1
        self.read_checks = 0

    def __len__(self): return 32

    def __getitem__(self, layer):
        if layer not in self.codes: return None
        dense_b = decode(self.codes[layer], self.scales[layer])
        current = torch.cat((self.fp[layer], dense_b), dim=0)
        self.read_ptrs[layer] = (self.token, current.data_ptr())
        self.read_counts[layer] += 1
        return current

    def __setitem__(self, layer, state):
        assert layer in LAYERS and state.dtype == torch.float32
        assert tuple(state.shape) == (3,32,128,128)
        if self.observer is not None:
            self.observer(layer, state)
        self.fp[layer] = state[0:1].detach().clone()
        self.codes[layer], self.scales[layer] = encode(state[1:3])
        assert self.fp[layer].untyped_storage().nbytes() == 32*128*128*4
        assert self.codes[layer].dtype == torch.int8
        assert self.scales[layer].shape == (2,32,1,128)
        self.write_counts[layer] += 1

    def check_consumed(self, layer, state):
        if self.token == 0:
            assert state is None
        else:
            assert self.read_ptrs[layer] == (self.token, state.data_ptr())
            assert state.dtype == torch.float32 and state.shape == (3,32,128,128)
            self.read_checks += 1

    def audit(self):
        def describe(t):
            return {'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),
                    'bytes':t.numel()*t.element_size(), 'storage_bytes':t.untyped_storage().nbytes()}
        first = LAYERS[0]
        return {'FP_reference_row0_only':describe(self.fp[first]),
                'B_codes_rows1_2':describe(self.codes[first]),
                'B_scale_rows1_2':describe(self.scales[first]),
                'persistent_B_dense_state':False,
                'write_counts':self.write_counts,'read_counts':self.read_counts,
                'core_read_pointer_checks':self.read_checks,
                'total_B_code_bytes':sum(t.numel()*t.element_size() for t in self.codes.values()),
                'total_B_scale_bytes':sum(t.numel()*t.element_size() for t in self.scales.values()),
                'FP_reference_bytes':sum(t.numel()*t.element_size() for t in self.fp.values())}

class Collector:
    def __init__(self, out, h, diagnostic=False):
        self.out=Path(out);self.h=h;self.diagnostic=diagnostic
        self.sample=None;self.token=None;self.positions=set();self.prompt=0
        self.rows={};self.coords={};self.hist={};self.slices={};self.seen=set()
        self.hist_zero={};self.cache_slices={}
        self.direct_checks=[];self.paired_norm_max=0.;self.coverage=[]
        self.edges=histogram_edges()
        self.tedges={k:torch.tensor(v,device=h.device,dtype=torch.float64) for k,v in self.edges.items()}
        self.event_meta=[]

    def start(self,sample,positions,prompt):
        self.sample=sample;self.positions=set(positions);self.prompt=prompt
        self.rows={};self.coords={};self.slices={};self.cache_slices={};self.seen=set();self.coverage=[]
        for cond in CONDITIONS:
            for layer in LAYERS:
                self.rows[cond,layer]=[];self.coords[cond,layer]=[];self.slices[cond,layer]=[]
                self.hist.setdefault((cond,layer),{k:np.zeros(len(v)+1,dtype=np.int64) for k,v in self.edges.items()})
                self.hist_zero.setdefault((cond,layer),0)
                if cond.startswith('B'):
                    self.cache_slices[cond,layer]=[]
                    self.hist.setdefault((cond+'_CACHE',layer),{k:np.zeros(len(v)+1,dtype=np.int64) for k,v in self.edges.items() if k!='peak_rms'})
                    self.hist_zero.setdefault((cond+'_CACHE',layer),0)

    def observe(self,layer,state):
        # Explicit layer argument and independent clone before any cache writeback.
        pos=self.token-self.prompt
        if pos not in self.positions:return
        key=(self.sample,pos,layer)
        if key in self.seen:raise RuntimeError(f'Duplicate collection: {key}')
        self.seen.add(key)
        snapshot=state.detach().clone()
        assert snapshot.data_ptr()!=state.data_ptr()
        sources=(snapshot[0:1],rotate_state(snapshot[0:1],self.h),snapshot[1:2],snapshot[2:3])
        n0=sources[0].double().square().sum(-2)
        n1=sources[1].double().square().sum(-2)
        norm_error=((n1-n0).abs()/(n0+EPS)).max().item()
        self.paired_norm_max=max(self.paired_norm_max,norm_error)
        if norm_error>2e-5:raise RuntimeError(f'Paired group L2 conservation failed: {norm_error}')
        for cond,s in zip(CONDITIONS,sources):
            m=group_metrics(s)[0]
            if layer in (LAYERS[0],LAYERS[-1]) and len(self.direct_checks)<8:
                self.direct_checks.append(independent_metrics_check(s,m,layer,self.sample,pos,cond))
            # FP64 metric arrays preserve exact observation values for later exact quantiles.
            self.rows[cond,layer].append(m.cpu().numpy())
            self.coords[cond,layer].append(pos)
            self.slices[cond,layer].append(s[0,0,:,0].cpu().numpy().copy())
            values={'signed':s.flatten().double(),'absolute':s.flatten().abs().double(),
                    'peak_rms':m[...,METRICS.index('peak_rms')].flatten()}
            for kind,x in values.items():
                ix=torch.bucketize(x,self.tedges[kind],right=True)
                counts=torch.bincount(ix,minlength=len(self.edges[kind])+1)
                self.hist[cond,layer][kind]+=counts.cpu().numpy()
            self.hist_zero[cond,layer]+=int((s==0).sum())
            if cond.startswith('B'):
                codes,scales=encode(s);cached=decode(codes,scales)
                self.cache_slices[cond,layer].append(cached[0,0,:,0].cpu().numpy().copy())
                for kind,x in {'signed':cached.flatten().double(),'absolute':cached.abs().flatten().double()}.items():
                    ix=torch.bucketize(x,self.tedges[kind],right=True)
                    counts=torch.bincount(ix,minlength=len(self.edges[kind])+1)
                    self.hist[cond+'_CACHE',layer][kind]+=counts.cpu().numpy()
                self.hist_zero[cond+'_CACHE',layer]+=int((cached==0).sum())
        self.event_meta.append({'experiment':'A_and_B_independent_batch_rows','sample_id':self.sample,
                                'token_idx':pos,'model_layer_id':layer,'heads':32,'values':128,
                                'key_elements_per_group':128,'state_dtype':'torch.float32',
                                'finite_elements_per_condition':32*128*128})

    def finish(self):
        expected={(self.sample,p,l) for p in self.positions for l in LAYERS}
        assert self.seen==expected,(len(self.seen),len(expected))
        for cond in CONDITIONS:
            exp=cond[0]
            for layer in LAYERS:
                path=self.out/exp/'groups'/f'{self.sample}__{cond}__L{layer:02d}.npz'
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise FileExistsError(path)
                with path.open('xb') as f:
                    np.savez_compressed(f,metrics=np.stack(self.rows[cond,layer]),metric_names=np.array(METRICS),
                                        experiment=exp,condition=cond,basis='KeyH128' if 'H128' in cond else 'Native',
                                        sample_id=self.sample,token_idx=np.array(self.coords[cond,layer]),
                                        model_layer_id=layer,head_idx=np.arange(32),value_idx=np.arange(128),
                                        state_dtype='float32',cache_dtype='float32' if exp=='A' else 'int8',
                                        scale_dtype='float32',key_axis=2,key_group_size=128)
                p=self.out/exp/'peak_matrices'/f'{self.sample}__{cond}__L{layer:02d}__H0V0.npz'
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:
                    np.savez_compressed(f,state=np.stack(self.slices[cond,layer]),
                                        token_idx=np.array(self.coords[cond,layer]),key_idx=np.arange(128),
                                        sample_id=self.sample,model_layer_id=layer,head_idx=0,value_idx=0,
                                        condition=cond,experiment=exp)
                if exp=='B':
                    p=self.out/exp/'peak_matrices'/f'{self.sample}__{cond}_CACHE__L{layer:02d}__H0V0.npz'
                    with p.open('xb') as f:
                        np.savez_compressed(f,state=np.stack(self.cache_slices[cond,layer]),
                                            token_idx=np.array(self.coords[cond,layer]),key_idx=np.arange(128),
                                            sample_id=self.sample,model_layer_id=layer,head_idx=0,value_idx=0,
                                            condition=cond+'_CACHE',experiment=exp,
                                            observation='decoded resident INT8 codes * FP32 scale after writeback')
        return {'sample_id':self.sample,'sampling_positions':sorted(self.positions),
                'sampled_token_count':len(self.positions),'layers':list(LAYERS),
                'sampled_layer_events':len(self.seen),'records_per_condition':len(self.seen)*32*128}

    def finish_all(self):
        for exp in ('A','B'):
            data={}
            for (cond,layer),counts in self.hist.items():
                if cond[0]!=exp:continue
                for kind,val in counts.items():data[f'{cond}__L{layer:02d}__{kind}']=val
                data[f'{cond}__L{layer:02d}__zero_elements']=np.array(self.hist_zero[cond,layer],dtype=np.int64)
            data.update({f'{k}_edges':v for k,v in self.edges.items()})
            with (self.out/exp/'histogram_counts.npz').open('xb') as f:np.savez_compressed(f,**data)
        dump(self.out/'COLLECTOR_VALIDATION.json',{'status':'PASS','real_direct_checks':self.direct_checks,
             'max_paired_group_relative_squared_L2_error':self.paired_norm_max,
             'explicit_layer_callbacks':True,'independent_cloned_snapshots':True,
             'quantiles':'not estimated here; exact numpy quantiles over saved FP64 group observations'})
        p=self.out/'collection_events.jsonl'
        with p.open('x') as f:
            for row in self.event_meta:f.write(json.dumps(row)+'\n')

def independent_metrics_check(s,actual,layer,sample,pos,cond):
    x=s.detach().cpu().numpy().astype(np.float64)
    # Direct NumPy calculation uses Key axis 2; torch QDQ rounding is checked
    # independently in float32 to retain the requested quantizer's precision.
    f=s.detach().cpu().numpy()
    maxabs=np.maximum(np.abs(f).max(axis=2,keepdims=True),np.float32(EPS))
    # PyTorch CUDA folds division by the constant127 to an FP32 reciprocal
    # multiply. This independently measured 1-ULP distinction changes the
    # oracle only, never encode(). CPU PyTorch retains direct division.
    scale=maxabs*np.float32(1/127) if s.device.type=='cuda' else maxabs/np.float32(127)
    code=np.clip(np.rint(f/scale),-127,127).astype(np.int8)
    dq=code.astype(np.float32)*scale
    rms=np.sqrt(np.mean(x*x,axis=2));absmax=np.abs(x).max(axis=2)
    centered=x-x.mean(axis=2,keepdims=True);var=np.mean(centered**2,axis=2)
    with np.errstate(invalid='ignore',divide='ignore'):
        kurt=np.mean(centered**4,axis=2)/(var**2)
    sqerr=np.sum((dq.astype(np.float64)-x)**2,axis=2);sqnorm=np.sum(x*x,axis=2)
    expected=np.stack([x.min(axis=2),x.max(axis=2),absmax,rms,absmax/(rms+EPS),kurt,
                       scale.squeeze(2),np.sqrt(sqerr)/(np.sqrt(sqnorm)+EPS),
                       (np.abs(code)==127).sum(axis=2),(np.abs(x)>3*rms[:,:,None,:]).sum(axis=2),
                       (np.abs(x)>6*rms[:,:,None,:]).sum(axis=2),sqerr,sqnorm,
                       absmax==0,var==0,np.zeros_like(absmax)],axis=-1)[0]
    got=actual.detach().cpu().numpy()
    assert np.array_equal(np.isnan(expected),np.isnan(got))
    finite=np.isfinite(expected)
    np.testing.assert_allclose(got[finite],expected[finite],rtol=2e-6,atol=1e-10)
    return {'sample_id':sample,'token_idx':pos,'model_layer_id':layer,'condition':cond,
            'all_heads':32,'all_values':128,'all_metrics':len(METRICS),
            'max_absolute_difference':float(np.max(np.abs(got[finite]-expected[finite]))),
            'scale_oracle':'CPU emulation of CUDA constant reciprocal multiply' if s.device.type=='cuda' else 'CPU direct divide',
            'status':'PASS'}
