from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LAYERS=tuple(i for i in range(32) if i%4!=3)
METHODS=('Native','Hadamard','C6R','C7')
SAMPLES=tuple(f'aime26_{i:02d}' for i in range(1,5))
NAMES=('min','max','absmax','rms','peak_rms','kurtosis','scale','qdq_rel_l2',
       'boundary_count','out3_count','out6_count','squared_error','squared_norm',
       'zero_group','zero_variance','nonfinite_count')
IDX={n:i for i,n in enumerate(NAMES)}
QUANTILES=('absmax','rms','peak_rms','kurtosis','scale','qdq_rel_l2')
PAIRS=(('C6R','Hadamard'),('C7','Hadamard'),('C7','C6R'),
       ('Hadamard','Native'),('C6R','Native'),('C7','Native'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def safe_json(x):
    if isinstance(x,dict):return {str(k):safe_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe_json(v) for v in x]
    if isinstance(x,np.generic):return safe_json(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def jd(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(safe_json(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class CSVStream:
    def __init__(self,path):self.path=Path(path);self.file=None;self.writer=None;self.count=0
    def write(self,row):
        if self.writer is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.file=self.path.open('x',newline='')
            self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.count+=1
    def close(self):
        if self.file:self.file.close()

def summary_function():
    path=Path(__file__).with_name('prior_analysis_source.py')
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
    space={'np':np,'NAMES':NAMES,'IDX':IDX,'QUANTILES':QUANTILES}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
    return space['summarize']

legacy_summarize=summary_function()

def summarize(values):
    result=legacy_summarize(values)
    result['undefined_kurtosis_groups']=result['group_observations']-result['kurtosis_valid_groups']
    assert result['undefined_kurtosis_groups']==result['zero_variance_groups']
    assert result['nonfinite_elements']==0
    return result

def metadata(exp,method,layer):
    return {'experiment':exp,'method':method,'condition':f'{exp}_{method}','basis':method,'model_layer_id':layer,
            'source_state_dtype':'float32','cache_dtype':'none_observation_of_shared_native_FP32_reference' if exp=='A' else 'int8',
            'scale_dtype':'float32','key_group_size':128,
            'comparison':'same_FP_State_strict_pair' if exp=='A' else 'same_inputs_independent_INT8_trajectory',
            'observation':'transformed_common_FP_snapshot' if exp=='A' else 'updated_FP32_before_INT8_write'}

def phase_root(exp,method):
    phase='A_ALL_B_NATIVE_H' if exp=='A' or method in ('Native','Hadamard') else 'B_LEARNED'
    return ROOT/'formal'/phase

def data_path(exp,method,layer,sample,kind='groups'):
    return phase_root(exp,method)/exp/kind/f'{sample}__{exp}_{method}__L{layer:02d}.npz'

def load_layer(exp,layer):
    values={};peaks={};positions=None
    for method in METHODS:
        blocks=[];indices=[]
        for sample in SAMPLES:
            with np.load(data_path(exp,method,layer,sample)) as z:
                assert tuple(z['metric_names'])==NAMES
                assert int(z['model_layer_id'])==layer and str(z['sample_id'])==sample
                assert str(z['condition'])==f'{exp}_{method}'
                m=z['metrics'].copy();ix=z['key_absmax_index'].copy();p=z['token_idx'].copy()
            assert m.shape==(129,32,128,16) and ix.shape==(129,32,128)
            expected=np.r_[0,np.arange(15,2048,16)]
            assert np.array_equal(p,expected)
            if positions is None:positions=p
            assert np.array_equal(positions,p)
            assert np.isfinite(m[...,list(range(5))+list(range(6,16))]).all()
            assert ((ix>=0)&(ix<128)).all()
            blocks.append(m);indices.append(ix)
        values[method]=np.concatenate(blocks,axis=0)
        peaks[method]=np.concatenate(indices,axis=0)
    return values,peaks,positions

def pearson(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1)
    valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
    if len(a)<2:return None
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else None

def paired(candidate,reference):
    result={}
    for name in ['peak_rms','qdq_rel_l2','out3_count','out6_count']:
        a=candidate[...,IDX[name]].reshape(-1);b=reference[...,IDX[name]].reshape(-1)
        valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid];n=len(a)
        lower=int((a<b).sum());higher=int((a>b).sum());equal=int((a==b).sum())
        assert lower+higher+equal==n
        result.update({name+'_valid_pairs':n,name+'_candidate_lower':lower,name+'_candidate_higher':higher,name+'_equal':equal,
                       name+'_lower_fraction':lower/n if n else None,name+'_higher_fraction':higher/n if n else None,
                       name+'_equal_fraction':equal/n if n else None})
        q=np.quantile(a-b,[.5,.95,.99],method='linear') if n else [np.nan]*3
        for label,value in zip(['median','p95','p99'],q):result[name+'_candidate_minus_reference_'+label]=float(value)
        if name=='qdq_rel_l2':
            positive=b>0;result['qdq_reference_zero_pairs']=int((b==0).sum())
            result['qdq_ratio_valid_positive_reference_pairs']=int(positive.sum())
            q=np.quantile(a[positive]/b[positive],[.5,.95,.99],method='linear') if positive.any() else [np.nan]*3
            for label,value in zip(['median','p95','p99'],q):result['qdq_ratio_candidate_over_positive_reference_'+label]=float(value)
    return result

def top_indices(values,k=5):
    x=np.asarray(values).reshape(-1);threshold=np.partition(x,len(x)-k)[len(x)-k]
    greater=np.flatnonzero(x>threshold);equal=np.flatnonzero(x==threshold)[:k-len(greater)]
    selected=np.r_[greater,equal]
    return selected[np.lexsort((selected,-x[selected]))]

def histograms(exp):
    roots={phase_root(exp,m) for m in METHODS};merged={}
    for root in sorted(roots):
        with np.load(root/exp/'histogram_counts.npz') as z:
            for key in z.files:
                data=z[key].copy()
                if key in merged:assert np.array_equal(merged[key],data),key
                else:merged[key]=data
    return merged
