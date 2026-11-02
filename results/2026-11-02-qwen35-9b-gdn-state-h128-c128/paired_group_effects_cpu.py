"""Exact SAME-State group win/loss counts for A, not layer-median comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,csv,json
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);rows=[]
layers=[i for i in range(32) if i%4!=3]
samples=[s['sample_id'] for s in json.loads((root/'inputs/SELECTION.json').read_text())['selected']]
fields=['absmax','peak_rms','kurtosis','scale','qdq_rel_l2']
for layer in layers:
    pairs=[]
    for sid in samples:
        files=[root/'formal/A/groups'/f'{sid}__A_{b}__L{layer:02d}.npz' for b in ['Native','KeyH128']]
        with np.load(files[0]) as a,np.load(files[1]) as b:
            assert np.array_equal(a['token_idx'],b['token_idx'])
            names=list(a['metric_names']);idx=[names.index(f) for f in fields]
            x=a['metrics'][...,idx].reshape(-1,len(fields));y=b['metrics'][...,idx].reshape(-1,len(fields))
            pairs.append((x.copy(),y.copy()))
    x=np.concatenate([p[0] for p in pairs]);y=np.concatenate([p[1] for p in pairs]);del pairs
    row={'experiment':'A','model_layer_id':layer,'pairing':'same_FP_State_same_sample_token_head_value','groups':len(x)}
    for j,name in enumerate(fields):
        good=np.isfinite(x[:,j])&np.isfinite(y[:,j]);a=x[good,j];b=y[good,j]
        row[name+'_valid_pairs']=int(good.sum())
        row[name+'_H_lower_pairs']=int((b<a).sum());row[name+'_H_higher_pairs']=int((b>a).sum());row[name+'_equal_pairs']=int((b==a).sum())
        row[name+'_H_lower_fraction']=float(np.mean(b<a)) if len(a) else float('nan')
        row[name+'_H_higher_fraction']=float(np.mean(b>a)) if len(a) else float('nan')
    a=x[:,-1];b=y[:,-1]
    for name,values in [('qdq_error_difference_H_minus_N',b-a),('qdq_error_ratio_H_over_N_positiveN',b[a>0]/a[a>0])]:
        for label,value in zip(['median','p95','p99'],np.quantile(values,[.5,.95,.99],method='linear')):row[name+'_'+label]=float(value)
    row['zero_Native_qdq_error_pairs']=int((a==0).sum())
    rows.append(row);print('SAME_STATE_GROUP_EFFECTS',layer,flush=True)
with (root/'analysis/A_paired_group_effects.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('ALL_A_PAIRED_GROUP_EFFECTS_COMPLETE',flush=True)
