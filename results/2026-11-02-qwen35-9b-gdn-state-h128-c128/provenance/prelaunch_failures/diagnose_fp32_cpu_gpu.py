
import os,json,torch,numpy as np
from pathlib import Path
from experiment_core import encode,decode,group_metrics,METRICS,dump
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-de1f9152-32e5-a033-d2bf-ea7abecb606f'
torch.set_num_threads(8);torch.manual_seed(20261102)
rows=[]
for layer,factor in [(0,1.),(30,13.)]:
 x=torch.randn((3,32,128,128),device='cuda:0')*factor;x[:,0,7,23]=57*factor;s=x[:1]
 c,sc=encode(s);d=decode(c,sc)
 f=s.cpu().numpy();c=c.cpu().numpy();sc=sc.cpu().numpy();d=d.cpu().numpy()
 cpu_div=np.maximum(np.abs(f).max(2,keepdims=True),np.float32(1e-12))/np.float32(127)
 cpu_mul=np.maximum(np.abs(f).max(2,keepdims=True),np.float32(1e-12))*np.float32(1/127)
 cc=np.clip(np.rint(f/sc),-127,127).astype(np.int8)
 dc=c.astype(np.float32)*sc
 mismatch=c!=cc
 expected_rel=np.sqrt(np.square(dc.astype(np.float64)-f.astype(np.float64)).sum(2))/(np.sqrt(np.square(f.astype(np.float64)).sum(2))+1e-12)
 actual=group_metrics(s).cpu().numpy()[...,METRICS.index('qdq_rel_l2')]
 rows.append({'layer':layer,'scale_CUDA_equals_CPU_divide':bool(np.array_equal(sc,cpu_div)),
 'scale_CUDA_equals_CPU_reciprocal_multiply':bool(np.array_equal(sc,cpu_mul)),
 'max_scale_ULP_from_CPU_division':float(np.max(np.abs(sc.astype(np.float64)-cpu_div)/np.spacing(cpu_div))),
 'codes_diff_from_CPU_with_actual_scale':int(mismatch.sum()),
 'max_divided_code_distance_to_half_if_different':float(np.max(np.abs((f/sc)[mismatch]-np.floor((f/sc)[mismatch])-.5))) if mismatch.any() else 0.,
 'CPU_codes_times_scale_equals_CUDA_dequant':bool(np.array_equal(dc,d)),
 'independent_actual_payload_metric_max_abs':float(np.max(np.abs(expected_rel-actual))),
 'same_scale_code_max_difference':int(np.abs(c.astype(np.int16)-cc.astype(np.int16)).max())})
r={'status':'DIAGNOSED_NOT_FORMAL','pid':os.getpid(),'CVD':os.environ['CUDA_VISIBLE_DEVICES'],'rows':rows,
 'scope':'FP32 CUDA constant division vs CPU divide, independent payload reconstruction, no model run'}
dump(Path(__file__).resolve().parent/'unused.json',{}) if False else None
print(json.dumps(r))
