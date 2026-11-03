"""Layer-explicit cloned observations; bounded sample/layer memmaps, never State replay."""
from pathlib import Path
import gc
import json
import numpy as np
import torch
from rotation_adapter import METHODS
from vendor.prior_core import (LAYERS, METRICS, EPS, dump, group_metrics,
                               independent_metrics_check, histogram_edges)

class Collector:
    def __init__(self, out, rotations, fp_reference, b_methods, diagnostic=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.rot=rotations;self.fp_reference=fp_reference;self.b_methods=tuple(b_methods)
        self.conditions=tuple(['A_'+m for m in METHODS] if fp_reference else [])+tuple('B_'+m for m in b_methods)
        self.diagnostic=diagnostic;self.token=-1;self.seen=set();self.post_seen=set()
        self.maps={};self.indices={};self.hist={};self.zeros={};self.events=[]
        self.norm_max={m:0. for m in METHODS};self.norm_seen={m:0 for m in METHODS}
        self.direct_checks=[];self.postwrite_checks=0;self.coverage=[]
        self.edges=histogram_edges()
        self.tedges={k:torch.tensor(v,device=rotations.device,dtype=torch.float64) for k,v in self.edges.items()}

    def start(self,sample,positions,prompt):
        assert not self.maps
        self.sample=sample;self.positions=tuple(positions);self.position_to_event={p:i for i,p in enumerate(self.positions)}
        assert len(self.position_to_event)==len(self.positions)
        self.prompt=prompt;self.seen=set();self.post_seen=set()
        self.tmp=self.out/'.staging'/sample;self.tmp.mkdir(parents=True,exist_ok=False)
        n=len(positions)
        for cond in self.conditions:
            for layer in LAYERS:
                key=(cond,layer)
                specs=[('metrics',np.float64,(n,32,128,len(METRICS))),
                       ('argmax',np.uint8,(n,32,128)),
                       ('key_absmax',np.float32,(n,32,128)),
                       ('key_out3',np.uint16,(n,32,128))]
                if cond.startswith('B_'):
                    specs += [('cache_key_absmax',np.float32,(n,32,128)),('cache_key_out3',np.uint16,(n,32,128))]
                self.maps[key]={}
                for name,dtype,shape in specs:
                    path=self.tmp/f'{cond}_L{layer:02d}_{name}.npy'
                    self.maps[key][name]=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
                self.hist.setdefault(key,{k:np.zeros(len(v)+1,dtype=np.int64) for k,v in self.edges.items()})
                self.zeros.setdefault(key,0)

    def _store_key(self,state,rms):
        x=state[0]
        maximum=x.abs().amax(-1).cpu().numpy()
        counts=(x.double().abs()>3*rms[...,None,:]).sum(-1).cpu().numpy().astype(np.uint16)
        assert maximum.shape==counts.shape==(32,128)
        return maximum,counts

    def observe(self,layer,state):
        pos=self.token-self.prompt
        if pos not in self.position_to_event:return
        key=(self.sample,pos,layer)
        assert key not in self.seen,('Duplicate sample/token/layer',key)
        self.seen.add(key);idx=self.position_to_event[pos]
        snapshot=state.detach().clone()
        assert snapshot.data_ptr()!=state.data_ptr()
        assert snapshot.dtype==torch.float32
        sources=[]
        if self.fp_reference:
            reference=snapshot[0:1]
            native_norm=reference.double().square().sum(-2)
            for method in METHODS:
                s=self.rot.to_basis(method,reference,layer)
                error=float(((s.double().square().sum(-2)-native_norm).abs()/(native_norm+EPS)).max())
                self.norm_max[method]=max(self.norm_max[method],error);self.norm_seen[method]+=1
                assert error<2e-5,(method,layer,error)
                sources.append(('A_'+method,s))
        offset=int(self.fp_reference)
        for row,method in enumerate(self.b_methods,offset):sources.append(('B_'+method,snapshot[row:row+1]))
        for cond,s in sources:
            metrics=group_metrics(s)[0]
            assert metrics.shape==(32,128,len(METRICS))
            # Full independent recomputation for every condition at the two endpoint layers.
            check_key=(cond,layer)
            if layer in (0,30) and not any(x['condition']==cond and x['model_layer_id']==layer for x in self.direct_checks):
                self.direct_checks.append(independent_metrics_check(s,metrics,layer,self.sample,pos,cond))
            target=self.maps[cond,layer]
            target['metrics'][idx]=metrics.cpu().numpy()
            target['argmax'][idx]=s[0].abs().argmax(dim=-2).cpu().numpy().astype(np.uint8)
            km,kc=self._store_key(s,metrics[...,3])
            assert int(kc.astype(np.int64).sum())==int(metrics[...,9].sum())
            target['key_absmax'][idx]=km;target['key_out3'][idx]=kc
            for kind,values in {'signed':s.flatten().double(),'absolute':s.flatten().abs().double(),
                                'peak_rms':metrics[...,4].flatten()}.items():
                buckets=torch.bucketize(values,self.tedges[kind],right=True)
                count=torch.bincount(buckets,minlength=len(self.edges[kind])+1)
                self.hist[cond,layer][kind]+=count.cpu().numpy()
            self.zeros[cond,layer]+=int((s==0).sum())
            self.events.append({'experiment':cond[0],'condition':cond,'sample_id':self.sample,
                                'token_idx':pos,'input_position_one_based':self.token+1,'model_layer_id':layer,
                                'head_count':32,'value_count':128,'key_group_size':128,
                                'source':'same_FP_reference_transformed_snapshot' if cond[0]=='A' else 'actual_updated_FP32_before_physical_INT8_write'})
        # No tensor in sources/snapshot is saved as a recurrent carrier.

    def observe_postwrite(self,layer,codes,scales):
        pos=self.token-self.prompt
        if pos not in self.position_to_event:return
        key=(self.sample,pos,layer)
        assert key in self.seen and key not in self.post_seen
        self.post_seen.add(key);idx=self.position_to_event[pos]
        from vendor.prior_core import decode
        cached=decode(codes,scales)  # Actual resident bank fields, not another encode of a clone.
        for row,method in enumerate(self.b_methods):
            s=cached[row:row+1];rms=s[0].double().square().mean(-2).sqrt()
            km,kc=self._store_key(s,rms)
            target=self.maps['B_'+method,layer]
            target['cache_key_absmax'][idx]=km;target['cache_key_out3'][idx]=kc
            self.postwrite_checks+=1

    def finish(self):
        expected={(self.sample,p,l) for p in self.positions for l in LAYERS}
        assert self.seen==expected and self.post_seen==expected
        for (cond,layer),data in list(self.maps.items()):
            exp=cond[0];method=cond[2:]
            dest=self.out/exp/'groups'/f'{self.sample}__{cond}__L{layer:02d}.npz'
            dest.parent.mkdir(parents=True,exist_ok=True)
            common={'experiment':exp,'condition':cond,'basis':method,'sample_id':self.sample,
                    'model_layer_id':layer,'token_idx':np.array(self.positions),
                    'input_position_one_based':np.array(self.positions)+self.prompt+1,
                    'head_idx':np.arange(32),'key_idx':np.arange(128),'value_idx':np.arange(128)}
            for m in data.values():m.flush()
            with dest.open('xb') as f:
                np.savez_compressed(f,metrics=data['metrics'],metric_names=np.array(METRICS),
                                    key_absmax_index=data['argmax'],state_dtype='float32',
                                    cache_dtype='none_FP_reference_observation' if exp=='A' else 'int8',
                                    scale_dtype='float32',group_axis='Key axis 2/-2',**common)
            kp=self.out/exp/'key_summaries'/f'{self.sample}__{cond}__L{layer:02d}.npz'
            kp.parent.mkdir(parents=True,exist_ok=True)
            extra={k:data[k] for k in ['cache_key_absmax','cache_key_out3'] if k in data}
            with kp.open('xb') as f:
                np.savez_compressed(f,key_absmax_over_value=data['key_absmax'],key_out3_value_count=data['key_out3'],
                                    array_axes=np.array(['sampled_token','head','key']),value_elements_per_key=128,
                                    main_observation='same_FP_State_basis' if exp=='A' else 'pre_write_updated_FP32_State',
                                    cache_observation='actual_resident_codes_times_scales' if exp=='B' else 'not_applicable',
                                    **extra,**common)
            assert dest.is_file() and kp.is_file()
            for m in data.values():m._mmap.close()
            del self.maps[cond,layer]
        gc.collect()
        for p in self.tmp.iterdir():p.unlink()
        self.tmp.rmdir()
        row={'sample_id':self.sample,'positions':list(self.positions),'sampled_token_count':len(self.positions),
             'layers':list(LAYERS),'conditions':list(self.conditions),'layer_events':len(self.seen),
             'groups_per_condition':len(self.seen)*32*128,'postwrite_layer_events':len(self.post_seen)}
        self.coverage.append(row)
        return row

    def finish_all(self):
        for exp in sorted({c[0] for c in self.conditions}):
            result={f'{k}_edges':v for k,v in self.edges.items()}
            for (cond,layer),data in self.hist.items():
                if cond[0]!=exp:continue
                for kind,values in data.items():result[f'{cond}__L{layer:02d}__{kind}']=values
                result[f'{cond}__L{layer:02d}__zero_elements']=np.array(self.zeros[cond,layer],dtype=np.int64)
            with (self.out/exp/'histogram_counts.npz').open('xb') as f:np.savez_compressed(f,**result)
        dump(self.out/'COLLECTOR_VALIDATION.json',{'status':'PASS','coverage':self.coverage,'conditions':self.conditions,
             'max_paired_group_relative_squared_L2_error':self.norm_max,'paired_norm_layer_events':self.norm_seen,
             'independent_full_group_checks':self.direct_checks,'postwrite_actual_bank_condition_events':self.postwrite_checks,
             'quantiles':'exact saved FP64 group observations; never averages of token quantiles',
             'key_summary_axes':['sampled_token','head','key'],'state_snapshots_retained':False,
             'chunking':'per-sample/per-layer memmaps compressed separately; no full-experiment concatenation'})
        with (self.out/'collection_events.jsonl').open('x') as f:
            for row in self.events:f.write(json.dumps(row)+'\n')
