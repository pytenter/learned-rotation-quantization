"""Physical INT8 B carriers; optional independent one-row FP reference for A."""
import types
import torch
from vendor.prior_core import LAYERS, encode, decode

class PhysicalStateBank:
    def __init__(self, fp_reference, b_methods, observer=None):
        self.fp_reference=bool(fp_reference);self.b_methods=tuple(b_methods)
        self.rows=int(self.fp_reference)+len(self.b_methods)
        self.fp={};self.codes={};self.scales={};self.read_ptrs={}
        self.read_counts={l:0 for l in LAYERS};self.write_counts={l:0 for l in LAYERS}
        self.observer=observer;self.token=-1;self.read_checks=0

    def __getitem__(self,layer):
        if layer not in self.codes:return None
        b=decode(self.codes[layer],self.scales[layer])
        value=torch.cat((self.fp[layer],b),0) if self.fp_reference else b
        self.read_ptrs[layer]=(self.token,value.data_ptr())
        self.read_counts[layer]+=1
        return value

    def __setitem__(self,layer,state):
        assert layer in LAYERS and state.dtype==torch.float32 and state.shape==(self.rows,32,128,128)
        if self.observer is not None:self.observer.observe(layer,state)
        if self.fp_reference:
            self.fp[layer]=state[0:1].detach().clone()
            assert self.fp[layer].untyped_storage().nbytes()==32*128*128*4
        self.codes[layer],self.scales[layer]=encode(state[int(self.fp_reference):])
        assert self.codes[layer].dtype==torch.int8 and self.scales[layer].dtype==torch.float32
        assert self.scales[layer].shape==(len(self.b_methods),32,1,128)
        self.write_counts[layer]+=1
        if self.observer is not None:self.observer.observe_postwrite(layer,self.codes[layer],self.scales[layer])

    def check_consumed(self,layer,state):
        if self.token==0:assert state is None
        else:
            assert self.read_ptrs[layer]==(self.token,state.data_ptr())
            assert state.dtype==torch.float32 and state.shape==(self.rows,32,128,128)
            self.read_checks+=1

    def audit(self,cache=None):
        assert set(self.codes)==set(self.scales)==set(LAYERS)
        assert set(self.fp)==(set(LAYERS) if self.fp_reference else set())
        def description(t):
            return {'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),
                    'bytes':t.numel()*t.element_size(),'storage_bytes':t.untyped_storage().nbytes()}
        per_layer={}
        for l in LAYERS:
            if cache is not None:
                slot=cache.layers[l].recurrent_states
                assert isinstance(slot,dict) and list(dict.values(slot))==[None]
            per_layer[str(l)]={'codes':description(self.codes[l]),'scales':description(self.scales[l]),
                               'writes':self.write_counts[l],'reads':self.read_counts[l],
                               'cache_recurrent_slot':'Python proxy containing only {0:None}',
                               'FP_reference':description(self.fp[l]) if self.fp_reference else None}
        return {'B_methods':self.b_methods,'rows':self.rows,'FP_reference_present':self.fp_reference,
                'per_layer':per_layer,'core_read_pointer_checks':self.read_checks,
                'write_counts':self.write_counts,'read_counts':self.read_counts,
                'persistent_B_dense_State':False,'B_codes_bytes':sum(t.numel()*t.element_size() for t in self.codes.values()),
                'B_scales_bytes':sum(t.numel()*t.element_size() for t in self.scales.values()),
                'FP_reference_bytes':sum(t.numel()*t.element_size() for t in self.fp.values())}

def make_cache(config,bank):
    from transformers import DynamicCache
    cache=DynamicCache(config=config)
    class Slot(dict):
        def __init__(self,layer):super().__init__({0:None});self.layer=layer
        def __getitem__(self,index):
            assert index==0
            return bank[self.layer]
    for l in LAYERS:cache.layers[l].recurrent_states=Slot(l)
    def update(self,state,layer_idx,state_idx=0,**kwargs):
        assert state_idx==0
        bank[layer_idx]=state
        self.layers[layer_idx].is_recurrent_states_initialized[0]=True
        return None
    cache.update_recurrent_state=types.MethodType(update,cache)
    return cache
