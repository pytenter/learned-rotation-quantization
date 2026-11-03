from pathlib import Path
import json
import os
import numpy as np
import torch
from collector import Collector
from state_bank import PhysicalStateBank
from rotation_adapter import Rotations, METHODS
from vendor.prior_core import LAYERS, decode, dump

ROOT=Path(__file__).resolve().parent

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(8);torch.manual_seed(20261103)
    rotations=Rotations(ROOT,'cpu');cfg=json.loads((ROOT/'CONFIG.json').read_text())
    results=[]
    for plan in cfg['batch_plan']:
        out=ROOT/'gates/cpu_collector'/plan['name']
        collector=Collector(out,rotations,plan['fp_reference'],plan['B_methods'],True)
        collector.start('synthetic_layer_mapping',[0,1],0)
        bank=PhysicalStateBank(plan['fp_reference'],plan['B_methods'],collector)
        dense_control={};expected={}
        for token in range(2):
            collector.token=bank.token=token
            for layer in LAYERS:
                scale=1.+layer/3
                previous=bank[layer]
                bank.check_consumed(layer,previous)
                if token:assert torch.equal(previous,dense_control[layer])
                state=torch.randn((plan['batch_rows'],32,128,128))*scale*.03
                if previous is not None:state+=previous*.9
                state[0,0,:,0]=0
                state[0,1,:,3]=1
                state[:,2,7,23]=57*scale
                snapshot=state.clone()
                bank[layer]=state
                dense_b=decode(bank.codes[layer],bank.scales[layer])
                dense_control[layer]=torch.cat((bank.fp[layer],dense_b),0) if plan['fp_reference'] else dense_b
                if layer in (0,30):
                    sources={}
                    if plan['fp_reference']:
                        sources.update({'A_'+m:rotations.to_basis(m,snapshot[0:1],layer) for m in METHODS})
                    for row,method in enumerate(plan['B_methods'],int(plan['fp_reference'])):sources['B_'+method]=snapshot[row:row+1]
                    for condition,s in sources.items():
                        x=s[0].numpy().copy();xd=x.astype(np.float64)
                        rms=np.sqrt(np.mean(xd*xd,axis=1))
                        expected[condition,layer,token]={
                            'key_absmax_over_value':np.abs(x).max(axis=2),
                            'key_out3_value_count':(np.abs(xd)>3*rms[:,None,:]).sum(axis=2),
                            'argmax':np.abs(x).argmax(axis=1)}
                state.zero_()
                if plan['fp_reference']:assert torch.equal(bank.fp[layer],snapshot[0:1])
                assert not torch.equal(dense_b,torch.zeros_like(dense_b))
        coverage=collector.finish();collector.finish_all()
        for (condition,layer,token),values in expected.items():
            exp=condition[0]
            kp=out/exp/'key_summaries'/f'synthetic_layer_mapping__{condition}__L{layer:02d}.npz'
            gp=out/exp/'groups'/f'synthetic_layer_mapping__{condition}__L{layer:02d}.npz'
            with np.load(kp) as k,np.load(gp) as g:
                assert np.array_equal(k['key_absmax_over_value'][token],values['key_absmax_over_value'])
                assert np.array_equal(k['key_out3_value_count'][token],values['key_out3_value_count'])
                assert np.array_equal(g['key_absmax_index'][token],values['argmax'])
                assert int(k['key_out3_value_count'][token].sum())==int(g['metrics'][token,...,9].sum())
        audit=bank.audit()
        assert all(x==2 for x in bank.write_counts.values()) and all(x==1 for x in bank.read_counts.values())
        assert bank.read_checks==24
        results.append({'phase':plan['name'],'coverage':coverage,'bank':audit,
                        'key_summary_and_argmax_full_endpoint_numpy':'EXACT',
                        'post_observation_mutation_snapshot_isolation':'PASS',
                        'persistent_physical_reads_equal_independent_dense_QDQ_control':'EXACT'})
    dump(ROOT/'gates/CPU_COLLECTOR_BANK_GATE.json',{'status':'PASS','plans':results,'cuda_initialized':torch.cuda.is_initialized()})
    print('CPU_COLLECTOR_BANK_GATE_PASS',flush=True)

if __name__=='__main__':main()
