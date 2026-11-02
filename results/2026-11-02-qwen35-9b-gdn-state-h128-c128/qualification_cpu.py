import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import torch
from experiment_core import *
from run_experiment import synthetic_gate
from vendor.canonical_h128 import hadamard,normalize_qk

def main():
    torch.set_num_threads(8)
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'metadata.json';dump(p,{'set_field':{3,1,2}})
        assert json.loads(p.read_text())=={'set_field':[1,2,3]}
    h=hadamard(device='cpu')
    result=synthetic_gate(h)
    torch.manual_seed(32)
    bank=PhysicalStateBank();reference=None
    for t in range(12):
        bank.token=t
        query=torch.randn((3,1,32,128),dtype=torch.bfloat16)
        key=torch.randn_like(query);value=torch.randn_like(query)
        q=normalize_qk(query).float();k=normalize_qk(key).float()
        q=torch.cat((q[:2],q[2:]@h));k=torch.cat((k[:2],k[2:]@h))
        g=-torch.rand((3,1,32));beta=torch.rand_like(g)
        before=bank[0];bank.check_consumed(0,before)
        a,s=Q.torch_recurrent_gated_delta_rule(q,k,value,g,beta,before,True,False)
        b,r=Q.torch_recurrent_gated_delta_rule(q,k,value,g,beta,reference,True,False)
        assert torch.equal(a,b) and torch.equal(s,r)
        bank[0]=s
        code,scale=encode(r[1:]);reference=torch.cat((r[:1].clone(),decode(code,scale)))
        assert torch.equal(bank.codes[0],code) and torch.equal(bank.scales[0],scale)
        s.fill_(999.)
        assert torch.equal(bank.fp[0],reference[:1])
    result['12step_physical_INT8_vs_dense_QDQ_reference']='PASS_BYTE_IDENTICAL'
    result['Qwen_module']=Q.__file__
    result['cuda_initialized']=torch.cuda.is_initialized()
    assert not result['cuda_initialized']
    dump(Path(__file__).resolve().parent/'gates/CPU_GATE_RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
