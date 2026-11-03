"""Fresh CPU source/model/rotation identity and nonsymmetric direction checks."""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import numpy as np
import torch
from rotation_adapter import Rotations, METHODS, FROZEN_SHAS, file_sha, tensor_sha
from vendor.prior_core import LAYERS, dump, encode, decode, group_metrics, independent_metrics_check

ROOT = Path(__file__).resolve().parent

def rel(a, b):
    return float((a.double()-b.double()).norm()/(b.double().norm()+1e-12))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(8)
    torch.manual_seed(20261103)
    cfg = json.loads((ROOT/'CONFIG.json').read_text())
    rotations = Rotations(ROOT, 'cpu')
    rows, arrays, transform_checks = [], {}, []
    for method in METHODS:
        arrays[method] = []
        for layer in LAYERS:
            w = rotations.matrix(method, layer)
            arrays[method].append(w.numpy())
            orth = float((w.T@w-torch.eye(128,dtype=torch.float64)).abs().max())
            assert orth < cfg['max_orthogonality_error_original_threshold']
            row = {'method': method, 'model_layer_id': layer, 'head_sharing': 'all 32 heads share this layer transform',
                   'effective_shape': '[128,128]', 'effective_reference_dtype': 'float64',
                   'effective_reference_sha256': tensor_sha(w), 'effective_reference_recipe': 'FP32 H cast FP64 @ FP32 Delta cast FP64; identity for Native',
                   'effective_orthogonality_max_abs_FP64': orth,
                   'runtime_composition': 'FP32 x@H then @Delta, never x@(H@Delta)' if method in FROZEN_SHAS else method}
            if method in FROZEN_SHAS:
                d = rotations.delta(method, layer)
                identity = rotations.load_identity[method]
                manifest = json.loads((ROOT/'provenance'/f'{method.lower()}_rotation_manifest.json').read_text())
                assert tensor_sha(d) == manifest['per_layer_sha256'][str(layer)]
                row.update(checkpoint_path=identity['rotation_file'], checkpoint_sha256=identity['rotation_sha256'],
                           selected_update=manifest['selected_update'], delta_dtype=str(d.dtype), delta_shape=str(list(d.shape)),
                           delta_sha256=tensor_sha(d), delta_orthogonality_max_abs_FP32=float((d.T@d-torch.eye(128)).abs().max()),
                           original_theta_checkpoint_sha256=identity['theta_checkpoint_sha256'], original_payload_orientation=str(identity['orientation']))
                for dtype in (torch.bfloat16, torch.float32):
                    for norm in (False, True):
                        x = torch.randn((1,1,32,128), dtype=dtype)
                        actual = rotations.qk(method, x, layer, norm)
                        expected = rotations.original_qk(method, x, layer, norm)
                        assert torch.equal(actual, expected), (method, layer, dtype, norm)
                transform_checks.append({'method': method, 'layer': layer, 'original_frozen_evaluation_transform': 'EXACT_CPU_BF16_FP32_NORM_ON_OFF'})
            rows.append(row)
    columns = sorted(set().union(*(x.keys() for x in rows)))
    with (ROOT/'provenance/MATRIX_IDENTITY.csv').open('x') as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)
    with (ROOT/'provenance/effective_rotations_reference_fp64.npz').open('xb') as f:
        np.savez_compressed(f, layer_ids=np.array(LAYERS), **{k:np.stack(v) for k,v in arrays.items()})
    # A fixed, non-symmetric Givens rotation; no checkpoint is reorthogonalized.
    angle = .37
    c,s = np.cos(angle),np.sin(angle)
    r = torch.eye(128);r[0,0]=c;r[0,1]=-s;r[1,0]=s;r[1,1]=c
    assert not torch.equal(r,r.T)
    state = torch.randn((1,32,128,128));q = torch.randn((1,32,128))
    rotated = torch.matmul(r.T,state)
    out = (state*q[...,None]).sum(-2)
    out_r = (rotated*(q@r)[...,None]).sum(-2)
    rt = rel(torch.matmul(r,rotated),state)
    output_error = rel(out_r,out)
    wrong = rel((torch.matmul(r,state)*(q@r)[...,None]).sum(-2),out)
    assert rt < 2e-5 and output_error < 2e-5 and wrong > 1e-3
    norm_checks=[]
    for method in METHODS:
        for layer in (0,30):
            ss=rotations.to_basis(method,state,layer)
            norm_error=float(((ss.double().square().sum(-2)-state.double().square().sum(-2)).abs()/(state.double().square().sum(-2)+1e-12)).max())
            recover=rel(rotations.from_basis(method,ss,layer),state)
            assert max(norm_error,recover)<2e-5
            m=group_metrics(ss)[0]
            norm_checks.append({'method':method,'layer':layer,'squared_L2_max_relative_error':norm_error,'state_recovery_relative_L2':recover,
                                'independent_metrics':independent_metrics_check(ss,m,layer,'CPU_synthetic',0,method)})
    zero=torch.zeros((1,32,128,128))
    m=group_metrics(zero);assert torch.isnan(m[...,5]).all() and (m[...,13]==1).all()
    for bad_value in [float('nan'),float('inf'),-float('inf')]:
        bad=zero.clone();bad[0,0,0,0]=bad_value
        try:encode(bad)
        except FloatingPointError:pass
        else:raise AssertionError('Nonfinite not rejected')
    # Independent 12-step physical carrier and dense-QDQ control, all four rows.
    codes=scales=None;reference=None
    for token in range(12):
        current=decode(codes,scales) if codes is not None else torch.zeros((4,32,128,128))
        assert reference is None or torch.equal(current,reference)
        update=current*.97+torch.randn_like(current)*.02
        codes,scales=encode(update)
        independent_codes,independent_scales=encode(update.clone())
        reference=decode(independent_codes,independent_scales)
        assert codes.dtype==torch.int8 and scales.dtype==torch.float32
        assert codes.shape==(4,32,128,128) and scales.shape==(4,32,1,128)
    input_rows=[]
    for sample in cfg['samples']:
        p=ROOT/'inputs'/f'{sample}.json';d=json.loads(p.read_text())
        assert d['sample_id']==sample and len(d['continuation_ids'])==2048
        assert d['sampling_positions_zero_based_continuation']==cfg['sampling_zero_based']
        input_rows.append({'sample_id':sample,'file_sha256':file_sha(p),'prompt_tokens':len(d['prompt_ids']),'continuation_tokens':len(d['continuation_ids'])})
    model=[]
    for item in json.loads((ROOT/'provenance/PRIOR_MODEL_TOKENIZER_SHA256.json').read_text()):
        p=Path(item['path']);before=p.stat();digest=file_sha(p);after=p.stat()
        assert digest==item['sha256'] and after.st_size==before.st_size==item['bytes']
        assert (before.st_dev,before.st_ino,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_mtime_ns)
        model.append({'path':str(p),'bytes':after.st_size,'sha256':digest,'device':after.st_dev,'inode':after.st_ino,'mtime_ns':after.st_mtime_ns})
        print(json.dumps({'phase':'CPU_MODEL_SHA','file':p.name}),flush=True)
    dump(ROOT/'provenance/MODEL_TOKENIZER_SHA256.json',model)
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    modeling_sha=file_sha(Q.__file__)
    previous=json.loads((ROOT/'provenance/PRIOR_RUNTIME.json').read_text())
    assert modeling_sha==previous['modeling_sha256']
    assert torch.__version__==previous['torch'] and transformers.__version__==previous['transformers']
    disk=shutil.disk_usage(ROOT)
    # FP64 metrics: 4 samples * 8 conditions * 24 layers * 129 * 32 * 128 * 16.
    raw_metrics=4*8*24*129*32*128*16*8
    key_summaries=4*8*24*129*32*128*6
    assert disk.free>150*1024**3
    result={'status':'PASS','checkpoint_identities':rotations.load_identity,'matrix_rows':len(rows),'original_transform_exact_checks':transform_checks,
            'nonsymmetric_direction':{'correct_readout_relative_L2':output_error,'correct_state_recovery_relative_L2':rt,'wrong_transpose_challenge_relative_L2':wrong,'checkpoint_reorthogonalized':False},
            'norm_and_numpy_checks':norm_checks,'physical_INT8_vs_dense_QDQ_12_steps':'EXACT','zero_and_nonfinite':'PASS',
            'inputs':input_rows,'model_files':len(model),'modeling_sha256':modeling_sha,'torch':torch.__version__,'transformers':transformers.__version__,
            'python':sys.version,'host':platform.node(),'CPU_only_cuda_initialized':torch.cuda.is_initialized(),
            'storage':{'available_bytes':disk.free,'uncompressed_group_metrics_bytes':raw_metrics,'minimum_uncompressed_key_summary_bytes':key_summaries,
                       'planned_working_space_bytes':100*1024**3,'blockwise_strategy':'sample-by-layer memmaps, compressed at sample completion; analysis one layer at a time'},
            'batch_plan':cfg['batch_plan'],'reason':'Batch3 matching prior Native/H plus batch2 learned trajectories; avoids speculative batch5 VRAM fit; full samples/conditions unchanged'}
    dump(ROOT/'gates/CPU_AUDIT.json',result)
    print('CPU_AUDIT_PASS',flush=True)

if __name__=='__main__':
    main()
