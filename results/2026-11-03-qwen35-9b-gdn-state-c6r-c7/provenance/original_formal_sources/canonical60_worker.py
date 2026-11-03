"""One model-resident, single-GPU worker, directly invoking frozen generate_one."""
import argparse, fcntl, os, shutil, time, traceback
import canonical60_common as A

def signal_stop(arm,reason):
    try: A.save('logs/'+arm.lower()+'.stop.json',dict(reason=reason,utc=A.utc(),pid=os.getpid(),retry=False))
    except FileExistsError: pass
    except RuntimeError:
        if not (A.ROOT/'logs'/f'{arm.lower()}.stop.json').exists(): raise

def complete_existing(arm,job,g):
    p=A.final_path(arm,job);row=A.read(p);A.validate_raw(row,arm,job,g);p.chmod(0o444)
    c=A.completion_path(arm,job)
    if c.exists():
        r=A.read(c);A.require(r['file_sha256']==A.sha(p) and r['payload_sha256']==row['output_payload_sha256'],'completed output changed')
    else:
        A.save(c,dict(status='COMPLETE',canonical_file=str(p),file_sha256=A.sha(p),payload_sha256=row['output_payload_sha256'],problem_id=job['problem_id'],seed=job['seed'],condition=arm,reconstructed_metadata_only=True,utc=A.utc()),True)
    return row

def main(arm,gpu):
    A.verify_freeze(gold_free_worker=True)
    A.require(not (A.ROOT/'logs'/f'{arm.lower()}.stop.json').exists(),'affected condition is STOPPED')
    A.require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu),'physical GPU environment mismatch')
    lock=(A.ROOT/'logs'/f'{arm.lower()}_gpu{gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    n=A.nvml();s=n.snapshot(gpu)
    A.require(s['uuid']==A.UUIDS[gpu] and not s['processes'],'GPU busy or UUID mismatch before worker load')
    jobs=[j for j in A.samples() if j['gpu']==gpu];A.require(len(jobs)==15,'worker assignment cardinality')
    for j in jobs:
        A.require(not A.started_path(arm,j).exists() or A.final_path(arm,j).exists(),'post-forward incomplete sample; explicit user authorization required')
    A.save(f'logs/{arm.lower()}_gpu{gpu}_startup.json',dict(pid=os.getpid(),arm=arm,gpu=gpu,uuid=s['uuid'],MODEL_FORWARD_STARTED='NO',started_utc=A.utc()),True)
    g=A.generator()
    import torch
    A.require(torch.cuda.device_count()==1,'not independent single-GPU worker')
    cfg=A.read(A.ROOT/'configs/canonical60_protocol.json');gen=A.read(A.ROOT/'configs/canonical60_generation_config.json')
    A.require(g.runtime_version()==cfg['runtime'],'runtime version mismatch')
    flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    A.require(flags==gen['deterministic_runtime_flags'],'runtime inference flags mismatch')
    g.setup_seed(1);model,tok=g.load_model_and_tokenizer()
    for p in model.parameters(): p.requires_grad_(False)
    A.require(next(model.parameters()).dtype==torch.bfloat16,'model dtype drift')
    path,manifest=A.rotation_paths(arm)
    A.require(A.sha(path)==A.SHAS[arm],'rotation changed before load')
    bank,ident=g.load_frozen_rotation(path,manifest,torch.device('cuda:0'),arm)
    config_sha=A.sha(A.ROOT/'configs/canonical60_generation_config.json')
    model_sha=A.digest(A.read(A.ROOT/'provenance/canonical60_runtime_identity.json')['model_identity'])
    commit=A.git('rev-parse','HEAD');done=[]
    for j in jobs:
        if A.final_path(arm,j).exists():
            r=complete_existing(arm,j,g);done.append(dict(rank=j['rank'],tokens=r['generated_token_count'],existing_immutable=True));continue
        A.require(not (A.ROOT/'logs'/f'{arm.lower()}.stop.json').exists(),'peer failure: no next sample launch')
        A.require(shutil.disk_usage('/data').free>=cfg['disk_stop_threshold_bytes'],'disk below frozen 15GiB; no next sample')
        snapshot=n.snapshot(gpu)
        A.require(snapshot['uuid']==j['GPU_uuid'] and all(p['pid']==os.getpid() for p in snapshot['processes']),'GPU UUID or foreign compute process changed')
        messages,ids=g.render_input(tok,j['problem'])
        A.require(g.ids_sha256(ids[0].tolist())==j['input_ids_hash'] and messages==j['raw_messages'],'prompt/input identity drift')
        del messages,ids
        # Conservative durable marker immediately BEFORE the first canonical call.
        # If a crash leaves this marker without final output, no automatic retry.
        A.save(A.started_path(arm,j),dict(MODEL_FORWARD_STARTED='YES_CONSERVATIVE_PRECALL_MARKER',pid=os.getpid(),condition=arm,rank=j['rank'],problem_id=j['problem_id'],seed=j['seed'],gpu=gpu,uuid=j['GPU_uuid'],utc=A.utc()),True)
        print('SAMPLE_STARTED',arm,j['problem_id'],j['seed'],'rank',j['rank'],'GPU',gpu,flush=True)
        item=dict(problem_id=j['problem_id'],problem_idx=j['problem_idx'],problem=j['problem'],answer=None)
        row=g.generate_one(model,tok,item,j['seed'],arm.lower(),str(gpu),gpu,j['rank'],bank,ident,config_sha)
        canonical_payload_sha=row['output_payload_sha256']
        if row['runtime_error'] or row['nonfinite']:
            A.save(f'logs/failed_outputs/{arm.lower()}_{j["problem_id"]}_seed{j["seed"]}.json',row,True)
            raise RuntimeError('post-forward runtime/numerical failure; preserved noncanonical failed output')
        row.update(condition=arm,sample_rank=j['rank'],assigned_gpu_index=gpu,physical_gpu_uuid=j['GPU_uuid'],model_identity_sha256=model_sha,runtime_config_sha256=config_sha,prompt_sha256=j['prompt_sha256'],num_generated_tokens=row['generated_token_count'],generation_source_sha256=A.FORMAL_SHA,canonical_generator_payload_sha256=canonical_payload_sha,adapter_premeasurement_commit=commit,gold_accessed=False)
        row['output_payload_sha256']=A.payload_sha(row)
        A.validate_raw(row,arm,j,g)
        # Validate the persisted temp payload too; publish by atomic rename only.
        p=A.final_path(arm,j);p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.validated.tmp')
        import json
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(row,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        A.validate_raw(A.read(tmp),arm,j,g)
        A.require(not p.exists(),'immutable final already exists: refusing overwrite')
        os.rename(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        p.chmod(0o444)
        complete_existing(arm,j,g)
        torch.cuda.synchronize()
        done.append(dict(rank=j['rank'],tokens=row['generated_token_count'],termination=row['finish_reason'],latency_s=row['latency_s'],allocated_end=torch.cuda.memory_allocated(),reserved_end=torch.cuda.memory_reserved()))
        print('SAMPLE_COMPLETE',arm,j['problem_id'],j['seed'],row['generated_token_count'],row['finish_reason'],flush=True)
        del row
    A.save(f'analysis/{arm.lower()}_gpu{gpu}_worker.json',dict(status='PASS',pid=os.getpid(),gpu=gpu,uuid=A.UUIDS[gpu],condition=arm,completed=15,requests=done,model_loaded_once=True,rotation_loaded_once=True,normal_exit=True,semantic_retries=0,completed_utc=A.utc()),True)
    print('WORKER_NORMAL_EXIT',arm,gpu,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('C6R','C7'),required=True);p.add_argument('--gpu',type=int,choices=range(4),required=True);args=p.parse_args()
    try: main(args.arm,args.gpu)
    except BaseException as e:
        signal_stop(args.arm,str(e))
        forward=any(A.read(p).get('pid')==os.getpid() for p in (A.ROOT/'logs/started'/args.arm.lower()).glob('*.json'))
        A.save(f'logs/{args.arm.lower()}_gpu{args.gpu}.error.json',dict(status='FAIL',pid=os.getpid(),error=str(e),traceback=traceback.format_exc(),MODEL_FORWARD_STARTED='YES' if forward else 'NO',automatic_retry=False,utc=A.utc()),True)
        raise
