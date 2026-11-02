import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json, hashlib, subprocess, time
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
MODEL=Path('/data/zypan/modelscope_models/Qwen3.5-9B')
BACKUP=Path('/data/zypan/server55_zypan_backup_20261029_87e45424/restored')
FP=BACKUP/'worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    (ROOT/'inputs').mkdir(exist_ok=False)
    (ROOT/'provenance').mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    candidates=[];orig=[]
    for p in sorted(FP.glob('worker*/fp_state.jsonl')):
        orig.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
        for line_no,line in enumerate(p.open(),1):
            row=json.loads(line)
            if int(row['seed'])==1 and len(row.get('generated_token_ids',[]))>0:
                candidates.append((int(row['canonical_index']),str(p),line_no,row))
    candidates.sort(key=lambda x:(x[0],x[1],x[2]))
    seen=set();selected=[]
    for index,path,line_no,row in candidates:
        if row['problem_id'] in seen:continue
        seen.add(row['problem_id'])
        prompt=tok.apply_chat_template(row['raw_messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        prompt_ids=tok(prompt,add_special_tokens=False)['input_ids']
        assert len(prompt_ids)==int(row['input_tokens']), (row['problem_id'],len(prompt_ids),row['input_tokens'])
        continuation=list(map(int,row['generated_token_ids'][:2048]))
        n=len(continuation)
        positions=sorted({0,n-1,*range(15,n,16)})
        sample={'sample_id':row['problem_id'],'canonical_index':index,'seed':1,
                'source_file':path,'source_line':line_no,'original_FP_generation_tokens':len(row['generated_token_ids']),
                'historical_runtime':row.get('runtime_version'),'historical_git_commit':row.get('git_commit'),
                'historical_input_ids_hash':row.get('input_ids_hash'),
                'prompt_ids':prompt_ids,'prompt_tokens':len(prompt_ids),
                'continuation_ids':continuation,'continuation_tokens':n,
                'sampling_positions_zero_based_continuation':positions,
                'sampling_positions_one_based_continuation':[x+1 for x in positions],
                'token_stream_sha256':hashlib.sha256(json.dumps(prompt_ids+continuation,separators=(',',':')).encode()).hexdigest(),
                'selection_rule':'first4 unique seed1 by original canonical_index, no outcomes or magnitudes inspected',
                'prompt_processing':'every prompt token processed sequentially, no chunk prefill',
                'statistics_coverage':'continuation only; first, each16th, last; all24GDN/all32heads/all128values/all128keys'}
        write(ROOT/'inputs'/f'{row["problem_id"]}.json',sample)
        selected.append({k:v for k,v in sample.items() if k not in ['prompt_ids','continuation_ids']})
        if len(selected)==4:break
    assert selected
    write(ROOT/'inputs/SELECTION.json',{'candidates':len(candidates),'source_files':orig,'selected':selected,
          'requested_samples':4,'selected_samples':len(selected),'status':'PASS' if len(selected)==4 else 'PARTIAL'})
    print('INPUTS_SELECTED',json.dumps([{'id':s['sample_id'],'prompt':s['prompt_tokens'],'continuation':s['continuation_tokens'],'sampled':len(s['sampling_positions_zero_based_continuation'])} for s in selected]),flush=True)
    identity=[]
    for p in sorted(MODEL.iterdir()):
        if p.is_file():
            st=p.stat();h=sha(p);after=p.stat()
            assert (st.st_size,st.st_mtime_ns,st.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
            identity.append({'path':str(p),'bytes':st.st_size,'sha256':h,'inode':st.st_ino,'device':st.st_dev})
            print('MODEL_SHA',p.name,flush=True)
    write(ROOT/'provenance/MODEL_TOKENIZER_SHA256.json',identity)
    repo='/data/zypan/repos/GDN-quantization';env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    git={}
    for name,args in [('branch',['branch','--show-current']),('commit',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
        p=subprocess.run(['git','-C',repo]+args,capture_output=True,text=True,env=env,check=True);git[name]=p.stdout
    git['repo']=repo
    write(ROOT/'provenance/GIT_BEFORE.json',git)
    write(ROOT/'operations/INPUT_PREPARATION_COMPLETE.json',{'status':'PASS','selected_samples':len(selected),'model_files':len(identity),'cuda_initialized':False})

if __name__=='__main__':main()
