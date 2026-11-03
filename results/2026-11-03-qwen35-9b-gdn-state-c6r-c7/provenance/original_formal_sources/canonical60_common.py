"""Canonical60 orchestration and identity helpers; no new model numerics."""
import hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

TASK = 'QWEN_GDN_C6R_C7_CANONICAL60_V1'
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1')
V2 = SOURCE/'experiments/QWEN_GDN_C6R_C7_4X3090_PERSISTENT_WORKER_READINESS_V2'
V1 = SOURCE/'experiments/QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1'
BASE = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
OLD = BASE/'experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1'
FORMAL = OLD/'scripts/run_qwen_c5_c6_formal.py'
POST = OLD/'scripts/postprocess_canonical60.py'
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
DATASET = Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl')
SCORER = Path('/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/shared/scoring/aime26_scorer_v4.py')
PY = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
PARENT = 'cf850cfc06f5384f4e7aa36018fcc07fee45374d'
BRANCH = 'exp/qwen-gdn-c6r-c7-canonical60-v1'
UUIDS = ['GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75']
SHAS = {'C6R':'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6','C7':'57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8'}
FORMAL_SHA = '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
SCORER_SHA = 'fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b'
DATASET_SHA = '0a744825990f027c6c4455b78776512964bfdbde1e39e8826ad188c6a0d74b9d'

def require(v, message):
    if not v: raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def payload_sha(row):
    d = dict(row); d.pop('output_payload_sha256',None)
    return digest(d)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def git(*args): return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def write_text(path, text, immutable=False):
    p = ROOT/path; p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.exists(), 'refuse overwrite '+str(p))
    t = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with t.open('x',encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    require(not p.exists(), 'refuse concurrent overwrite '+str(p))
    os.rename(t,p)
    fd = os.open(str(p.parent),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    if immutable: p.chmod(0o444)

def save(path,x,immutable=False):
    write_text(path,json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n',immutable)

def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def generator():
    require(sha(FORMAL)==FORMAL_SHA,'canonical generation source drift')
    for p in ('/data/zypan/transformers-qwen35/src','/data/zypan/experiments/qwen35_gdn_quant'):
        if p not in sys.path: sys.path.insert(0,p)
    os.environ['QWEN_TRANSFORMERS_SRC']='/data/zypan/transformers-qwen35/src'
    g=mod('canonical60_original_generator',FORMAL)
    for arm in SHAS: g.METHODS[arm.lower()]='QWEN_INT8_C128_KEY_'+arm
    require(g.MAX_NEW_TOKENS==81920,'canonical ceiling changed')
    return g

def nvml():
    # Reuse the exact byte-resolution parent helper; it imports no torch.
    return mod('canonical60_parent_lifecycle_helpers',V2/'scripts/persistent_v2_common.py').Nvml()

def rotation_paths(arm):
    return V1/'rotations'/arm/'frozen_rotation_fp32.pt',V1/'configs'/f'{arm.lower()}_rotation_manifest.json'

def manifest_entries(p):
    return [dict(sha256=l.split('  ',1)[0],path=l.split('  ',1)[1]) for l in Path(p).read_text().splitlines() if l.strip()]

def verify_manifest(p,skip_paths=()):
    rows=manifest_entries(p)
    for r in rows:
        if r['path'] not in skip_paths: require(sha(r['path'])==r['sha256'],'SHA drift '+r['path'])
    return rows

def verify_freeze(gold_free_worker=False):
    # Worker must not even open the separated gold manifest for hashing.
    # CPU prelaunch/scoring controller validates that file separately.
    skip=(str(ROOT/'manifests/canonical60_gold_manifest.json'),) if gold_free_worker else ()
    verify_manifest(ROOT/'manifests/canonical60_premeasurement_sha256.txt',skip)
    require(git('rev-parse','--abbrev-ref','HEAD')==BRANCH,'branch drift')
    require(not git('status','--porcelain'),'worktree not clean')
    require(read(ROOT/'provenance/canonical60_runtime_identity.json')['CANONICAL60_RUNTIME_IDENTITY_GATE']=='PASS','identity FAIL')

def samples(): return read(ROOT/'manifests/canonical60_sample_manifest.json')['samples']
def final_path(arm,job): return ROOT/'raw'/arm.lower()/f"{job['problem_id']}_seed{job['seed']}.json"
def completion_path(arm,job): return ROOT/'logs/completions'/arm.lower()/f"{job['problem_id']}_seed{job['seed']}.json"
def started_path(arm,job): return ROOT/'logs/started'/arm.lower()/f"{job['problem_id']}_seed{job['seed']}.json"

def validate_raw(row,arm,job,g=None):
    require(row.get('condition')==arm,'condition mismatch')
    for key in ('problem_id','seed','sample_rank','assigned_gpu_index','physical_gpu_uuid','input_ids_hash','prompt_sha256'):
        expected={'sample_rank':job['rank'],'assigned_gpu_index':job['gpu'],'physical_gpu_uuid':job['GPU_uuid']}.get(key,job.get(key))
        require(row.get(key)==expected,'raw identity '+key)
    require(row['problem']==job['problem'],'problem drift')
    require(row['raw_messages']==job['raw_messages'],'prompt messages drift')
    require(row['gold_answer'] is None and row['correct'] is None and row['extracted_answer'] is None,'generation gold/scoring exposure')
    require(row['rotation_sha256']==SHAS[arm],'rotation identity')
    require(row['runtime_config_sha256']==sha(ROOT/'configs/canonical60_generation_config.json'),'config identity')
    require(row['formal_config_sha256']==row['runtime_config_sha256'],'canonical supplied config SHA drift')
    require(row['model_identity_sha256']==digest(read(ROOT/'provenance/canonical60_runtime_identity.json')['model_identity']),'model identity')
    require(row['generation_source_sha256']==FORMAL_SHA,'generation source identity')
    require(row['runtime_version']==read(ROOT/'configs/canonical60_protocol.json')['runtime'],'runtime identity')
    require(row['canonical_index']==job['rank'] and int(row['gpu'])==job['gpu'] and row['worker_id']==job['gpu'],'canonical GPU/rank identity')
    require(row['output_payload_sha256']==payload_sha(row),'payload integrity')
    require(row['runtime_error'] is None and row['nonfinite'] is False,'post-forward runtime/numerical failure')
    ids=row['generated_token_ids'];n=len(ids)
    require(n>0 and all(isinstance(i,int) and not isinstance(i,bool) and i>=0 for i in ids),'invalid tokens')
    require(n==row['generated_token_count']==row['num_generated_tokens']==row['output_tokens'],'token count mismatch')
    require(row['finish_reason'] in ('eos','length') and row['max_new_tokens']==81920,'invalid termination')
    require(row['eos_seen']==(row['finish_reason']=='eos') and row['truncated']==(row['finish_reason']=='length'),'termination flags')
    require(row['finish_reason']!='length' or n==81920,'invalid ceiling')
    q=row['quantizer_audit'];require(q['calls']==n-1,'QDQ coverage count')
    if n>1: require(q['layers']==read(ROOT/'configs/canonical60_protocol.json')['GDN_layers'],'QDQ layer coverage')
    require(row['retry_count']==0 and row['direct_frozen_load'] is True and row['theta_reconstruction'] is False,'load/retry semantics')
    require(isinstance(row['response'],str) and row['response'],'empty canonical output')
    expected=dict(read(ROOT/'configs/canonical60_generation_config.json')['protocol'],sampling_seed=job['seed'])
    expected={k:expected[k] for k in row['effective_generation_config']}
    require(row['effective_generation_config']==expected,'effective sampler drift')
    if g is not None: require(g.is_valid_record(row,arm.lower()),'original canonical raw validator failed')
    return True

def protected_revalidate():
    source=read(V2/'provenance/source_artifact_revalidation.json')
    rows=source['files']
    for r in rows: require(sha(r['path'])==r['sha256'],'protected source drift '+r['path'])
    require(len(rows)==source['count']==3174,'parent protected manifest count')
    for arm in SHAS: require(sha(rotation_paths(arm)[0])==SHAS[arm],'frozen rotation drift')
    return dict(status='PASS',source=str(V2/'provenance/source_artifact_revalidation.json'),source_sha256=sha(V2/'provenance/source_artifact_revalidation.json'),count=len(rows),files=rows,checked_utc=utc())
