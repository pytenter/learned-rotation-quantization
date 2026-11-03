"""Independent CPU-only tick-layout repair; retain initial PNG/QA, unchanged data/bins."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_FOUR_ROTATIONS_C6R_C7_20261103_V1')
assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
assert json.loads((ROOT/'operations/CPU_PIPELINE_RESULT.json').read_text())['status']=='PASS_COMPLETE_CPU_ANALYSIS_REPORT_PIPELINE'
assert not (ROOT/'operations/CPU_TICK_LAYOUT_CLAIM.json').exists()
assert not (ROOT/'figures_initial_layout').exists() and not (ROOT/'visual_qa_initial_layout').exists()
claim={'status':'CLAIMED_CPU_LAYOUT_ONLY','pid':os.getpid(),'original_data_recomputed':False,'new_GPU_work':False,
       'reason':'Actual visual inspection found overlapping symlog tick labels and clustered learned-method layer annotations'}
with (ROOT/'operations/CPU_TICK_LAYOUT_CLAIM.json').open('x') as f:json.dump(claim,f,indent=2)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

cpu_sources=json.loads((ROOT/'operations/CPU_PIPELINE_CLAIM.json').read_text())['sources']
gpu_sources=json.loads((ROOT/'provenance/SOURCE_MANIFEST.json').read_text())
for row in cpu_sources+gpu_sources:assert sha(ROOT/row['path'])==row['sha256']
protected=sorted((ROOT/'analysis').glob('*'))+sorted((ROOT/'figures').glob('*/figure_data/*.npz'))
before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
for exp in ['A','B']:
    dest=ROOT/'figures_initial_layout'/exp;dest.mkdir(parents=True)
    for p in [*(ROOT/'figures'/exp).glob('*_distribution.png'),ROOT/'figures'/exp/'outlier_error_relationship.png']:
        shutil.copy2(p,dest/p.name);assert sha(p)==sha(dest/p.name)
shutil.move(str(ROOT/'visual_qa'),str(ROOT/'visual_qa_initial_layout'))
sys.path.insert(0,str(ROOT/'analysis_cpu'))
import numpy as np
import matplotlib.ticker as ticker
import plots
from common import LAYERS,METHODS,histograms
source=(ROOT/'analysis_cpu/plots.py').read_text()
tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='distributions')
function=ast.get_source_segment(source,node)
insertion='''
    # Display-only locator/formatter change, all data arrays and x/y limits retained.
    for ax in axes[:2]:
        lo,hi=ax.get_xlim();limit=max(abs(lo),abs(hi))
        exponent=int(np.floor(np.log10(max(limit,1e-30))))
        powers=sorted(set([exponent-6,exponent-4,exponent-2,exponent]))
        positive=[10.**p for p in powers if 10.**p<=limit and 10.**p>=4*(2**-20)]
        ticks=([-v for v in positive[::-1]]+[0.]+positive) if lo<0 else ([0.]+positive)
        if len(ticks)<3:
            ticks=np.linspace(lo,hi,5)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value,pos: f'{value:.1e}' if value else '0'))
        else:
            def format_tick(value,pos):
                if value==0:return '0'
                power=int(round(np.log10(abs(value))))
                return ('$-10^{'+str(power)+'}$') if value<0 else ('$10^{'+str(power)+'}$')
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis='x',labelsize=9)
'''
needle="    fig.savefig(out/f'L{layer:02d}_distribution.png',bbox_inches='tight');plt.close(fig)"
assert function.count(needle)==1
function=function.replace("zero={int(hist[", "zero elements={int(hist[")
function=function.replace(needle,insertion+'\n'+needle)
namespace=dict(vars(plots));namespace.update(ticker=ticker,np=np)
exec(compile(function,'independent_tick_layout_v2','exec'),namespace)
fn=namespace['distributions']
rel_node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='relationship')
rel_function=ast.get_source_segment(source,rel_node)
assert rel_function.count('for layer in [0,14,30]:')==1
rel_function=rel_function.replace('for layer in [0,14,30]:','for layer in ([0,14,30] if method=="Native" else []):')
rel_function=rel_function.replace('labelled layers 0,14,30 were fixed in advance.','Native labels 0,14,30 fixed in advance; rotated-method points unlabelled to prevent overlap.')
exec(compile(rel_function,'independent_relationship_annotation_layout_v2','exec'),namespace)
relationship_fn=namespace['relationship']
rows=list(csv.DictReader((ROOT/'analysis/per_layer_summary.csv').open()))
for exp in ['A','B']:
    hist=histograms(exp)
    summ={}
    for row in rows:
        if row['experiment']!=exp:continue
        values={k:float(v) for k,v in row.items() if k in ['overall_absmax','group_observations','element_observations','peak_rms_median','out3_element_fraction','qdq_rel_l2_median']}
        summ[row['method'],int(row['model_layer_id'])]=values
    for layer in LAYERS:
        fn(exp,layer,summ,hist,ROOT/'figures'/exp)
        print('TICK_LAYOUT_V2',exp,layer,flush=True)
    relationship_fn(exp,summ,ROOT/'figures'/exp)
for rel,digest in before.items():assert sha(ROOT/rel)==digest
for row in cpu_sources+gpu_sources:assert sha(ROOT/row['path'])==row['sha256']
PY='/data/zypan/envs/qwen-gdn-group64-historical-port-v1/bin/python'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
     'MPLCONFIGDIR':str(ROOT/'.cache/matplotlib'),'OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'8'}
with (ROOT/'cpu_bitmap_tick_layout.log').open('x') as log:
    child=subprocess.Popen([PY,'-B',str(ROOT/'analysis_cpu/bitmap_qa.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    code=child.wait()
assert code==0
assert json.loads((ROOT/'visual_qa/BITMAP_CHECKS.json').read_text())['status']=='PASS'
result={'status':'PASS_CPU_TICK_LAYOUT_V2','repainted_distribution_PNGs':48,'original_distribution_PNGs_retained':48,
        'relationship_annotation_layout_PNGs':2,'original_relationship_PNGs_retained':2,
        'relationship_change':'Same points and axes; only Native preset 0/14/30 annotations retained, updated caption',
        'original_visual_QA_retained':'visual_qa_initial_layout','new_bitmap_real_wait_integer':code,
        'all_analysis_and_plot_data_protected_hashes_unchanged':before,'original_9_CPU_pipeline_sources_unchanged':True,
        'frozen_GPU_sources_unchanged':True,'raw_data_modified':False,'new_GPU_work':False,
        'source_sha256':sha(__file__)}
with (ROOT/'operations/CPU_TICK_LAYOUT_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print('CPU_TICK_LAYOUT_REPAIR_PASS',flush=True)
