import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,csv,json,shutil,hashlib
from pathlib import Path
import numpy as np
from plot_layout_v2 import distribution_fig,peak_fig,LAYERS

root=Path(sys.argv[1]);old=root/'visual_qa_initial_layout'
(root/'visual_qa').rename(old)
(root/'figures').rename(old/'figures')
with (root/'analysis/per_layer_summary.csv').open() as f:rows=list(csv.DictReader(f))
summ={}
for r in rows:
    for k,v in list(r.items()):
        try:r[k]=float(v)
        except ValueError:pass
    summ[r['condition'],int(r['model_layer_id'])]=r
selection=json.loads((root/'inputs/SELECTION.json').read_text())['selected']
for exp in ('A','B'):
    out=root/'figures'/exp;out.mkdir(parents=True,exist_ok=False)
    with np.load(root/'formal'/exp/'histogram_counts.npz') as z:hist={k:z[k].copy() for k in z.files}
    for layer in LAYERS:
        distribution_fig(exp,layer,summ,hist,selection,out)
        peak_fig(exp,layer,selection[0]['sample_id'],selection[0]['prompt_tokens'],root/'formal',out)
        prior=old/'figures'/exp/'figure_data'/f'L{layer:02d}_H0V0_exact_plot_matrices.npz'
        current=out/'figure_data'/prior.name
        with np.load(prior) as a,np.load(current) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        print('SAME_DATA_LAYOUT_REPAIR',exp,layer,flush=True)
    shutil.copyfile(old/'figures'/exp/'cross_layer_summary.png',out/'cross_layer_summary.png')
with (root/'analysis/PLOT_LAYOUT_REPAIR.json').open('x') as f:
    json.dump({'status':'PASS_IDENTICAL_PLOT_DATA','all48_peak_matrices_and_coordinates_byte_values_equal':True,
               'changed':'axes layout, dedicated external colorbar, common histogram ylim after all conditions, readable symlog ticks',
               'quantile_tables_and_raw_observations_changed':False,'prior_figures_preserved':str(old/'figures')},f,indent=2)
print('ALL146_PLOTS_REPAIRED_SAME_DATA',flush=True)
