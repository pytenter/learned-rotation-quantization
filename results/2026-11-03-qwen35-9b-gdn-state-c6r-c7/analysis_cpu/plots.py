import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from common import LAYERS,METHODS,SAMPLES,data_path

COLORS={'Native':'#4C78A8','Hadamard':'#F58518','C6R':'#54A24B','C7':'#B279A2'}
MARKERS={'Native':'o','Hadamard':'s','C6R':'^','C7':'D'}
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':170})

def distributions(exp,layer,summ,hist,out):
    fig,axes=plt.subplots(1,3,figsize=(17,5.2),layout='constrained')
    for ax,kind in zip(axes,['signed','absolute','peak_rms']):
        edges=hist[kind+'_edges'];top=1
        for method in METHODS:
            cond=f'{exp}_{method}';counts=hist[f'{cond}__L{layer:02d}__{kind}']
            expected=summ[method,layer]['group_observations'] if kind=='peak_rms' else summ[method,layer]['element_observations']
            assert int(counts.sum())==expected
            top=max(top,int(counts[1:-1].max()))
            ax.stairs(counts[1:-1],edges,color=COLORS[method],label=method,linewidth=1.4)
        ax.set_yscale('log');ax.set_ylim(.7,top*1.6)
        ax.set_ylabel('Count per common bin (log y; zeros not raised)')
        if kind=='peak_rms':ax.set_xlim(0,12);ax.set_xlabel('C128 group absmax / (RMS + 1e-12)')
        else:
            maximum=max(summ[m,layer]['overall_absmax'] for m in METHODS)
            pos=hist['absolute_edges'];limit=pos[min(len(pos)-1,int(np.searchsorted(pos,maximum,side='right')))]
            limit=max(limit,2**-19)
            ax.set_xscale('symlog',linthresh=2**-20)
            ax.set_xlim((-limit,limit) if kind=='signed' else (0,limit))
            ax.set_xlabel('Signed State value (a.u.; symlog x)' if kind=='signed' else '|State| (a.u.; symlog x)')
        ax.legend(fontsize=9)
    detail=[]
    for m in METHODS:
        c=f'{exp}_{m}';a=hist[f'{c}__L{layer:02d}__absolute'];p=hist[f'{c}__L{layer:02d}__peak_rms']
        detail.append(f"{m}: zero={int(hist[f'{c}__L{layer:02d}__zero_elements']):,}; abs under/over={int(a[0])}/{int(a[-1])}; peak under/over={int(p[0])}/{int(p[-1])}")
    scope='A: paired transforms of one FP State' if exp=='A' else 'B: independent INT8 trajectories; updated FP32 before write'
    fig.suptitle(f'Qwen3.5-9B | original GDN layer {layer} | {scope}\n4 samples x 129 sampled positions; all 32 heads, 128 values and all 128 Keys\n'+' | '.join(detail[:2])+'\n'+' | '.join(detail[2:]),fontsize=9)
    fig.savefig(out/f'L{layer:02d}_distribution.png',bbox_inches='tight');plt.close(fig)

def key_maps(exp,layer,values,out):
    maxima={};fractions={};counts={}
    for method in METHODS:
        max_parts=[];count_parts=[]
        for sid in SAMPLES:
            with np.load(data_path(exp,method,layer,sid,'key_summaries')) as z:
                assert tuple(z['array_axes'])==('sampled_token','head','key')
                km=z['key_absmax_over_value'].copy();kc=z['key_out3_value_count'].copy()
            assert km.shape==kc.shape==(129,32,128) and km.dtype==np.float32 and kc.dtype==np.uint16
            assert np.isfinite(km).all() and int(kc.max())<=128
            max_parts.append(km);count_parts.append(kc)
        km=np.stack(max_parts);kc=np.stack(count_parts)
        maxima[method]=km.max(axis=(0,1))
        counts[method]=kc.astype(np.int64).sum(axis=(0,1))
        fractions[method]=counts[method]/(4*129*128)
        assert np.array_equal(counts[method].sum(axis=1),values[method][...,9].sum(axis=(0,2)).astype(np.int64))
        assert float(maxima[method].max())==float(values[method][...,2].max())
    plotdata=out/'figure_data';plotdata.mkdir(exist_ok=True)
    with (plotdata/f'L{layer:02d}_all_heads_key_exact.npz').open('xb') as f:
        np.savez_compressed(f,model_layer_id=layer,experiment=exp,samples=np.array(SAMPLES),head_idx=np.arange(32),key_idx=np.arange(128),
            event_count_per_layer=516,value_elements_per_key_per_event=128,
            **{m+'_max_abs_over_samples_tokens_values':maxima[m] for m in METHODS},
            **{m+'_out3_value_count_sum':counts[m] for m in METHODS},
            **{m+'_out3_element_fraction':fractions[m] for m in METHODS})
    amax=max(float(maxima[m].max()) for m in METHODS);rate=max(float(fractions[m].max())*100 for m in METHODS)
    maxnorm=Normalize(0,max(amax,1e-12));ratenorm=Normalize(0,max(rate,1e-12))
    fig,axes=plt.subplots(4,2,figsize=(15,13),layout='constrained')
    for row,method in enumerate(METHODS):
        for col,(data,norm,cmap) in enumerate([(maxima[method],maxnorm,'magma'),(fractions[method]*100,ratenorm,'viridis')]):
            ax=axes[row,col];ax.grid(False)
            ax.imshow(data,origin='lower',aspect='auto',interpolation='nearest',norm=norm,cmap=cmap,extent=[-.5,127.5,-.5,31.5])
            ax.set_ylabel(method+' | head index');ax.set_xlabel('Key coordinate in '+method+' basis')
            ax.set_title('Max |State| over all 4 samples, 129 positions and 128 values' if col==0 else 'Pooled |x| > 3 x own-group RMS element percentage')
    fig.colorbar(ScalarMappable(norm=maxnorm,cmap='magma'),ax=list(axes[:,0]),shrink=.9,label='State absolute value (a.u.; common full range)')
    fig.colorbar(ScalarMappable(norm=ratenorm,cmap='viridis'),ax=list(axes[:,1]),shrink=.9,label='Percent; count / (516 events x128 values), common full range')
    scope='A: same FP State, transformed bases' if exp=='A' else 'B: separate INT8 histories; updated FP32 before write'
    fig.suptitle(f'Qwen3.5-9B | GDN layer {layer} | {scope}\nAll 32 heads shown. Same Key number across bases does NOT denote the same original semantic feature.',fontsize=11)
    fig.savefig(out/f'L{layer:02d}_key_outlier_locations.png',bbox_inches='tight');plt.close(fig)

def cross_layer(exp,summ,out):
    fields=[('peak_rms_median','Median group peak / RMS',1),('out3_element_fraction','Elements >3 x group RMS (%)',100),
            ('qdq_rel_l2_median','Median local INT8 QDQ relative L2 (%)',100)]
    fig,axes=plt.subplots(1,3,figsize=(17,5.2),layout='constrained')
    for ax,(field,label,scale) in zip(axes,fields):
        for method in METHODS:
            y=[summ[method,l][field]*scale for l in LAYERS]
            ax.plot(LAYERS,y,marker=MARKERS[method],color=COLORS[method],label=method,markersize=3,linewidth=1.2)
        ax.set_xlabel('Original model layer ID (24 GDN layers only)');ax.set_ylabel(label)
        ax.set_xticks(LAYERS);ax.tick_params(axis='x',rotation=55,labelsize=8);ax.set_ylim(bottom=0);ax.legend(fontsize=9)
    scope='A: strict paired observations of the same FP State' if exp=='A' else 'B: same token stream, independent physical INT8 histories'
    fig.suptitle(f'Qwen3.5-9B | experiment {exp}: {scope}\n4 seed1 samples x2048 continuation tokens; 516 sample-token events/layer; exact group quantiles; no accuracy evaluation',fontsize=11)
    fig.savefig(out/'cross_layer_summary.png',bbox_inches='tight');plt.close(fig)

def relationship(exp,summ,out):
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for method in METHODS:
        q=np.array([summ[method,l]['qdq_rel_l2_median'] for l in LAYERS])*100
        for ax,field,label,scale in [(axes[0],'peak_rms_median','Median group peak / RMS',1),
                                    (axes[1],'out3_element_fraction','Pooled >3 x RMS element percentage',100)]:
            x=np.array([summ[method,l][field] for l in LAYERS])*scale
            ax.scatter(x,q,color=COLORS[method],marker=MARKERS[method],label=method,s=28,alpha=.85)
            for layer in [0,14,30]:
                i=LAYERS.index(layer);ax.annotate(f'L{layer}',(x[i],q[i]),xytext=(3,4),textcoords='offset points',fontsize=7)
            ax.set_xlabel(label);ax.set_ylabel('Median local INT8 QDQ relative L2 (%)');ax.legend(fontsize=8)
    fig.suptitle(f'Experiment {exp}: descriptive layer-summary relationship, not independent-sample significance\nOne dot = method/layer summary over the same 4 samples; labelled layers 0,14,30 were fixed in advance.',fontsize=10)
    fig.savefig(out/'outlier_error_relationship.png',bbox_inches='tight');plt.close(fig)
