"""CPU-only exact group reductions and figures from THIS run, never old smoke."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,hashlib,math,gc,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent)
FORMAL=ROOT/'formal'
LAYERS=[i for i in range(32) if i%4!=3]
NAMES=('min','max','absmax','rms','peak_rms','kurtosis','scale','qdq_rel_l2',
       'boundary_count','out3_count','out6_count','squared_error','squared_norm',
       'zero_group','zero_variance','nonfinite_count')
IDX={n:i for i,n in enumerate(NAMES)}
QUANTILES=('absmax','rms','peak_rms','kurtosis','scale','qdq_rel_l2')
LABELS={'Native':'Native','KeyH128':'Key-side Hadamard H128'}
COLORS=['#2458a5','#db6625']
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':170})

def jd(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvout(p,rows):
    assert rows
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(m):
    x=m.reshape(-1,len(NAMES));n=x.shape[0]
    r={'group_observations':n,'element_observations':n*128,
       'zero_groups':int(x[:,IDX['zero_group']].sum()),
       'zero_variance_groups':int(x[:,IDX['zero_variance']].sum()),
       'nonfinite_elements':int(x[:,IDX['nonfinite_count']].sum()),
       'overall_absmax':float(np.max(x[:,IDX['absmax']])),
       'state_min':float(np.min(x[:,IDX['min']])),'state_max':float(np.max(x[:,IDX['max']]))}
    for name in QUANTILES:
        a=x[:,IDX[name]];a=a[np.isfinite(a)]
        q=np.quantile(a,[.5,.95,.99],method='linear') if a.size else [np.nan]*3
        for title,v in zip(['median','p95','p99'],q):r[f'{name}_{title}']=float(v)
        r[f'{name}_valid_groups']=len(a)
    for name in ('out3_count','out6_count','boundary_count'):
        r[name]=int(x[:,IDX[name]].sum());r[name.replace('_count','_element_fraction')]=r[name]/(n*128)
    r['pooled_qdq_relative_L2']=float(np.sqrt(x[:,IDX['squared_error']].sum())/(np.sqrt(x[:,IDX['squared_norm']].sum())+1e-12))
    return r

def metadata(exp,condition,layer):
    return {'experiment':exp,'condition':condition,'basis':'KeyH128' if 'H128' in condition else 'Native',
            'model_layer_id':layer,'source_state_dtype':'float32',
            'persistent_cache_dtype':'float32_reference' if exp=='A' else 'int8',
            'scale_dtype':'float32','key_group_size':128,
            'observation':'common_FP_state_pair' if exp=='A' else 'updated_state_immediately_before_INT8_writeback'}

def distribution_fig(exp,layer,summ,hist,selection,out):
    fig,axes=plt.subplots(1,3,figsize=(17,4.6))
    conds=[exp+'_Native',exp+'_KeyH128']
    for cond,color in zip(conds,COLORS):
        for ax,kind in zip(axes,['signed','absolute','peak_rms']):
            edge=hist[kind+'_edges'];counts=hist[f'{cond}__L{layer:02d}__{kind}']
            ax.stairs(counts[1:-1],edge,color=color,label=LABELS[cond.split('_',1)[1]]+' pre-writeback' if exp=='B' else LABELS[cond.split('_',1)[1]],linewidth=1.4)
            if exp=='B' and kind!='peak_rms':
                cache=hist[f'{cond}_CACHE__L{layer:02d}__{kind}']
                ax.stairs(cache[1:-1],edge,color=color,linestyle='--',linewidth=1.,label=LABELS[cond.split('_',1)[1]]+' cache decoded')
            assert int(counts.sum())==(summ[cond,layer]['element_observations'] if kind!='peak_rms' else summ[cond,layer]['group_observations'])
            ax.set_yscale('log')
        zeros=int(hist[f'{cond}__L{layer:02d}__zero_elements'])
        axes[0].text(.02,.98-.075*conds.index(cond),f'{cond}: zero elements={zeros:,}',transform=axes[0].transAxes,va='top',color=color,fontsize=8)
    for ax,kind in zip(axes,['signed','absolute','peak_rms']):
        maximum=max(int(hist[f'{c}__L{layer:02d}__{kind}'][1:-1].max()) for c in conds)
        if exp=='B' and kind!='peak_rms':
            maximum=max(maximum,max(int(hist[f'{c}_CACHE__L{layer:02d}__{kind}'][1:-1].max()) for c in conds))
        ax.set_ylim(.7,max(2.,maximum*1.4))
    limit=max(summ[c,layer]['overall_absmax'] for c in conds)*1.08
    signed_threshold=min(2**-12,limit/50)
    abs_threshold=min(2**-20,limit/1000)
    axes[0].set_xlim(-limit,limit);axes[0].set_xscale('symlog',linthresh=signed_threshold)
    axes[1].set_xlim(0,limit);axes[1].set_xscale('symlog',linthresh=abs_threshold)
    powers=10.**np.arange(math.ceil(math.log10(signed_threshold*5)),math.floor(math.log10(limit))+1)
    axes[0].set_xticks(np.r_[-powers[::-1],0.,powers])
    powers=10.**np.arange(math.ceil(math.log10(abs_threshold*5)),math.floor(math.log10(limit))+1)
    axes[1].set_xticks(np.r_[0.,powers])
    axes[2].set_xlim(0,12)
    axes[0].set_xlabel('Signed State value (arbitrary units; symlog)')
    axes[1].set_xlabel('|State| (arbitrary units; symlog)')
    axes[2].set_xlabel('group absmax / (RMS + 1e-12)')
    axes[0].set_ylabel('Sampled elements per common bin')
    axes[1].set_ylabel('Sampled elements per common bin')
    axes[2].set_ylabel('Sampled C128 groups per common bin')
    axes[2].legend(fontsize=8);axes[1].legend(fontsize=7)
    scope='A: common FP State pair' if exp=='A' else 'B: independent physical INT8 trajectories'
    fig.suptitle(f'Qwen3.5-9B | model layer {layer} | {scope}\nGDN State, INT8 Key-C128; samples aime26_01..04; continuation 1..2048 sampled every16 + first/last; all32 heads &128 values',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.87]);fig.savefig(out/f'L{layer:02d}_distribution.png');plt.close(fig)

def peak_fig(exp,layer,sample,prompt,formal,out):
    # B figures explicitly show dequantized resident INT8 cache; A shows same-S pair.
    matrices=[];plotdata={}
    for basis in ('Native','KeyH128'):
        cond=f'{exp}_{basis}'+('_CACHE' if exp=='B' else '')
        p=formal/exp/'peak_matrices'/f'{sample}__{cond}__L{layer:02d}__H0V0.npz'
        with np.load(p) as z:
            state=z['state'].copy();cont=z['token_idx'].copy();key=z['key_idx'].copy()
        full=prompt+cont+1
        matrices.append((basis,state,full,key))
        plotdata[basis+'_state']=state;plotdata[basis+'_full_input_position_one_based']=full
        plotdata[basis+'_continuation_position_one_based']=cont+1;plotdata['key_idx']=key
    p=out/'figure_data';p.mkdir(exist_ok=True)
    with (p/f'L{layer:02d}_H0V0_exact_plot_matrices.npz').open('xb') as f:
        np.savez_compressed(f,**plotdata,sample_id=sample,head_idx=0,value_idx=0,
                            experiment=exp,model_layer_id=layer,
                            observation='common FP State pair' if exp=='A' else 'physical INT8 cache decoded with its FP32 scale')
    limit=max(np.abs(a[1]).max() for a in matrices);limit=max(float(limit),1e-12)
    norm=Normalize(-limit,limit)
    fig=plt.figure(figsize=(18,8.2));axs=[]
    grid=fig.add_gridspec(1,3,width_ratios=[1,1,.035],left=.055,right=.95,bottom=.16,top=.83,wspace=.20)
    for i,(basis,s,t,k) in enumerate(matrices):
        ax=fig.add_subplot(grid[0,i],projection='3d');axs.append(ax)
        x,y=np.meshgrid(t,k,indexing='ij')
        ax.plot_surface(x,y,s,cmap='coolwarm',norm=norm,rcount=s.shape[0],ccount=s.shape[1],linewidth=0,antialiased=False)
        ax.set_zlim(-limit,limit);ax.view_init(elev=27,azim=-125)
        ax.set_xlabel('Input token position (1-based)',labelpad=12,fontsize=9)
        ax.set_ylabel('Key index (this basis)',labelpad=10,fontsize=9)
        ax.set_zlabel('Signed State (a.u.)',labelpad=10,fontsize=9)
        ax.set_xticks(np.linspace(t[0],t[-1],5,dtype=int));ax.set_yticks([0,32,64,96,127])
        ax.set_zticks(np.linspace(-limit,limit,5));ax.tick_params(labelsize=8,pad=1)
        ax.set_box_aspect((1.25,1.,.85));ax.set_title(LABELS[basis],pad=15)
    scope='A common FP State pair' if exp=='A' else 'B independent physical INT8 cache, decoded'
    fig.suptitle(f'Qwen3.5-9B | model layer {layer} | {scope}\n{sample}, head=0, value=0; continuation 1..2048, 129 actual sample positions; GDN State, INT8 Key-C128',fontsize=11)
    cax=fig.add_subplot(grid[0,2])
    fig.colorbar(ScalarMappable(norm=norm,cmap='coolwarm'),cax=cax,label='Signed State (common color / height scale)')
    fig.canvas.draw()
    fig.savefig(out/f'L{layer:02d}_H0V0_peak3d.png',bbox_inches='tight',pad_inches=.4);plt.close(fig)
    fig=plt.figure(figsize=(16,6.2))
    grid=fig.add_gridspec(1,3,width_ratios=[1,1,.035],left=.06,right=.94,bottom=.17,top=.78,wspace=.15)
    axs=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
    for ax,(basis,s,t,k) in zip(axs,matrices):
        ax.pcolormesh(t,k,s.T,shading='nearest',cmap='coolwarm',norm=norm,rasterized=True)
        ax.set_title(LABELS[basis]);ax.set_xlabel('Full input token position (1-based)');ax.set_ylabel('Key index in this basis')
    fig.suptitle(f'Qwen3.5-9B | model layer {layer} | {scope}\n{sample}, H0/V0, sampled continuation1..2048; State, INT8 Key-C128. Slice is not a whole-layer summary.',fontsize=11)
    cax=fig.add_subplot(grid[0,2])
    fig.colorbar(ScalarMappable(norm=norm,cmap='coolwarm'),cax=cax,label='Signed State (a.u.)')
    fig.savefig(out/f'L{layer:02d}_H0V0_heatmap.png',bbox_inches='tight',pad_inches=.25);plt.close(fig)

def cross_fig(exp,summ,out):
    fields=[('kurtosis','Group Pearson kurtosis'),('peak_rms','Group absmax / RMS'),
            ('qdq_rel_l2','Local INT8-C128 QDQ relative L2'),('scale','Group FP32 scale'),
            ('out3','Element fraction |x| > 3 x group RMS'),('out6','Element fraction |x| > 6 x group RMS')]
    fig,axs=plt.subplots(3,2,figsize=(14,12))
    for ax,(field,label) in zip(axs.flat,fields):
        for basis,color in zip(('Native','KeyH128'),COLORS):
            cond=f'{exp}_{basis}'
            if field in ('out3','out6'):
                y=[summ[cond,l][field+'_element_fraction'] for l in LAYERS]
                ax.plot(LAYERS,y,'o-',color=color,label=LABELS[basis]+' (pooled elements)',markersize=3)
            else:
                for agg,style in [('median','-'),('p95',':')]:
                    ax.plot(LAYERS,[summ[cond,l][field+'_'+agg] for l in LAYERS],style,color=color,label=LABELS[basis]+' '+agg)
            ax.set_xlabel('Original model layer ID');ax.set_ylabel(label);ax.set_xticks(LAYERS[::2]);ax.legend(fontsize=7)
        if field in ('qdq_rel_l2','scale'):ax.set_yscale('log')
    scope='same FP State pair' if exp=='A' else 'separate physical INT8 trajectories, pre-writeback observations'
    fig.suptitle(f'Qwen3.5-9B | experiment {exp}: {scope}\nGDN State, INT8 Key-C128;4 seed1 samples x2048 continuation tokens; all heads/values; exact group quantiles over516 sampled sample-token events/layer',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'cross_layer_summary.png');plt.close(fig)

DICTIONARY='''# 数据字典与复绘说明

所有数据来自本次 GPU4 新运行。实验 A 的 Native/H128 是共同 FP State 的严格配对；B 是相同 teacher-forced tokens 上各自递推的真实 INT8 轨迹，不能解释成同一张量的旋转前后。

## 逐组原始文件

`formal/{A,B}/groups/*.npz`：`metrics` shape=[sampled_token,32 heads,128 values,16 metrics]，FP64；`metric_names` 给出最后一轴。sample_id、token_idx（continuation 从0计数）、model_layer_id（原始模型层号）、head_idx、value_idx、basis、state_dtype、cache_dtype、scale_dtype 均在文件内。每组覆盖 Key 轴全部128元素，没有只选head0。缺失/未定义数值为NaN，仅零方差Pearson峰度允许NaN。

16列依次为 min/max/absmax/rms/peak_rms/kurtosis/scale/qdq_rel_l2/boundary_count/out3_count/out6_count/squared_error/squared_norm/zero_group/zero_variance/nonfinite_count。RMS=sqrt(mean(x²))；peak_rms=absmax/(RMS+1e-12)；Pearson峰度=mean((x-mean(x))⁴)/mean((x-mean(x))²)²，不减3。scale/QDQ 使用 FP32：max(abs(x),Key).clamp_min(1e-12)/127；round采用ties-to-even；code范围[-127,127]。统计误差时转换FP64归约，但量化公式不改。

qdq_rel_l2=sqrt(squared_error)/(sqrt(squared_norm)+1e-12)。boundary_count 是 |code|=127 的元素数，不是clipping数。out3/out6使用严格大于3×/6×本组RMS，分母128。零组/零方差为0或1指示量。B逐组主文件的source是本步更新后、写INT8之前的FP32临时State，其QDQ误差有实际意义；不能把已恢复的INT8 State再QDQ的零误差冒充新误差。

## 汇总 CSV

`per_layer_summary.csv`：每experiment/condition/model_layer_id一行。`*_median/p95/p99`均是本条件本层全部已采样group观察值的精确NumPy linear分位数。未先对token分位数取平均/中位。`*_valid_groups`单列有效数；zero_variance_groups对应峰度未定义。`out3/out6/boundary_element_fraction`=对应计数之和/element_observations，按元素加权。`pooled_qdq_relative_L2`=sqrt(sum各组误差平方和)/(sqrt(sum各组source平方和)+eps)，不是group相对误差的均值。

`per_layer_head_summary.csv`增加head_idx，其他口径相同。`per_token_layer_summary.csv`增加sample_id/token_idx/prompt_tokens/full_input_position_one_based，只对这一采样事件的4096个(head,value)组汇总。group_observations=组观察数；element_observations=group_observations×128；sampled_sample_token_events不是连续递推token数。

`high_outlier_groups.csv`：每层在Native侧按absmax/RMS降序，固定取前5个（同分按canonical样本、token、head、value），再读取H128对应坐标。A为同S配对；B只对齐输入token与坐标，内部State/输入已不同。此为数据驱动选择，不是随机代表切片；不得用于选择实验样本。

## 直方图

`formal/{A,B}/histogram_counts.npz`保存共享精确bin边界与计数。每个counts长度=len(edges)+1；counts[0]为小于首边界，counts[-1]为大于等于末边界，counts[1:-1]对应[edges[i],edges[i+1])。边界命中归右侧。signed/absolute单位是State任意数值单位；peak_rms无量纲。zero_elements明确记录零值数，零值同时保留在其bin内；非有限值拒绝运行并记录，不静默丢弃。所有条件/层使用同一组bins；绘图只裁剪显示范围，原始计数全保留。

B的`*_CACHE`直方图和峰值矩阵专门表示真实codes×scale恢复后的驻留State分布；不是另一条轨迹。B预写回分布用实线，INT8恢复后分布用虚线。

## 峰值图

`formal/{A,B}/peak_matrices`为每个sample/层/H0V0的129×128 FP32矩阵，保存实际采样token_idx和key_idx。`figures/{A,B}/figure_data`另保存实际用于图中的full_input_position_one_based坐标，数值未经平滑/制造尖峰。A为共同FP State；B默认峰值图画物理INT8 cache恢复值。两侧统一色标/高度/视角，Key index基底不同，不代表同一语义特征。此切片不代表全层分布；全层分布/统计覆盖全部head/value。3D视觉借用SpinQuant图2形式；[原论文](https://arxiv.org/abs/2405.16406)图2是activation，本次为recurrent State，未训练旋转。
'''

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    gate=json.loads((ROOT/'gates/GATE_RESULT.json').read_text());assert gate['status']=='PASS'
    result=json.loads((FORMAL/'RESULT.json').read_text());assert result['status']=='PASS'
    selection=json.loads((ROOT/'inputs/SELECTION.json').read_text())['selected']
    samples=[x['sample_id'] for x in selection];prompt={x['sample_id']:x['prompt_tokens'] for x in selection}
    output=ROOT/'analysis';output.mkdir(exist_ok=False)
    all_layer=[];all_head=[];all_token=[];high=[];summ={};histograms={};paired_rms_errors=[]
    for exp in ('A','B'):
        with np.load(FORMAL/exp/'histogram_counts.npz') as z:histograms[exp]={k:z[k].copy() for k in z.files}
        for layer in LAYERS:
            pair={};event_coords={}
            for basis in ('Native','KeyH128'):
                cond=f'{exp}_{basis}';parts=[];events=[]
                for sid in samples:
                    p=FORMAL/exp/'groups'/f'{sid}__{cond}__L{layer:02d}.npz'
                    with np.load(p) as z:
                        assert tuple(z['metric_names'])==NAMES
                        assert int(z['model_layer_id'])==layer
                        m=z['metrics'].copy();positions=z['token_idx'].copy()
                    assert m.shape==(len(positions),32,128,16)
                    expected=next(s['sampling_positions_zero_based_continuation'] for s in selection if s['sample_id']==sid)
                    assert positions.tolist()==expected
                    parts.append(m)
                    for t,a in zip(positions,m):
                        row=metadata(exp,cond,layer)
                        row.update(sample_id=sid,token_idx=int(t),prompt_tokens=prompt[sid],full_input_position_one_based=prompt[sid]+int(t)+1)
                        row.update(summarize(a));row.update(valid_samples=1,sampled_sample_token_events=1)
                        all_token.append(row);events.append((sid,int(t)))
                m=np.concatenate(parts,axis=0);del parts
                pair[cond]=m;event_coords[cond]=events
                row=metadata(exp,cond,layer);row.update(summarize(m))
                row.update(valid_samples=len(samples),sampled_sample_token_events=len(events))
                all_layer.append(row);summ[cond,layer]=row
                for head in range(32):
                    row=metadata(exp,cond,layer);row['head_idx']=head;row.update(summarize(m[:,head]))
                    row.update(valid_samples=len(samples),sampled_sample_token_events=len(events));all_head.append(row)
            native=pair[f'{exp}_Native'];had=pair[f'{exp}_KeyH128']
            assert event_coords[f'{exp}_Native']==event_coords[f'{exp}_KeyH128']
            if exp=='A':
                error=float(np.max(np.abs(native[...,IDX['rms']]-had[...,IDX['rms']])/(native[...,IDX['rms']]+1e-12)))
                assert error<1e-5,(layer,error)
                paired_rms_errors.append({'model_layer_id':layer,'all_saved_group_max_relative_RMS_difference':error})
            rank=np.argsort(-native[...,IDX['peak_rms']].reshape(-1),kind='stable')[:5]
            for ranknum,idx in enumerate(rank,1):
                event,h,v=np.unravel_index(idx,native.shape[:-1]);sid,t=event_coords[f'{exp}_Native'][event]
                row={'experiment':exp,'model_layer_id':layer,'rank':ranknum,'selection_rule':'top5 Native group absmax/RMS; stable canonical order ties',
                     'sample_id':sid,'token_idx':t,'head_idx':int(h),'value_idx':int(v),
                     'pairing':'same_FP_State' if exp=='A' else 'same_teacher_tokens_distinct_states'}
                for metric in ('absmax','rms','peak_rms','kurtosis','scale','qdq_rel_l2','out3_count','out6_count'):
                    row['Native_'+metric]=float(native[event,h,v,IDX[metric]])
                    row['KeyH128_'+metric]=float(had[event,h,v,IDX[metric]])
                high.append(row)
            del pair,native,had,m;gc.collect()
            print('EXACT_REDUCTION',exp,layer,flush=True)
    csvout(output/'per_layer_summary.csv',all_layer)
    csvout(output/'per_layer_head_summary.csv',all_head)
    csvout(output/'per_token_layer_summary.csv',all_token)
    csvout(output/'high_outlier_groups.csv',high)
    (ROOT/'DATA_DICTIONARY_zh.md').write_text(DICTIONARY)
    comparisons=[]
    for exp in ('A','B'):
        out=ROOT/'figures'/exp;out.mkdir(parents=True,exist_ok=False)
        for layer in LAYERS:
            distribution_fig(exp,layer,summ,histograms[exp],selection,out)
            peak_fig(exp,layer,samples[0],prompt[samples[0]],FORMAL,out)
            row={'experiment':exp,'model_layer_id':layer}
            for field in ('peak_rms_median','kurtosis_median','scale_median','qdq_rel_l2_median','out3_element_fraction','out6_element_fraction'):
                a=summ[f'{exp}_Native',layer][field];b=summ[f'{exp}_KeyH128',layer][field]
                row['Native_'+field]=a;row['KeyH128_'+field]=b
                row['H_over_N_'+field]=b/a if a else np.nan
            comparisons.append(row)
            print('FIGURES',exp,layer,flush=True)
        cross_fig(exp,summ,out)
    csvout(output/'paired_or_aligned_layer_comparison.csv',comparisons)
    findings={}
    for exp in ('A','B'):
        findings[exp]={}
        for field in ('peak_rms_median','kurtosis_median','scale_median','qdq_rel_l2_median','out3_element_fraction','out6_element_fraction'):
            lower=[l for l in LAYERS if summ[f'{exp}_KeyH128',l][field]<summ[f'{exp}_Native',l][field]]
            higher=[l for l in LAYERS if summ[f'{exp}_KeyH128',l][field]>summ[f'{exp}_Native',l][field]]
            findings[exp][field]={'lower_layers':lower,'higher_layers':higher,'unchanged_layers':[l for l in LAYERS if l not in lower+higher]}
    jd(output/'FINDINGS.json',findings)
    jd(output/'ANALYSIS_COMPLETE.json',{'status':'PASS','layer_rows':len(all_layer),'head_rows':len(all_head),'token_layer_rows':len(all_token),
        'high_outlier_rows':len(high),'quantiles':'EXACT NumPy linear over all sampled group observations',
        'plots':len(list((ROOT/'figures').rglob('*.png'))),'approximate_quantiles':False,
        'all_saved_A_group_RMS_checks':paired_rms_errors})
    print(json.dumps(findings,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
