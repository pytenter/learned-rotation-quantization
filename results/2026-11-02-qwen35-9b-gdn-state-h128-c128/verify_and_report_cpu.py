"""Independent terminal CPU checks and Chinese findings; no CUDA import."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,sys,hashlib,subprocess,math
from pathlib import Path
import numpy as np

ROOT=Path(sys.argv[1]);TOTAL=ROOT.parents[1]
LAYERS=[i for i in range(32) if i%4!=3]

def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def line(p,needle):
    for i,s in enumerate(p.read_text().splitlines(),1):
        if needle in s:return f'{p}:{i}'
    raise AssertionError((p,needle))

def main():
    result=read(ROOT/'formal/RESULT.json');assert result['status']=='PASS'
    analysis=read(ROOT/'analysis/ANALYSIS_COMPLETE.json');assert analysis['status']=='PASS'
    assert read(ROOT/'operations/PARENT_COMPLETE.json')['status']=='PASS'
    closed=read(ROOT/'operations/OWNING_CLOSED.json')
    assert closed['worker_wait']==0 and closed['own_NVML_released'] and closed['worker_proc_gone']
    wp=Path('/proc')/str(closed['worker']['pid'])
    assert not wp.exists()
    assert not (Path('/proc')/str(closed['owning_parent']['pid'])).exists()
    gate=read(ROOT/'gates/GATE_RESULT.json');assert gate['status']=='PASS'
    collector=read(ROOT/'formal/COLLECTOR_VALIDATION.json');assert collector['status']=='PASS'
    assert {x['model_layer_id'] for x in collector['real_direct_checks']}=={0,30}
    selection=read(ROOT/'inputs/SELECTION.json')['selected']
    for f in read(ROOT/'inputs/SELECTION.json')['source_files']:assert sha(Path(f['path']))==f['sha256']
    assert [s['sample_id'] for s in selection]==['aime26_01','aime26_02','aime26_03','aime26_04']
    totals={'prompt':0,'continuation':0,'input_steps':0,'sampled_sample_token_events':0,'B_encoding_calls_batch2':0,'B_layer_writes_per_trajectory':0,'B_core_read_checks_batch2':0}
    for s in selection:
        sid=s['sample_id'];inp=read(ROOT/'inputs'/f'{sid}.json');r=read(ROOT/'formal'/f'{sid}_RESULT.json')
        n=len(inp['prompt_ids'])+len(inp['continuation_ids']);a=r['physical_INT8_cache']
        assert r['status']=='PASS' and r['forwards']==n and r['continuation_tokens']==2048
        assert hashlib.sha256(json.dumps(inp['prompt_ids']+inp['continuation_ids'],separators=(',',':')).encode()).hexdigest()==inp['token_stream_sha256']
        assert hashlib.sha256(json.dumps(inp['prompt_ids'],separators=(',',':')).encode('ascii')).hexdigest()==inp['historical_input_ids_hash']
        with Path(s['source_file']).open() as f:
            original=next(json.loads(line) for j,line in enumerate(f,1) if j==s['source_line'])
        assert original['seed']==1 and original['problem_id']==sid
        assert original['generated_token_ids'][:2048]==inp['continuation_ids']
        assert all(int(k) in LAYERS and v==n for k,v in a['write_counts'].items())
        assert all(v==n-1 for v in a['read_counts'].values())
        assert a['core_read_pointer_checks']==24*(n-1)
        assert a['B_codes_rows1_2']['dtype']=='torch.int8' and a['B_scale_rows1_2']['dtype']=='torch.float32'
        assert a['B_codes_rows1_2']['shape']==[2,32,128,128] and a['B_scale_rows1_2']['shape']==[2,32,1,128]
        assert a['FP_reference_row0_only']['storage_bytes']==2097152
        assert a['total_B_code_bytes']==25165824 and a['total_B_scale_bytes']==786432
        assert not a['persistent_B_dense_state']
        assert len(r['collection']['sampling_positions'])==129 and r['collection']['sampled_layer_events']==129*24
        totals['prompt']+=s['prompt_tokens'];totals['continuation']+=2048;totals['input_steps']+=n
        totals['sampled_sample_token_events']+=129;totals['B_encoding_calls_batch2']+=24*n
        totals['B_layer_writes_per_trajectory']+=24*n;totals['B_core_read_checks_batch2']+=24*(n-1)
    layer=rows(ROOT/'analysis/per_layer_summary.csv');heads=rows(ROOT/'analysis/per_layer_head_summary.csv');tokens=rows(ROOT/'analysis/per_token_layer_summary.csv')
    assert len(layer)==96 and len(heads)==3072 and len(tokens)==49536
    expected_groups=516*32*128
    for r in layer:
        assert int(r['group_observations'])==expected_groups and int(r['element_observations'])==expected_groups*128
        assert int(r['valid_samples'])==4 and int(r['nonfinite_elements'])==0
    for r in heads:assert int(r['group_observations'])==516*128 and int(r['valid_samples'])==4
    histogram_checks=[]
    for exp in ('A','B'):
        with np.load(ROOT/'formal'/exp/'histogram_counts.npz') as z:
            for key in z.files:
                if '__L' not in key or key.endswith('zero_elements'):continue
                values=z[key];expected=expected_groups if key.endswith('peak_rms') else expected_groups*128
                assert int(values.sum())==expected,(key,values.sum(),expected)
                histogram_checks.append({'key':key,'sum':int(values.sum()),'underflow':int(values[0]),'overflow':int(values[-1])})
    assert len(list((ROOT/'formal').glob('*/groups/*.npz')))==384
    assert len(list((ROOT/'figures').rglob('*.png')))==146
    for f in read(ROOT/'provenance/SOURCE_MANIFEST.json'):assert sha(ROOT/f['path'])==f['sha256']
    model_files=read(ROOT/'provenance/MODEL_TOKENIZER_SHA256.json')
    for f in model_files:
        p=Path(f['path']);s=p.stat()
        assert s.st_size==f['bytes'] and s.st_ino==f['inode'] and s.st_dev==f['device']
        assert sha(p)==f['sha256'],f['path']
    live_runtime=read(ROOT/'provenance/RUNTIME.json')
    assert sha(Path(live_runtime['modeling_file']))==live_runtime['modeling_sha256']
    git_before=read(ROOT/'provenance/GIT_BEFORE.json');env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    git_after={}
    for name,args in [('branch',['branch','--show-current']),('commit',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
        git_after[name]=subprocess.run(['git','-C',git_before['repo']]+args,capture_output=True,text=True,env=env,check=True).stdout
        assert git_after[name]==git_before[name]
    gpu=subprocess.run(['nvidia-smi','--id=4','--query-gpu=index,name,uuid,driver_version,memory.used,memory.total,utilization.gpu','--format=csv'],capture_output=True,text=True,check=True).stdout
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    assert not any(str(closed['worker']['pid'])==x.split(',')[0].strip() for x in apps.splitlines())
    verification={'status':'PASS','totals':totals,'histogram_checks':histogram_checks,'git_unchanged':True,
                  'model_original_bytes_and_identity_unchanged':True,'worker_actual_integer_wait':0,'own_NVML_released':True,
                  'GPU4_after':gpu,'analysis':analysis,'source_unchanged':True}
    write(ROOT/'analysis/INDEPENDENT_FINAL_VERIFICATION.json',verification)
    table={(r['condition'],int(r['model_layer_id'])):r for r in layer}
    def val(cond,l,f):return float(table[cond,l][f])
    def extremes(cond,field):
        seq=[(val(cond,l,field),l) for l in LAYERS]
        lo=min(seq);hi=max(seq)
        return f'最低为 L{lo[1]}={lo[0]:.5g}，最高为 L{hi[1]}={hi[0]:.5g}'
    def layer_table(exp):
        t=['|原始层 ID|Native 峰均比 median|H128 峰均比 median|Native QDQ误差 median (%)|H128 QDQ误差 median (%)|Native峰度 median|H128峰度 median|',
           '|---:|---:|---:|---:|---:|---:|---:|']
        for l in LAYERS:
            a=f'{exp}_Native';b=f'{exp}_KeyH128'
            t.append(f'|{l}|{val(a,l,"peak_rms_median"):.4g}|{val(b,l,"peak_rms_median"):.4g}|{100*val(a,l,"qdq_rel_l2_median"):.4g}|{100*val(b,l,"qdq_rel_l2_median"):.4g}|{val(a,l,"kurtosis_median"):.4g}|{val(b,l,"kurtosis_median"):.4g}|')
        return '\n'.join(t)
    findings=read(ROOT/'analysis/FINDINGS.json')
    paired=rows(ROOT/'analysis/A_paired_group_effects.csv');assert len(paired)==24
    text=[]
    text.append('# Qwen3.5-9B GDN State 异常值分布实测报告\n\n日期：2026-11-02（北京时间）；服务器 .29；仅物理 GPU4 / RTX3090 24GB。')
    text.append(f'''## 实际完成范围

**实验 A：{result['A']}；实验 B：{result['B']}。** 实际使用 aime26_01～04 的既有 FP_STATE seed1 token 轨迹，固定按 canonical_index 选前4题，没有按异常值或正确率挑样本。各题 continuation 均为2048 token，首个、第16/32/…/2048个位置采样，每题129个采样事件；每层累计516个 sample-token 事件。前缀分别为158/78/129/45 token，总计{totals['prompt']}。每条轨迹实际逐token处理{totals['input_steps']}个输入（{totals['continuation']} continuation + {totals['prompt']} prompt），前缀同样逐token递推，B从第一个prompt token就量化写回。

24个 GDN 原始层 ID：`{','.join(map(str,LAYERS))}`。全注意力层为3/7/11/15/19/23/27/31，未虚构GDN State。每个condition/层统计2,113,536个 C128 group观察值、270,532,608个元素观察值；跨条件或时间重复观察不是独立实验样本。

没有生成新AIME答案、评分、训练、INT4、Value-side Hadamard或Ling测试。146张PNG和所有统计均由本次新GPU数据产生；旧smoke结果没有作为本次观测数据。''')
    text.append('## 主要发现：严格区分共同 State 与递推轨迹')
    for exp in ('A','B'):
        prefix='共同FP State配对的实验A' if exp=='A' else '独立物理INT8轨迹的实验B'
        lines=[f'### {prefix}']
        for field,title in [('peak_rms_median','group 峰均比 median'),('kurtosis_median','Pearson峰度 median'),('scale_median','scale median'),('qdq_rel_l2_median','局部QDQ相对L2 median'),('out3_element_fraction','3×RMS元素比例'),('out6_element_fraction','6×RMS元素比例')]:
            f=findings[exp][field]
            lines.append(f'- H128 的{title}低于 Native：{len(f["lower_layers"])}/24层；高于：{len(f["higher_layers"])}/24层；相同：{len(f["unchanged_layers"])}/24层。较高层：`{f["higher_layers"]}`。')
        worse_tails={}
        for metric in ('absmax','peak_rms','kurtosis','scale','qdq_rel_l2'):
            for q in ('p95','p99'):
                key=metric+'_'+q
                higher=[l for l in LAYERS if val(exp+'_KeyH128',l,key)>val(exp+'_Native',l,key)]
                if higher:worse_tails[key]=higher
        lines.append('逐层检查absmax、峰均比、峰度、scale、局部QDQ误差的p95/p99后，H128更高的层：'+(str(worse_tails) if worse_tails else '均未发现。')+' 这仍不等于每个group均改善。')
        lines.append(f'Native group 峰均比 median 的层间范围：{extremes(exp+"_Native","peak_rms_median")}。Native group absmax p99（State原始数值单位）：{extremes(exp+"_Native","absmax_p99")}。Native 局部QDQ误差 median：{extremes(exp+"_Native","qdq_rel_l2_median")}（误差为比例）。')
        ids=np.array(LAYERS,dtype=float)
        for field,title in [('peak_rms_median','峰均比'),('qdq_rel_l2_median','局部量化误差')]:
            y=np.array([val(exp+'_Native',l,field) for l in LAYERS]);inc=np.all(np.diff(y)>=0);dec=np.all(np.diff(y)<=0)
            corr=float(np.corrcoef(ids,y)[0,1])
            lines.append(f'Native 各层{title} median {"单调不减" if inc else "单调不增" if dec else "不呈严格单调变化"}；原始层ID与该指标的描述性Pearson相关={corr:.3f}，这不是普遍显著性检验。')
        if exp=='B':lines.append('B两条轨迹虽用相同teacher-forced tokens，各层隐藏输入及State已可能分叉；这些差异不能全部归因于同一张量的旋转。误差CSV针对实际更新后、INT8写回前的FP32临时State；B峰值/热力图默认画INT8写回恢复值，另保存写回前矩阵。CACHE图的数值由同一克隆写回源调用相同纯encode/decode重建，并非另做写后内存采样；因回调不修改source，其与实际写回的数值等价属于CODE_VERIFIED。真实bank的INT8/FP32字段、字节数与逐token读取路径属于RUNTIME_VERIFIED。')
        else:
            rates=[float(r['qdq_rel_l2_H_lower_fraction']) for r in paired]
            lines.append(f'A逐组平方L2守恒的最大相对误差为 {collector["max_paired_group_relative_squared_L2_error"]:.4g}。同组RMS在数值误差内守恒，峰值或scale降低表示能量重新分布，并不表示能量被删除。')
            lines.append(f'进一步按同sample/token/head/value逐组配对，H128局部QDQ误差更低的group比例在各层为 {min(rates)*100:.2f}%～{max(rates)*100:.2f}%。因此“24层的median改善”不表示每个group都改善。完整胜/负/相同计数与误差差值精确分位数见 `analysis/A_paired_group_effects.csv`。')
        lines.append(layer_table(exp));text.append('\n\n'.join(lines))
    text.append('''## 分布与位置如何阅读

每层分布图覆盖4题、全部采样位置、32个head、128个value以及每组全部128个Key元素。Native/H128使用同一bin边界与坐标范围。头间差异见`analysis/per_layer_head_summary.csv`；每层Native侧峰均比最大的前5个group及其配对/对齐坐标见`analysis/high_outlier_groups.csv`，可追到sample/token/head/value。该清单是按事先固定规则从数据选择的高异常组，不能当作随机代表样本。

H0/V0峰值图只展示确定性第一题的一个切片；若它尖峰不明显，也未人为筛选或放大。横轴是含prompt的实际输入token位置，另一轴为本基底Key index，颜色/高度为有符号State。Native与H128的同名Key index不是同一原始语义特征。全层逐Key最大值轨迹未转储，因此不将H0/V0图当成整层所有异常位置的证明。

3D图仅借用[SpinQuant图2](https://arxiv.org/abs/2405.16406)的视觉形式。SpinQuant原图是activation，本报告画的是recurrent State，没有运行其旋转训练方法。

## 实现和物理缓存

原工程的`quantize_c128`和`PersistentQDQ`将QDQ结果写回FP32 carrier，不能作为本次B的物理INT8存储证据；这并不意味着历史AIME分数错误。本次独立参考路径沿用已存在的归一化Sylvester H128：q/k先按原BF16公式归一化，再FP32右乘H；对应State左乘H转置。H对称，但代码仍明确写H^T@S。没有做Value-side替换，也没有对已旋转State二次旋转。

真实State shape=[batch,head,key,value]。K=V=128，16个Key head按原模型重复到32个value head；固定batch/head/value，Key轴128个元素构成一组。模型权重和普通激活BF16，GDN更新临时计算FP32，统计归约FP64。实际INT8公式保持：`scale=absmax(Key).clamp_min(1e-12)/127`，`codes=torch.round(source/scale).clamp(-127,127).to(torch.int8)`；ties-to-even。scale为FP32。|code|=127只是边界占用，不能叫clipping。

三行batch分别为A的FP参考、B-Native、B-H128。无跨batch行的归约，B的下一步只从自己的INT8 codes及scale恢复。A持久FP State是独立clone，特意不保留包含B行的完整FP32 storage视图。INT8缓存兼容槽只有Python None与代理，不存FP/BF16 recurrent carrier。每层dense解码/更新只在本步局部存在，写回后不被B缓存持有。观测用FP32切片/FP64统计缓冲与绘图文件不参与递推。

|单条B轨迹的持久字段|每层实际shape/dtype|24层实际bytes|
|---|---|---:|
|State codes|[1,32,128,128], INT8|12,582,912|
|scale|[1,32,1,128], FP32|393,216|
|合计|8-bit codes + scale元数据|12,976,128（12.375MiB）|

A的FP参考24层为50,331,648 bytes（48MiB），只属于实验A；B两轨迹合计代码25,165,824 bytes、scale786,432 bytes。全注意力KV、GDN卷积缓存、模型权重、allocator reserved以及采集临时scratch另计，不能计作INT8 recurrent State或忽略后宣称整模型内存缩小4倍。''')
    locations=['## 实验A各层最高峰均比组的位置（数据驱动选择）',
               '以下是每层Native侧按预定规则选出的第1名，再读取H128同sample/token/head/value组。token按continuation从1计数；不是随机切片，也不表示H128后同Key索引具有原语义。完整前5名及B输入对齐清单见CSV。',
               '|原始层|sample|token|head|value|Native absmax|Native峰均比|H128同组峰均比|',
               '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows(ROOT/'analysis/high_outlier_groups.csv'):
        if r['experiment']=='A' and int(r['rank'])==1:
            locations.append(f'|{r["model_layer_id"]}|{r["sample_id"]}|{int(r["token_idx"])+1}|{r["head_idx"]}|{r["value_idx"]}|{float(r["Native_absmax"]):.5g}|{float(r["Native_peak_rms"]):.4g}|{float(r["KeyH128_peak_rms"]):.4g}|')
    text.insert(5,'\n\n'.join(locations))
    text.append(f'''逐token实际存储写回：每条B轨迹 `{totals['B_layer_writes_per_trajectory']:,}` 次layer更新；两条合计 `{2*totals['B_layer_writes_per_trajectory']:,}`。实现每次encode同时处理两条B行，因此bank的Python encode调用数为 `{totals['B_encoding_calls_batch2']:,}`，不是把两行合并后只算一条轨迹。非首token每层的读取指针核对共 `{totals['B_core_read_checks_batch2']:,}` 次，每次涉及两条B行。逐样本逐层计数、实际shape/stride/storage bytes见`formal/*_RESULT.json`。''')
    text.append('''## 校验与保留的问题记录

- 两个明显不同的合成层0/30、独立NumPy直接统计、原地修改后快照不变、全零/零方差/NaN/Inf、H正交往返、12步物理INT8与dense-QDQ数值参考已通过。非有限输入在整数转换前拒绝；零方差Pearson峰度保留未定义NaN，不填0。
- 真实模型第0和30层使用同一事件的完整32×128组独立重算；共同State的H128读出与状态恢复和FP64单步公式对照均通过。所有正式目标事件各24层恰好一次，无漏层/重复。
- FP32 CUDA常数除法会乘预先舍入的1/127，与CPU直接除法最多差1 ULP。最初严格CPU参考未考虑此差别而失败，已保留；独立小诊断证实CUDA scale逐字等于CPU模拟的FP32乘倒数，实际codes及codes×scale与独立CPU恢复逐字一致，误差统计相差约1e-17。修正的是独立参考的浮点执行口径，不改encode或调宽量化公式。正式门禁和数据来自此后独立版本。
- 初始候选还发生过环境模块缺失、加载审计set无法JSON序列化、复制继承的部分RUNTIME输出触发NOREPLACE拒绝。这些预采集失败均保留在总目录及旧release中；没有冒充正式数据，也没有覆盖失败或历史工件。
- 未使用sys.settrace旧smoke采集器。不能把旧采集器的潜在风险无证据扩大为历史AIME正式实验错误。

## 数据、图和证据路径

完整列定义、统计单位与精确聚合公式见[DATA_DICTIONARY_zh.md](DATA_DICTIONARY_zh.md)。逐层表[per_layer_summary.csv](analysis/per_layer_summary.csv)，逐层逐头表[per_layer_head_summary.csv](analysis/per_layer_head_summary.csv)，逐token逐层表[per_token_layer_summary.csv](analysis/per_token_layer_summary.csv)，异常组[high_outlier_groups.csv](analysis/high_outlier_groups.csv)。

实验A：[跨层汇总](figures/A/cross_layer_summary.png)，每层`figures/A/Lxx_distribution.png`、`Lxx_H0V0_peak3d.png`、`Lxx_H0V0_heatmap.png`。实验B同名文件位于`figures/B/`。可复绘矩阵在`formal/{A,B}/peak_matrices/`，实际作图坐标和值另存`figures/{A,B}/figure_data/`。完整逐group观测在`formal/{A,B}/groups/`，直方图边界、计数、零值与上下溢出在`formal/{A,B}/histogram_counts.npz`。

读取与写回证据：`gates/GATE_RESULT.json`、`formal/COLLECTOR_VALIDATION.json`、`formal/aime26_*_RESULT.json`、`provenance/ACTUAL_GPU4_PROCESS.json`、`operations/WORKER_WAIT.json`、`operations/OWNING_CLOSED.json`；源/模型身份在`provenance/`，文件SHA清单见`MANIFEST.sha256`。''')
    references={
      'INT8 encode / Key axis':line(ROOT/'experiment_core.py','def encode('),
      'Independent snapshot collector':line(ROOT/'experiment_core.py','def observe('),
      'INT8 read path':line(ROOT/'experiment_core.py','def __getitem__('),
      'INT8 write path':line(ROOT/'experiment_core.py','def __setitem__('),
      'Core callbacks / H on q,k':line(ROOT/'run_experiment.py','class CorePatch'),
      'No dense cache slot':line(ROOT/'run_experiment.py','def make_cache('),
      'Per-token full prompt+continuation':line(ROOT/'run_experiment.py',"tokens=sample['prompt_ids']+sample['continuation_ids']"),
      'Existing normalized Sylvester H128':line(ROOT/'vendor/canonical_h128.py','def hadamard('),
      'Historical FP32 QDQ carrier, not B':line(ROOT/'vendor/canonical_h128.py','state.copy_(dequant'),
      'Exact group quantiles':line(ROOT/'analyze_plot_cpu.py','np.quantile(a,[.5,.95,.99]')}
    text.append('\n'.join(['### 源码定位']+[f'- {k}: `{v}`' for k,v in references.items()]))
    text.append(f'''## 解释边界与最终状态

当前仓库分支 `{git_before['branch'].strip()}`、commit `{git_before['commit'].strip()}`，工作区状态前后逐字不变。模型所有原文件预先逐SHA，运行后size/inode/device身份复核通过。本次源文件独立冻结，未推送任何仓库。

这是4题seed1、前2048 continuation token的State机制与分布测试；不等于完整AIME准确率、自由生成质量或长期稳定性结论。跨层/跨head差异有真实统计依据，但不外推到所有题、所有seed或其他模型。B的NaN/Inf计数为0限本次覆盖范围。峰值图的显示裁剪或单切片不能代替全层组统计。

最终GPU worker真实Popen.wait整数0、进程精确消失、own NVML释放；CPU独立校验通过。实验A={result['A']}，B={result['B']}。只完成本请求，不自动继续准确率、训练、其他周期或推送。''')
    (ROOT/'REPORT_zh.md').write_text('\n\n'.join(text)+'\n')
    runtime=live_runtime
    manifest={'status':'PASS','A':result['A'],'B':result['B'],'client_date':'2026-11-02','server':'202.38.247.29',
              'remote_root':str(ROOT),'gpu_physical_index':4,'gpu_uuid':'GPU-de1f9152-32e5-a033-d2bf-ea7abecb606f',
              'model':'Qwen3.5-9B','model_identity_file':'provenance/MODEL_TOKENIZER_SHA256.json','code_git':git_before,
              'source_manifest':'provenance/SOURCE_MANIFEST.json','runtime':runtime,'quantizer':{'bits':8,'symmetric':True,'group_size':128,'axis':2,'layout':'[batch,head,key,value]','codes_dtype':'int8','scale_dtype':'float32','range':[-127,127],'rounding':'ties-to-even','epsilon':1e-12},
              'rotation':{'axis':'Key','matrix':'normalized Sylvester H128, no sign/permutation changes','state_transform':'H.T @ S','qk_transform':'normalize BF16 then float32 @ H','Value_side':False},
              'samples':selection,'layers':LAYERS,'heads':32,'values':128,'keys':128,'totals':totals,
              'A_state':'separate persistent FP32 reference row; paired snapshots S and H.T@S','B_state':'INT8 codes plus FP32 scale; dense FP32 is transient per layer/read only',
              'statistics':'all sampled head/value groups, all128Key elements; FP64 reductions; exact linear quantiles',
              'cache_plot_evidence':'CODE_VERIFIED deterministic encode/decode reconstruction from the same unmodified writeback source; not a separate postwrite memory sampler. Physical cache dtype/shape/bytes and next reads are RUNTIME_VERIFIED.',
              'data_dictionary':'DATA_DICTIONARY_zh.md','report':'REPORT_zh.md','figures':'figures/{A,B}/','data':'formal/{A,B}/',
              'independent_final_review':'analysis/INDEPENDENT_FINAL_VERIFICATION.json','findings':findings,
              'source_evidence':references,'not_accuracy_evaluation':True,'not_pushed':True}
    write(ROOT/'manifest.json',manifest)
    # Manifest checksum excludes itself, then its own digest is separately recorded.
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.cache' not in p.relative_to(ROOT).parts and p.name not in ['MANIFEST.sha256','DELIVERY_READY.json']]
    with (ROOT/'MANIFEST.sha256').open('x') as f:
        for p in sorted(files):f.write(f'{sha(p)}  {p.relative_to(ROOT)}\n')
    write(ROOT/'DELIVERY_READY.json',{'status':'PASS','files':len(files),'manifest_sha256':sha(ROOT/'MANIFEST.sha256'),
        'A':result['A'],'B':result['B'],'CPU_verification':'PASS','own_GPU_released':True})
    print('REPORT_AND_INDEPENDENT_VERIFICATION_PASS',json.dumps({'totals':totals,'figures':146,'files':len(files)}),flush=True)

if __name__=='__main__':main()
