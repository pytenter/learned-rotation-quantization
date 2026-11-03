"""Chinese report from completed, independently reviewed real data only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv
import json
from common import ROOT,LAYERS,METHODS,SAMPLES,jd,sha

def read_csv(name):return list(csv.DictReader((ROOT/'analysis'/name).open()))
def number(x,digits=5):
    if x is None:return '未定义'
    return f'{float(x):.{digits}g}'
def ids(x):return ','.join(str(i) for i in x) if x else '无'
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in row)+'|' for row in rows])

DICTIONARY='''# 数据字典与复绘说明

本次数据为新回放产生的 State 观察，不是旧 NPZ 统计量反推的 State。A 是同一个 FP reference State 的严格配对；B 是相同 teacher-forced 输入下的四条独立、真实 INT8 递推轨迹。样本单位只有4道题，token/head/value/group均为重复观察。

## 文件定位与坐标

第一阶段 `formal/A_ALL_B_NATIVE_H/` 保存 A 的四方法与 B-Native/B-Hadamard；第二阶段 `formal/B_LEARNED/` 保存 B-C6R/B-C7。各阶段下 `{A,B}/groups/<sample>__<A/B>_<method>__Lxx.npz` 为逐组数据；`key_summaries/` 同名文件为逐 Key 数据。GPU smoke 在 `gates/gpu/`，CPU合成检查在 `gates/cpu_collector/`，均未混入正式汇总。

`token_idx` 从 continuation 的0计数，真实采样位置 `[0,15,31,...,2047]`，每题129个；完整输入的一基位置为 `prompt_tokens + token_idx + 1`。保存完整prompt后再逐token回放2048 continuation。4题prompt长度158/78/129/45；每条件8602实际输入更新。原始GDN层ID为0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30；没有给8个全注意力层虚构State。

## 逐组文件

`metrics` 是 `[129 sampled tokens,32 heads,128 values,16 metrics]` FP64，不是完整State。每组覆盖Key轴全部128元素。`metric_names` 依次为 min,max,absmax,rms,peak_rms,kurtosis,scale,qdq_rel_l2,boundary_count,out3_count,out6_count,squared_error,squared_norm,zero_group,zero_variance,nonfinite_count。

- RMS=sqrt(mean_Key(x²))，peak_rms=max_Key(|x|)/(RMS+1e-12)。
- Pearson峰度=mean_Key((x-mean_Key(x))⁴)/mean_Key((x-mean_Key(x))²)²，不减3；零方差记NaN并单独计数。
- 原FP32量化：scale=max_Key(|x|).clamp_min(1e-12)/127；codes=round(x/scale).clamp(-127,127).to(int8)；dequant=codes.float()*scale。torch.round为ties-to-even。GPU常数除法的1ULP执行口径与上一轮相同；独立NumPy oracle模拟已验证的CUDA FP32乘倒数，不改实际量化公式。
- 局部误差=FP64 norm(dequant-x)/(FP64 norm(x)+1e-12)。B的x是实际本步更新后、物理INT8写入前的FP32临时State，未对缓存恢复值再次QDQ来制造接近零的误差。
- out3_count/out6_count是**本组Key元素数**，严格 `|x|>3/6×本组RMS`，范围0..128。boundary_count是 `|code|=127` 占用，不称为clipping。
- squared_error/squared_norm为Key轴误差/source平方和；zero_group、zero_variance为0/1；非有限值在量化前拒绝，不静默替换。
- `key_absmax_index` 是 `[129,32,128 values]` uint8，保存该组最大绝对值的Key坐标；相同最大值时取最小Key index。

## 逐 Key 文件

`key_absmax_over_value` FP32与 `key_out3_value_count` UINT16，shape均为 `[129 sampled tokens,32 heads,128 keys]`。前者是固定Key跨128个Value的最大绝对值；后者是固定Key处、超过各自Value对应C128 group的3×RMS的**Value元素数**（0..128）。数组轴、sample_id、层ID、方法、token位置全部保留。

B另存 `cache_key_absmax`、`cache_key_out3`，从刚写入的实际bank codes/scales反量化观察，属于真实缓存字段读后观察，不是另编码克隆source的推断。CACHE的阈值使用该缓存恢复State自身的group RMS。主图及QDQ误差采用预写回State，CACHE仅为单独数据与物理存储证据。

Native/Hadamard/C6R/C7处于不同Key基底；相同Key编号不自动表示相同原始语义特征。数据不能恢复全部原始State：逐组仅有指标与峰值位置，逐Key仅有跨Value的max/计数。

## 精确汇总 CSV

- per_layer_summary：192行；每实验/方法/原始层一行。每行2,113,536 group观察、270,532,608元素观察，4题×129事件。
- per_layer_head_summary：6144行；每head有66,048 group观察。
- per_sample_layer_summary：768行；逐题、方法、层，528,384 group观察，保留每道题结果。
- per_token_layer_summary：99,072行；每事件4096组。
- 各 `*_median/p95/p99` 为相应scope中所有group的精确NumPy linear分位数，**不是先对token求分位数再平均**。`*_valid_groups`、zero_groups、zero_variance_groups、undefined_kurtosis_groups、nonfinite_elements单列。
- out3/out6/boundary_element_fraction=实际计数之和/(group_observations×128)，按元素数加权；CSV是比例，图中乘100标为%。
- pooled_qdq_relative_L2=sqrt(sum squared_error)/(sqrt(sum squared_norm)+1e-12)，不是group相对误差的均值。
- `group_peak_QDQ_Pearson`、`group_out3_count_QDQ_Pearson`是同层同条件的描述性相关系数；不把相关group当成独立题目做显著性推断。

## 严格配对与异常组选择

`A_paired_group_effects.csv` 720行，六组候选/参照比较×24层×(4题分别+四题合并)。核心三组是C6R/Hadamard、C7/Hadamard、C7/C6R，另含三种旋转对Native。配对单位固定sample/token/layer/head/value。

candidate_lower表示候选值严格更小；higher更大；equal浮点数精确相等，不另加未声明容差。fraction分母为有效配对组数，包括相同项；差值始终candidate-reference。对peak、局部QDQ、out3/out6元素数分别给胜/负/相同及差值median/p95/p99。QDQ比值仅在reference>0时计算，reference=0另计；全零时比值未定义而非0。原正式输入所有非有限State已被拒绝，只有零方差峰度允许NaN。

`high_outlier_groups.csv` 1920行，按peak/RMS降序取前5，同分按canonical sample、采样token、head、value的扁平坐标升序。Native_selected在Native选组再读取四种方法同坐标；A严格配对，B只是相同输入位置的独立State对齐。self_selected为各自方法选出的前5，两种策略明确分开。保存absmax、peak/RMS、局部误差、Key outlier数与本基底argmax Key index；不是随机代表样本，不能据此改选题目。

## 图与复绘数据

每实验50张PNG：24分布图、24全head Key位置图、cross_layer_summary及outlier_error_relationship，共100。分布共享bin和坐标：signed/absolute使用symlog x、log count y，peak/RMS使用线性x、log count y；bin为共同的对数幅度边界/线性peak边界，曲线是计数而非等宽密度。under/over和零元素另存；不抬高零bin，不裁剪数据。

Key位置图展示全部32head×128Key：左图每坐标在4题×129采样时间×128Value上的最大绝对值；右图累积3×RMS outlier计数/(516事件×128Value)，以百分比显示。同层四方法共用完整色标范围，各自坐标基底不同。它不是H0/V0切片。精确作图矩阵在 `figures/{A,B}/figure_data/`。

关系图每点是一种方法/层的汇总，预先固定标注层0/14/30，不能将96个点称为96个独立题目。排序CSV区分完整同序、outlier相同但误差不同、严格相反的排序。

## 矩阵身份

`provenance/MATRIX_IDENTITY.csv` 对每原始层保存参数/有效变换SHA、dtype、shape、head共享和正交性。实际q/k为原BF16归一化后FP32 `@H` 再 `@Delta`；State观察为 `Delta.T@(H.T@S)`；缓存已在该基底时不再旋转。有效矩阵FP64参考为实际FP32 H/Delta转FP64后相乘，仅用于身份和正交性，不作为单GEMM捷径。未对checkpoint做QR/SVD或任何重新正交化。
'''

def main():
    complete=json.loads((ROOT/'analysis/ANALYSIS_COMPLETE.json').read_text());assert complete['status']=='PASS'
    independent=json.loads((ROOT/'analysis/INDEPENDENT_DATA_REVIEW.json').read_text());assert independent['status']=='PASS'
    bitmap=json.loads((ROOT/'visual_qa/BITMAP_CHECKS.json').read_text());assert bitmap['status']=='PASS'
    rows=read_csv('per_layer_summary.csv');index={(r['experiment'],r['method'],int(r['model_layer_id'])):r for r in rows}
    findings=json.loads((ROOT/'analysis/FINDINGS.json').read_text());history=json.loads((ROOT/'analysis/HISTORICAL_COMPARISON.json').read_text())
    cpu=json.loads((ROOT/'gates/CPU_AUDIT.json').read_text());gpu=json.loads((ROOT/'gates/GPU_SMOKE_RESULT.json').read_text())
    runtime=json.loads((ROOT/'provenance/ACTUAL_RUNTIME.json').read_text());formal=json.loads((ROOT/'formal/RESULT.json').read_text())
    wait=json.loads((ROOT/'operations/WORKER_WAIT.json').read_text());closed=json.loads((ROOT/'operations/OWNING_CLOSED.json').read_text())
    assert wait['owning_Popen_wait_integer']==0 and closed['own_NVML_released']
    source_use=json.loads((ROOT/'provenance/FORMAL_CHECKPOINT_USE_EVIDENCE.json').read_text())
    paired_rows=read_csv('A_paired_group_effects.csv')
    text=['# Qwen3.5-9B GDN State：Native / Hadamard / C6R / C7 实测对比',
          '\n实验日期：2026-11-03，北京时间；.29物理GPU4，RTX3090。仅State分布与真实INT8机制观察，没有新AIME自由生成、准确率评分或训练。',
          '\n## 直接回答',
          '\n**实验A：四配置均PASS；实验B：四配置均PASS。**',
          '\n**1. 旋转后是否仍存在outlier？**']
    for method in ['C6R','C7']:
        pieces=[]
        for exp in ['A','B']:
            p=findings[exp]['outlier_presence'][method]
            pieces.append(f"{exp}中>3×RMS元素观察{p['out3_elements']:,}个（{len(p['layers_with_out3'])}/24层），>6×RMS为{p['out6_elements']:,}个（{len(p['layers_with_out6'])}/24层）")
        exists=any(findings[e]['outlier_presence'][method]['out3_elements']>0 for e in ['A','B'])
        text.append(f"\n**{method}旋转后{'仍存在' if exists else '在本次覆盖内未观察到'}按本组RMS定义的outlier**："+'；'.join(pieces)+'。这些是重复元素观察数，不是独立题目数。')
    compare_rows=[]
    for exp in ['A','B']:
        for method,reference in [('C6R','Hadamard'),('C7','Hadamard'),('C7','C6R')]:
            for field,label in [('peak_rms_median','峰均比median'),('out3_element_fraction','>3RMS元素比例'),('qdq_rel_l2_median','局部QDQ误差median')]:
                f=findings[exp]['comparisons'][method+'_vs_'+reference][field]
                compare_rows.append([exp,method+'/'+reference,label,ids(f['candidate_lower_layers']),ids(f['candidate_higher_layers']),ids(f['equal_layers'])])
    text+=['\n**2. 与Hadamard相比，分别在哪些层更好或更差？** 不预设学习旋转总是更好。前两组直接对照Hadamard，第三组直接比较C7与C6R。下表“低”表示候选的该具体指标更小：峰均比更不突出、比例更少或局部误差更小；三个指标分别判断。',
           table(['实验','候选/参照','指标','候选更低的层ID','候选更高的层ID','相同层ID'],compare_rows)]
    text.append('\n**3. outlier指标与局部INT8误差排序是否一致？**')
    for exp in ['A','B']:
        f=findings[exp]
        text.append(f"\n{exp}中，四方法的完整峰均比median排序与局部误差median排序在{len(f['peak_and_error_full_order_same_layers'])}/24层完全同序；>3RMS比例与误差完整同序为{len(f['out3_and_error_full_order_same_layers'])}/24层。包含{len(f['strict_opposite_order_cases'])}个方法对/层的严格相反排序记录；相同outlier指标而误差不同另作相同项处理，不强行称反向。完整差值与方向在`analysis/metric_order_comparison.csv`。")
    concordance=[]
    for r in findings['A_B_direction_concordance']:
        if (r['candidate'],r['reference']) in [('C6R','Hadamard'),('C7','Hadamard'),('C7','C6R')] and r['metric'] in ['peak_rms_median','out3_element_fraction','qdq_rel_l2_median']:
            concordance.append([r['candidate']+'/'+r['reference'],r['metric'],len(r['same_direction_layers']),ids(r['different_direction_layers'])])
    text+=['\n**4. A与B结论是否一致？** 下表统计方法间比较方向在A/B是否一致；B的状态与隐藏输入已分叉，不能将B差异解释为对同一State只旋转一次的效果。',
           table(['候选/参照','指标','A/B同方向层数','不同方向层ID'],concordance),
           '\n**5. 哪些来自严格配对？** 严格配对证据来自A。同sample/token/layer/head/value覆盖全部128个Key，下面给出各层候选局部QDQ误差更低的group比例范围；不是四题正确率。B仅是相同teacher-forced输入下的独立递推。',
           '\n**6. 哪些仍无法判断？** 只有4题、seed1和2048 continuation前缀，不宣称统计显著、普遍最优、完整AIME准确率或长期稳定性。C6R/C7针对历史各自目标训练，不保证以本次outlier或局部量化误差为最优目标；本次未重新训练或重选checkpoint。']
    paired_table=[]
    for candidate,reference in [('C6R','Hadamard'),('C7','Hadamard'),('C7','C6R')]:
        for sample in ['ALL4',*SAMPLES]:
            sub=[r for r in paired_rows if r['candidate']==candidate and r['reference']==reference and r['sample_id']==sample]
            assert len(sub)==24
            win=[float(r['qdq_rel_l2_lower_fraction'])*100 for r in sub]
            loss=[float(r['qdq_rel_l2_higher_fraction'])*100 for r in sub]
            paired_table.append([candidate+'/'+reference,sample,f'{min(win):.2f}–{max(win):.2f}',f'{min(loss):.2f}–{max(loss):.2f}'])
    text.append(table(['A候选/参照','题目scope','逐层候选误差更低group比例(%)','逐层候选误差更高group比例(%)'],paired_table))
    text+=['\n## 固定代表层与全部24层', '\n以下层0/14/30在看数据前固定，用作早/中/晚层例子。百分比栏已将CSV比例乘100；peak/RMS无量纲。']
    detail=[]
    for exp in ['A','B']:
        for layer in [0,14,30]:
            for method in METHODS:
                r=index[exp,method,layer]
                detail.append([exp,layer,method,number(r['peak_rms_median']),number(float(r['out3_element_fraction'])*100),
                               number(float(r['qdq_rel_l2_median'])*100),number(r['kurtosis_median'])])
    text.append(table(['实验','原层ID','方法','peak/RMS median','>3RMS元素(%)','局部QDQ median(%)','Pearson峰度median'],detail))
    for exp in ['A','B']:
        data=[]
        for layer in LAYERS:data.append([layer]+[number(float(index[exp,m,layer]['qdq_rel_l2_median'])*100) for m in METHODS])
        text += [f'\n### {exp}：全部24层局部QDQ误差median（%）',table(['原层ID',*METHODS],data),
                 f'\n![{exp}跨层主图](figures/{exp}/cross_layer_summary.png)',
                 f'\n![{exp}异常指标与误差关系](figures/{exp}/outlier_error_relationship.png)']
    text+=['\n## 检查点、有效变换与方向',
           '\nC6R锁定历史update94/exposure752；C7锁定update19/exposure152，未按文件时间或本次结果选择。实际正式canonical60记录两者各60条，逐条核对均绑定下列SHA，见`provenance/FORMAL_CHECKPOINT_USE_EVIDENCE.json`。初始部署工程记录的NOT_FROZEN/失败范围仍保留，不据此冒称旧失败通过；这里使用后续实际正式评测绑定的冻结参数，并进行了本次独立验证。']
    matrix=[]
    for method in ['C6R','C7']:
        ident=cpu['checkpoint_identities'][method]
        matrix.append([method,ident['rotation_file'],ident['rotation_sha256'],'每层FP32[128,128] Delta；32 heads共享'])
    text.append(table(['配置','实际加载副本绝对路径','文件SHA256','参数布局'],matrix))
    text+=['\n原参数来自.55迁移备份中的`restored/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1/.../rotations/`，副本SHA不变。本次fresh读取的是实际正式评测使用的FP32冻结导出；原A800 theta未在本机重新读取，其路径/SHA及与导出逐层exact一致的原审计保留在导出/选择记录中。正式推理合同是直接加载冻结Delta，不在推理主机重建theta。',
           '\n实际q/k顺序为：原BF16归一化 → FP32右乘H128 → FP32右乘该层Delta。有效数学变换W=H×Delta；State观察为Deltaᵀ(HᵀS)，恢复为H(DeltaS)。B缓存本来就处于各自旋转坐标，下一步只反量化，不再重复旋转State。Value轴不旋转。C6R/C7并非对称矩阵；Hadamard对称性不能替代转置方向验证。',
           '\n[逐层矩阵身份表](provenance/MATRIX_IDENTITY.csv)含Native/Hadamard/C6R/C7共96行、原层映射、共享关系、参数与有效矩阵SHA及正交性。`effective_rotations_reference_fp64.npz`的参考W只用于身份/正交性，H和Delta取实际FP32值再转FP64；实际q/k仍是两次FP32 GEMM，不以一次合并矩阵乘法冒充原运算。未进行QR/SVD重新正交化。']
    norm=json.loads((ROOT/'formal/A_ALL_B_NATIVE_H/COLLECTOR_VALIDATION.json').read_text())
    normrows=[[m,number(norm['max_paired_group_relative_squared_L2_error'][m])] for m in METHODS]
    text.append(table(['A方法','所有实际group平方L2守恒最大相对误差'],normrows))
    op=gpu['plans'][0]['common_state_operator_checks']
    text.append(f"\n真实24层×四方法共96项共同State算子检查，最大读出相对L2误差{max(r['readout_relative_L2'] for r in op):.6g}，恢复State最大相对误差{max(r['state_recovery_relative_L2'] for r in op):.6g}；预声明算子阈值2e-5，检查点原正交性阈值1e-4，均未提高。非对称合成Givens的正确方向/错误转置反例、FP64单步对照、独立NumPy全组指标、Key坐标统计、零/非有限值处理见CPU/GPU gates。")
    text+=['\n## 真实INT8持续写回与内存',
           '\n第一阶段batch3为A的独立Native FP32 reference与B-Native/B-Hadamard；A对同一参考快照做四方法观察，QDQ不回写参考。第二阶段batch2只有B-C6R/B-C7，完全没有FP reference字段。每题每阶段都从空cache、prompt第一个token开始，所有GDN每步实际写INT8，未使用窗口、跳token量化或高精度B carrier。',
           '\n每阶段两个B配置是同一批次张量中两个不重叠行，所有规范化/量化归约不跨batch行。实际bank每层codes shape `[2,32,128,128]` INT8、scales `[2,32,1,128]` FP32；下表为每配置一行的逻辑贡献，不能把整块共享storage重复算给每条轨迹。',
           table(['每个B配置持久字段','每层一行shape/dtype','24层bytes'],[
               ['codes','[1,32,128,128] INT8','12,582,912'],['scales','[1,32,1,128] FP32','393,216'],['合计','INT8 + FP32 metadata','12,976,128（12.375MiB）']]),
           '\n每个B配置跨4题共8602输入更新（8192 continuation+410 prompt）、每层8602次写回，合计206,448次layer写回；四配置合计825,792次单配置layer写回。首输入后每层下一步读回从codes×scale恢复，实际pointer检查与计数逐样本保留。缓存兼容槽是字典代理`{0:None}`，不是额外Tensor；A仅持有独立clone的一行FP32，48MiB。B临时FP32解码/更新、FP64观测、CPU memmap统计均不参与下一步持久State。这里按本请求从prompt首token即物理INT8写回，不能将B称为原历史AIME自由生成的完整路径复现；复用的是已确认旋转参数及指定量化口径。',
           '\nB主误差与主Key图来自写入前更新State；另保存从刚写入的真实bank字段解码的CACHE Key汇总，已区分于上一轮仅同source重建CACHE的证据范围。完整shape/dtype/storage与每层计数见`formal/*/*_RESULT.json`。模型权重、attention KV、GDN卷积、临时scratch及allocator不能忽略后宣称整模型显存缩小4倍。']
    samples=formal['results']
    mem=[['CUDA峰值allocated（样本内forward+观测）',f"{max(r['peak_allocated_bytes'] for r in samples):,} bytes"],
         ['CUDA峰值reserved（同scope）',f"{max(r['peak_reserved_bytes'] for r in samples):,} bytes"],
         ['样本末端attention KV最大观察值',f"{max(r['full_attention_KV_bytes'] for r in samples):,} bytes"],
         ['样本末端GDN卷积最大观察值',f"{max(r['GDN_conv_bytes'] for r in samples):,} bytes"],
         ['GPU worker总实际运行时间（含加载/smoke/压缩）',f"{wait['elapsed_seconds']:.2f} s"],
         ['正式forward+观测累计',f"{sum(r['model_forward_and_observation_seconds'] for r in samples):.2f} s"],
         ['正式回放加CPU压缩累计',f"{sum(r['with_compression_seconds'] for r in samples):.2f} s"],
         ['CPU精确汇总和绘图',f"{complete['elapsed_seconds']:.2f} s"]]
    text.append(table(['资源/时长口径','实际记录'],mem))
    text+=['\n这些时长包含大量采样、同步、D2H、压缩和不同batch布局，不能当成四种方法的公平吞吐benchmark；worker墙钟也不等于GPU实际忙时，未连续测量的GPU active时间不填0。KV只测样本末端，未宣称完整生命周期峰值。实际总文件/bytes见最终`DELIVERY_READY.json`与`MANIFEST.sha256`，不在封存前预填存储量。',
           '\n## 与上一轮Native/Hadamard对照',
           f"\n模型/tokenizer17个文件fresh SHA、Transformers modeling字节、Torch/Transformers版本、同一GPU4、输入tokens均与上一轮匹配。Native/H阶段保留batch3。对照归档的12个数值字段×96条件/层组合，共{history['compared_numeric_fields']}项，其中{history['numeric_exact_fields']}项数值完全相同，最大相对绝对差{history['maximum_relative_absolute_difference']:.6g}；这是描述性对照，没有临时制造新的PASS容差。",
           '\n源码核对确认，两轮Hadamard的State观察均为显式matmul(Hᵀ,S)，q/k均在原BF16归一化后FP32右乘H；量化和group_metrics函数保留原字节。不能只凭设计匹配或目录名宣称实际数值完全重复，完整实测差值见`analysis/historical_native_hadamard_comparison.csv`；若出现差异，应继续依据真实输入、dtype、批次和环境证据解释，不编造运算路径变化。']
    text.append(table(['实验','指标','H/Native方向改变的层ID'],[[r['experiment'],r['metric'],ids(r['changed_Hadamard_vs_Native_direction_layers'])] for r in history['trend_comparisons']]))
    text+=['\n## 数据、图与校验入口',
           '\n完整列定义、单位、量化和零/相等/分母零规则见[DATA_DICTIONARY_zh.md](DATA_DICTIONARY_zh.md)。逐层、逐头、逐题、逐token、严格A配对及异常组CSV均在`analysis/`。各层分布图`figures/{A,B}/Lxx_distribution.png`，全32head Key图`Lxx_key_outlier_locations.png`；准确作图汇总矩阵在`figure_data/`，原逐Key时间数据在各阶段`key_summaries/`。',
           '\nGPU worker真实owning Popen.wait整数0，精确epoch退出，own NVML释放；未向其他任务发信号。模型与24项冻结源运行后再次核对通过。独立数据复核见`analysis/INDEPENDENT_DATA_REVIEW.json`，位图检查见`visual_qa/BITMAP_CHECKS.json`；最终本地镜像另逐SHA验证。没有覆盖历史checkpoint、模型或旧实验结果。',
           '\n最终结论仅覆盖本次4题/seed1/2048前缀的State机制观察，不能外推准确率、其他模型或训练收益。完成后停止，不自动新实验、训练、准确率或推送。']
    report='\n\n'.join(text)+'\n'
    with (ROOT/'REPORT_zh.md').open('x') as f:f.write(report)
    with (ROOT/'DATA_DICTIONARY_zh.md').open('x') as f:f.write(DICTIONARY)
    jd(ROOT/'analysis/REPORT_CREATED.json',{'status':'PASS','report_sha256':sha(ROOT/'REPORT_zh.md'),
        'dictionary_sha256':sha(ROOT/'DATA_DICTIONARY_zh.md'),'actual_runtime':runtime,'formal_checkpoint_record_counts':{m:source_use['conditions'][m]['formal_record_count'] for m in ['C6R','C7']}})
    print('CHINESE_REPORT_CREATED',flush=True)

if __name__=='__main__':main()
