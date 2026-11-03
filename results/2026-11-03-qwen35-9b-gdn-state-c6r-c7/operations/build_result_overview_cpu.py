"""Independent interpretation supplement, solely from closed existing measurements."""
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT=Path('/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_FOUR_ROTATIONS_C6R_C7_20261103_V1')
assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
assert json.loads((ROOT/'operations/CPU_PIPELINE_RESULT.json').read_text())['status']=='PASS_COMPLETE_CPU_ANALYSIS_REPORT_PIPELINE'
assert json.loads((ROOT/'operations/CPU_TICK_LAYOUT_RESULT.json').read_text())['status']=='PASS_CPU_TICK_LAYOUT_V2'
assert json.loads((ROOT/'visual_qa/VISUAL_REVIEW.json').read_text())['status']=='PASS'
rows=list(csv.DictReader((ROOT/'analysis/per_layer_summary.csv').open()))
findings=json.loads((ROOT/'analysis/FINDINGS.json').read_text())
history=json.loads((ROOT/'analysis/HISTORICAL_COMPARISON.json').read_text())
assert len(rows)==192
methods=['Native','Hadamard','C6R','C7']

def table(headers,data):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(str(v) for v in row)+'|' for row in data])

text=['# 实测结论速览与解释',
      '本文件是原始报告完成后的独立 CPU 解释补充，只读取已经完成的数据；未修改原报告、统计、模型或实验源码。完整24层、各题配对结果及身份见[REPORT_zh.md](REPORT_zh.md)，列定义见[DATA_DICTIONARY_zh.md](DATA_DICTIONARY_zh.md)。',
      '**A四配置均PASS，B四配置均PASS。C6R和C7仍有outlier；相较Native，三种旋转都明显降低本次观察的异常程度，但C6R/C7没有在所有层全面优于Hadamard。**',
      'A为同一个FP State的严格配对；B为同token输入条件下、从prompt首token开始逐步物理INT8写回的独立递推。以下表格不把B当作同一State的旋转配对，也不把group当成独立题目。',
      '## 全层累计元素比例（实际计数加权）']
out=[]
for exp in ['A','B']:
    for method in methods:
        sub=[r for r in rows if r['experiment']==exp and r['method']==method]
        denominator=sum(int(r['element_observations']) for r in sub)
        out.append([exp,method,f"{sum(int(r['out3_count']) for r in sub)/denominator*100:.6f}",
                    f"{sum(int(r['out6_count']) for r in sub)/denominator*100:.6f}"])
text.append(table(['实验','方法','>3×本组RMS元素(%)','>6×本组RMS元素(%)'],out))
text.append('C6R对极端>6×RMS尾部的改善比>3×RMS更明显。例如A中C6R的>6×RMS计数为93,589，Hadamard为171,007；但>3×RMS累计比例略高于Hadamard。因此“outlier更少”必须说明阈值和指标，不能合成一个无条件排名。')
text.append('## 比Hadamard更低的层数（24层中）')
data=[]
for exp in ['A','B']:
    for candidate in ['C6R','C7']:
        comp=findings[exp]['comparisons'][candidate+'_vs_Hadamard']
        data.append([exp,candidate,*[len(comp[key]['candidate_lower_layers']) for key in
            ['peak_rms_median','out3_element_fraction','qdq_rel_l2_median']]])
text.append(table(['实验','方法','峰均比median更低','>3RMS元素比例更低','局部QDQ误差median更低'],data))
text.append('A中C6R局部误差median较低的层为8/12/18/20/22/24/25/28；C7为8/12/14/18/25/28/30。B中分别为8/22/24/25/28和8/12/18/21/25/28/30。其余层相应指标较高，没有相同层。该层数判断取每层全部group的精确median，不能替代逐题逐组的配对胜率；后者在A_paired_group_effects.csv中单独提供。')
text.append('## 峰均比、outlier与INT8误差的关系')
text.append('**实测事实：峰均比与局部误差有关，但不构成完全相同的排序。** 下表为每个方法在24层内逐组Pearson描述性相关系数的最小—最大值，不是显著性检验。')
corr=[]
for exp in ['A','B']:
    for method in methods:
        sub=[r for r in rows if r['experiment']==exp and r['method']==method]
        a=[float(r['group_peak_QDQ_Pearson']) for r in sub]
        b=[float(r['group_out3_count_QDQ_Pearson']) for r in sub]
        corr.append([exp,method,f'{min(a):.4f}—{max(a):.4f}',f'{min(b):.4f}—{max(b):.4f}'])
text.append(table(['实验','方法','peak/RMS vs本步QDQ误差 r范围','本组>3RMS元素数 vs本步QDQ误差 r范围'],corr))
text.append('**机制解释（由量化公式推断）：** maxabs决定INT8步长，RMS关联相对误差的分母，因此peak/RMS通常与相对误差紧密相关。但是实际误差还取决于128个元素落在量化格点的什么位置，outlier计数又只反映是否超过某阈值。峰值、尾部计数或其median变小，都不能单独保证真实舍入误差变小。B中历史量化改变了当前State及格点位置；观察到的相关性变弱不能单凭本实验进一步归因为唯一原因。')
index={(r['experiment'],r['method'],int(r['model_layer_id'])):r for r in rows}
detail=[]
for method in ['Hadamard','C6R','C7']:
    r=index['A',method,30]
    detail.append([method,f"{float(r['peak_rms_median']):.8f}",f"{float(r['out3_element_fraction'])*100:.8f}",f"{float(r['qdq_rel_l2_median'])*100:.8f}"])
text.append('一个直接反例是A的第30层：C6R峰均比median和>3RMS比例略低，但局部误差median反而略高于Hadamard。')
text.append(table(['A第30层','peak/RMS median','>3RMS(%)','本步QDQ误差median(%)'],detail))
for exp in ['A','B']:
    p=findings[exp]
    text.append(f"{exp}的四方法完整排序中，峰均比median与误差median在{len(p['peak_and_error_full_order_same_layers'])}/24层同序；>3RMS比例与误差median在{len(p['out3_and_error_full_order_same_layers'])}/24层同序。")
text.extend(['## 完成范围与证据',
             f"旧Native/Hadamard对照的{history['compared_numeric_fields']}个汇总数值字段中，{history['numeric_exact_fields']}个完全相同；只对比这些已列字段，不扩称所有历史原始张量相同。",
             '仅4题、seed1、每题2048 continuation前缀；没有新自由生成、评分、训练、INT4或准确率结论。全部原始统计、逐Key时间数组、96行矩阵身份、运行日志与真实owning wait/释放证据均保留。',
             '100张最终主图已做位图检查及视觉检查。首版分布图的symlog刻度过密，关系图的旋转方法层号文字重叠；保留48张分布原图、2张关系原图和原视觉检查，独立CPU重绘仅调整显示刻度、零元素标签及关系图层号标注。最终关系图仅对Native标注预定0/14/30层，所有方法的数据点仍完整显示。所有分析文件、bin、逐Key作图数据和冻结源码SHA均保持不变。见operations/CPU_TICK_LAYOUT_RESULT.json与visual_qa/VISUAL_REVIEW.json。',
             '主报告仍为原CPU pipeline生成的同一文件，REPORT_CREATED.json的SHA保持有效。本补充与所有文件将统一纳入最终MANIFEST.sha256。'])
path=ROOT/'RESULTS_AT_A_GLANCE_zh.md'
with path.open('x') as f:f.write('\n\n'.join(text)+'\n')
print(json.dumps({'status':'PASS_CPU_INTERPRETATION_SUPPLEMENT','path':str(path),
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'new_GPU_work':False},ensure_ascii=False),flush=True)
