# 数据字典与复绘说明

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


## CACHE图的证据口径补充

B的 *_CACHE 图/矩阵是从同一帧、未被回调修改的写回源，调用与实际bank完全相同的纯encode/decode函数所得的写回恢复值。它不是另一次写后内存拷贝采样；数值等价属于CODE_VERIFIED。真正B bank的INT8 codes、FP32 scales、实际bytes/stride、逐token写回与下一步读取有独立RUNTIME_VERIFIED证据。此说明不改变任何观测数据或图。

analysis/A_paired_group_effects.csv 逐层记录A同一State的全group严格配对比较：H_lower/H_higher/equal计数，局部QDQ误差差值和正Native误差组的比值精确分位数。观察单位为同sample/token/head/value的C128 group，不是层median之间的比较。
