# 数据字典与复绘说明

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
