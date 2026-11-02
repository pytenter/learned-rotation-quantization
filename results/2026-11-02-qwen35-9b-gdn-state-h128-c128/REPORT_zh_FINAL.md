# Qwen3.5-9B GDN State 异常值分布实测报告

日期：2026-11-02（北京时间）；服务器 .29；仅物理 GPU4 / RTX3090 24GB。

## 实际完成范围

**实验 A：PASS；实验 B：PASS。** 实际使用 aime26_01～04 的既有 FP_STATE seed1 token 轨迹，固定按 canonical_index 选前4题，没有按异常值或正确率挑样本。各题 continuation 均为2048 token，首个、第16/32/…/2048个位置采样，每题129个采样事件；每层累计516个 sample-token 事件。前缀分别为158/78/129/45 token，总计410。每种运行条件跨4题累计逐token处理8602个输入（8192 continuation + 410 prompt），前缀同样逐token递推，B从第一个prompt token就量化写回。

24个 GDN 原始层 ID：`0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30`。全注意力层为3/7/11/15/19/23/27/31，未虚构GDN State。每个condition/层统计2,113,536个 C128 group观察值、270,532,608个元素观察值；跨条件或时间重复观察不是独立实验样本。

没有生成新AIME答案、评分、训练、INT4、Value-side Hadamard或Ling测试。146张PNG和所有统计均由本次新GPU数据产生；旧smoke结果没有作为本次观测数据。

## 主要发现：严格区分共同 State 与递推轨迹

### 共同FP State配对的实验A

- H128 的group 峰均比 median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的Pearson峰度 median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的scale median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的局部QDQ相对L2 median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的3×RMS元素比例低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的6×RMS元素比例低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

逐层检查absmax、峰均比、峰度、scale、局部QDQ误差的p95/p99后，H128更高的层：均未发现。 这仍不等于每个group均改善。

Native group 峰均比 median 的层间范围：最低为 L20=4.0615，最高为 L26=6.0111。Native group absmax p99（State原始数值单位）：最低为 L1=0.043941，最高为 L28=0.70687。Native 局部QDQ误差 median：最低为 L20=0.0091877，最高为 L26=0.013567（误差为比例）。

Native 各层峰均比 median 不呈严格单调变化；原始层ID与该指标的描述性Pearson相关=0.160，这不是普遍显著性检验。

Native 各层局部量化误差 median 不呈严格单调变化；原始层ID与该指标的描述性Pearson相关=0.160，这不是普遍显著性检验。

A逐组平方L2守恒的最大相对误差为 6.622e-07。同组RMS在数值误差内守恒，峰值或scale降低表示能量重新分布，并不表示能量被删除。

进一步按同sample/token/head/value逐组配对，H128局部QDQ误差更低的group比例在各层为 90.10%～99.08%。因此“24层的median改善”不表示每个group都改善。完整胜/负/相同计数与误差差值精确分位数见 `analysis/A_paired_group_effects.csv`。

|原始层 ID|Native 峰均比 median|H128 峰均比 median|Native QDQ误差 median (%)|H128 QDQ误差 median (%)|Native峰度 median|H128峰度 median|
|---:|---:|---:|---:|---:|---:|---:|
|0|4.255|2.669|0.9619|0.6043|6.595|2.742|
|1|4.811|2.688|1.088|0.6083|8.852|2.76|
|2|5.117|2.7|1.155|0.611|10.75|2.768|
|4|5.304|2.643|1.198|0.598|12.63|2.7|
|5|5.18|2.666|1.171|0.6032|11.52|2.738|
|6|4.339|2.724|0.9808|0.6163|7.446|2.809|
|8|4.837|2.719|1.094|0.6152|9.319|2.797|
|9|4.459|2.712|1.008|0.6135|8.035|2.785|
|10|4.393|2.727|0.9938|0.6171|7.334|2.807|
|12|4.27|2.723|0.9658|0.6162|6.69|2.811|
|13|4.422|2.733|0.9996|0.6185|7.617|2.822|
|14|4.374|2.738|0.9894|0.6196|7.363|2.818|
|16|4.181|2.736|0.9454|0.6193|6.381|2.826|
|17|4.215|2.758|0.9529|0.6243|6.737|2.847|
|18|4.085|2.769|0.9242|0.6267|6.19|2.86|
|20|4.062|2.764|0.9188|0.6255|6.054|2.849|
|21|4.49|2.737|1.015|0.6199|8.08|2.829|
|22|4.475|2.734|1.012|0.6187|7.883|2.824|
|24|4.637|2.751|1.048|0.623|9.084|2.822|
|25|5.8|2.701|1.308|0.6111|17.29|2.775|
|26|6.011|2.59|1.357|0.586|17.81|2.638|
|28|4.724|2.738|1.068|0.6199|9.517|2.785|
|29|5.267|2.708|1.191|0.6135|12.93|2.754|
|30|4.611|2.756|1.042|0.6238|8.404|2.831|

### 独立物理INT8轨迹的实验B

- H128 的group 峰均比 median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的Pearson峰度 median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的scale median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的局部QDQ相对L2 median低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的3×RMS元素比例低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

- H128 的6×RMS元素比例低于 Native：24/24层；高于：0/24层；相同：0/24层。较高层：`[]`。

逐层检查absmax、峰均比、峰度、scale、局部QDQ误差的p95/p99后，H128更高的层：均未发现。 这仍不等于每个group均改善。

Native group 峰均比 median 的层间范围：最低为 L20=4.0472，最高为 L26=5.9733。Native group absmax p99（State原始数值单位）：最低为 L1=0.044961，最高为 L28=0.71378。Native 局部QDQ误差 median：最低为 L12=0.0080509，最高为 L26=0.012617（误差为比例）。

Native 各层峰均比 median 不呈严格单调变化；原始层ID与该指标的描述性Pearson相关=0.151，这不是普遍显著性检验。

Native 各层局部量化误差 median 不呈严格单调变化；原始层ID与该指标的描述性Pearson相关=0.192，这不是普遍显著性检验。

B两条轨迹虽用相同teacher-forced tokens，各层隐藏输入及State已可能分叉；这些差异不能全部归因于同一张量的旋转。误差CSV针对实际更新后、INT8写回前的FP32临时State；B峰值/热力图默认画INT8写回恢复值，另保存写回前矩阵。CACHE图的数值由同一克隆写回源调用相同纯encode/decode重建，并非另做写后内存采样；因回调不修改source，其与实际写回的数值等价属于CODE_VERIFIED。真实bank的INT8/FP32字段、字节数与逐token读取路径属于RUNTIME_VERIFIED。

|原始层 ID|Native 峰均比 median|H128 峰均比 median|Native QDQ误差 median (%)|H128 QDQ误差 median (%)|Native峰度 median|H128峰度 median|
|---:|---:|---:|---:|---:|---:|---:|
|0|4.237|2.671|0.8095|0.5831|6.558|2.744|
|1|4.798|2.689|0.988|0.5914|8.782|2.762|
|2|5.108|2.701|1.075|0.6008|10.71|2.769|
|4|5.289|2.645|1.092|0.5825|12.54|2.703|
|5|5.17|2.667|1.015|0.5892|11.47|2.739|
|6|4.321|2.726|0.8819|0.6018|7.373|2.811|
|8|4.815|2.72|0.9421|0.596|9.205|2.798|
|9|4.452|2.713|0.8209|0.5933|8.009|2.786|
|10|4.383|2.729|0.855|0.5973|7.287|2.81|
|12|4.251|2.726|0.8051|0.5939|6.626|2.815|
|13|4.417|2.735|0.8963|0.6026|7.605|2.824|
|14|4.364|2.739|0.8813|0.6062|7.33|2.818|
|16|4.165|2.737|0.8204|0.6013|6.337|2.828|
|17|4.209|2.76|0.8794|0.6134|6.712|2.849|
|18|4.085|2.769|0.8476|0.6106|6.183|2.863|
|20|4.047|2.766|0.8184|0.5987|6.023|2.853|
|21|4.471|2.741|0.8997|0.5976|7.942|2.833|
|22|4.465|2.734|0.9441|0.609|7.828|2.825|
|24|4.609|2.752|0.9161|0.5902|8.91|2.83|
|25|5.749|2.706|1.17|0.5978|16.57|2.78|
|26|5.973|2.596|1.262|0.5714|17.4|2.646|
|28|4.7|2.736|0.9795|0.6051|9.334|2.783|
|29|5.216|2.711|1.056|0.5979|12.55|2.758|
|30|4.601|2.759|0.8905|0.5988|8.333|2.838|

## 实验A各层最高峰均比组的位置（数据驱动选择）

以下是每层Native侧按预定规则选出的第1名，再读取H128同sample/token/head/value组。token按continuation从1计数；不是随机切片，也不表示H128后同Key索引具有原语义。完整前5名及B输入对齐清单见CSV。

|原始层|sample|token|head|value|Native absmax|Native峰均比|H128同组峰均比|
|---:|---|---:|---:|---:|---:|---:|---:|
|0|aime26_03|480|20|102|0.0094465|11.2|1.281|
|1|aime26_03|1040|6|43|0.0086639|10.84|1.897|
|2|aime26_03|2048|3|115|0.038951|10.71|1.722|
|4|aime26_03|256|2|98|0.00099901|10.72|1.918|
|5|aime26_04|1424|15|125|0.031287|10.49|1.929|
|6|aime26_04|288|11|89|0.010502|10.58|1.913|
|8|aime26_03|1776|21|102|0.14849|11|1.516|
|9|aime26_03|2048|14|35|0.078448|10.83|1.693|
|10|aime26_02|1264|13|17|0.11121|11.12|1.503|
|12|aime26_03|96|18|8|0.0090483|11.28|1.217|
|13|aime26_04|1|27|116|0.12424|11.04|1.485|
|14|aime26_02|1712|5|51|0.0024088|10.33|2.116|
|16|aime26_03|1|20|83|0.041297|10.22|2|
|17|aime26_02|1|9|3|0.0094077|10.15|2.218|
|18|aime26_02|304|2|62|0.013674|9.92|2.05|
|20|aime26_03|368|17|53|0.0076706|10.17|2.013|
|21|aime26_03|1376|24|21|0.039913|10.68|1.629|
|22|aime26_03|1216|26|9|0.078893|10.7|1.678|
|24|aime26_03|1824|7|50|0.026955|10.88|1.619|
|25|aime26_02|1600|26|36|0.16564|11.06|1.475|
|26|aime26_04|1|29|31|1.6974|11.09|1.447|
|28|aime26_02|160|17|87|0.078697|10.78|1.715|
|29|aime26_01|688|3|90|1.2443|10.95|1.633|
|30|aime26_01|1584|23|65|0.094744|10.7|1.769|

## 分布与位置如何阅读

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

A的FP参考24层为50,331,648 bytes（48MiB），只属于实验A；B两轨迹合计代码25,165,824 bytes、scale786,432 bytes。全注意力KV、GDN卷积缓存、模型权重、allocator reserved以及采集临时scratch另计，不能计作INT8 recurrent State或忽略后宣称整模型内存缩小4倍。

逐token实际存储写回：B-Native与B-H128各自跨4题累计 `206,448` 次layer更新；两配置合计 `412,896`。实现每次encode同时处理两条B行，因此bank的Python encode调用数为 `206,448`，不是把两行合并后只算一条轨迹。非首token每层的读取指针核对共 `206,352` 次，每次涉及两条B行。逐样本逐层计数、实际shape/stride/storage bytes见`formal/*_RESULT.json`。

## 校验与保留的问题记录

- 两个明显不同的合成层0/30、独立NumPy直接统计、原地修改后快照不变、全零/零方差/NaN/Inf、H正交往返、12步物理INT8与dense-QDQ数值参考已通过。非有限输入在整数转换前拒绝；零方差Pearson峰度保留未定义NaN，不填0。
- 真实模型第0和30层使用同一事件的完整32×128组独立重算；共同State的H128读出与状态恢复和FP64单步公式对照均通过。所有正式目标事件各24层恰好一次，无漏层/重复。
- FP32 CUDA常数除法会乘预先舍入的1/127，在本次独立诊断中，与CPU直接除法最多差1 ULP。最初严格CPU参考未考虑此差别而失败，已保留；独立小诊断证实CUDA scale逐字等于CPU模拟的FP32乘倒数，实际codes及codes×scale与独立CPU恢复逐字一致，误差统计相差约1e-17。修正的是独立参考的浮点执行口径，不改encode或调宽量化公式。正式门禁和数据来自此后独立版本。
- 初始候选还发生过环境模块缺失、加载审计set无法JSON序列化、复制继承的部分RUNTIME输出触发NOREPLACE拒绝。这些预采集失败均保留在总目录及旧release中；没有冒充正式数据，也没有覆盖失败或历史工件。
- 未使用sys.settrace旧smoke采集器。不能把旧采集器的潜在风险无证据扩大为历史AIME正式实验错误。

## 数据、图和证据路径

完整列定义、统计单位与精确聚合公式见[DATA_DICTIONARY_zh.md](DATA_DICTIONARY_zh.md)。逐层表[per_layer_summary.csv](analysis/per_layer_summary.csv)，逐层逐头表[per_layer_head_summary.csv](analysis/per_layer_head_summary.csv)，逐token逐层表[per_token_layer_summary.csv](analysis/per_token_layer_summary.csv)，异常组[high_outlier_groups.csv](analysis/high_outlier_groups.csv)。

实验A：[跨层汇总](figures/A/cross_layer_summary.png)，每层`figures/A/Lxx_distribution.png`、`Lxx_H0V0_peak3d.png`、`Lxx_H0V0_heatmap.png`。实验B同名文件位于`figures/B/`。可复绘矩阵在`formal/{A,B}/peak_matrices/`，实际作图坐标和值另存`figures/{A,B}/figure_data/`。完整逐group观测在`formal/{A,B}/groups/`，直方图边界、计数、零值与上下溢出在`formal/{A,B}/histogram_counts.npz`。

读取与写回证据：`gates/GATE_RESULT.json`、`formal/COLLECTOR_VALIDATION.json`、`formal/aime26_*_RESULT.json`、`provenance/ACTUAL_GPU4_PROCESS.json`、`operations/WORKER_WAIT.json`、`operations/OWNING_CLOSED.json`；源/模型身份在`provenance/`，文件SHA清单见`MANIFEST.sha256`。

### 源码定位

- INT8 encode / Key axis: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/experiment_core.py:31`
- Independent snapshot collector: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/experiment_core.py:170`
- INT8 read path: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/experiment_core.py:103`
- INT8 write path: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/experiment_core.py:111`
- Core callbacks / H on q,k: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/run_experiment.py:46`
- No dense cache slot: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/run_experiment.py:93`
- Per-token full prompt+continuation: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/run_experiment.py:223`
- Existing normalized Sylvester H128: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/vendor/canonical_h128.py:23`
- Historical FP32 QDQ carrier, not B: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/vendor/canonical_h128.py:109`
- Exact group quantiles: `/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_H128_C128_GPU4_20261102_V1/releases/ORACLE_FP32_V4/analyze_plot_cpu.py:47`

## 解释边界与最终状态

当前仓库分支 `research-sync-2026-09-02`、commit `b2b8467b4d0733425c4fcc8feff25f3d0aed7d33`，工作区状态前后逐字不变。模型所有原文件预先逐SHA，运行后size/inode/device身份复核通过。本次源文件独立冻结，未推送任何仓库。

这是4题seed1、前2048 continuation token的State机制与分布测试；不等于完整AIME准确率、自由生成质量或长期稳定性结论。跨层/跨head差异有真实统计依据，但不外推到所有题、所有seed或其他模型。B的NaN/Inf计数为0限本次覆盖范围。峰值图的显示裁剪或单切片不能代替全层组统计。

最终GPU worker真实Popen.wait整数0、进程精确消失、own NVML释放；CPU独立校验通过。实验A=PASS，B=PASS。只完成本请求，不自动继续准确率、训练、其他周期或推送。
