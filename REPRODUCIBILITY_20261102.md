# 结果读取、复绘与运行环境

实验目录：[`results/2026-11-02-qwen35-9b-gdn-state-h128-c128`](results/2026-11-02-qwen35-9b-gdn-state-h128-c128)。该目录中的交付文件保持原字节，包含实际执行过的源代码及运行收据。

## 不用 GPU 即可读取的结果

- `analysis/*.csv`：完整汇总与配对统计，列定义见 `DATA_DICTIONARY_zh.md`。
- `figures/{A,B}/`：每组 73 张最终 PNG。
- `figures/{A,B}/figure_data/*.npz`：实际作图矩阵及坐标。
- `formal/{A,B}/peak_matrices/*.npz`：真实 FP32 State 切片。
- `formal/{A,B}/histogram_counts.npz`：共用 bin、计数、零值及上下溢出。
- `formal/{A,B}/groups/*.npz`：下载 Release 后可用。每层/条件/样本为 `[129,32,128,16]` FP64 指标，不是完整 State 快照；16 个字段及轴语义见数据字典。

NumPy 用于 NPZ 读取；Matplotlib 用于绘图。`analyze_plot_cpu.py`、`paired_group_effects_cpu.py`、`plot_layout_v2.py`、`rerender_layout_cpu.py` 与 `visual_qa_cpu.py` 保留了本次实际分析及布局修复代码。它们带有历史固定路径，若在别处复绘，应使用独立工作副本适配路径，避免覆盖本仓库的冻结结果。

## 原实际 GPU 环境

| 项目 | 运行记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，.29 物理 GPU4 |
| Python | 3.10.18 |
| PyTorch | 2.5.1+cu121 |
| CUDA（PyTorch 编译版本） | 12.1 |
| Transformers | 5.16.0.dev0，使用历史 Qwen3.5 backport overlay |
| 模型 | Qwen3.5-9B，未修改 BF16 checkpoint |
| Attention | SDPA |
| Batch | 3 行：A-FP reference / B-Native / B-H128 |
| GDN 临时计算 | FP32 |
| B 持久 State | INT8 codes + FP32 scales |
| 统计归约 | FP64 |

完整版本与实际加载文件哈希在 `provenance/RUNTIME.json`、`MODEL_TOKENIZER_SHA256.json` 和 `SOURCE_MANIFEST.json`。原 Qwen3.5 modeling 文件 SHA256 为 `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`。版本字符串相同不能替代这个文件身份。

本仓库保留采集器、物理 INT8 bank、H128 实现和独立验证代码；不包含模型权重、历史 Transformers 完整 overlay 或服务器 Python 环境。因此不能把普通 `pip install transformers` 视为已复原同一执行环境。

## 源码与生命周期证据

- `experiment_core.py`：物理 bank、INT8 encode/decode、逐组统计、采样与存盘。
- `run_experiment.py`：真实 forward/core patch、三行隔离、全 prompt + continuation 逐 token 路径及正式采集。
- `vendor/canonical_h128.py`：既有归一化 Sylvester H128；其中旧 FP32 QDQ carrier 不作为 B 的持久缓存。
- `qualification_cpu.py`、`gates/`：合成与真实层检查。
- `supervise.py`、`operations/WORKER_WAIT.json`、`OWNING_CLOSED.json`：唯一 owning Popen 的真实整数退出和 NVML 释放。
- `provenance/` 中保留预采集失败及独立解释，具体路径以工件清单为准；失败未删除，也未算作正式样本。

原启动代码含固定模型路径、GPU UUID 与 NOREPLACE 保护，属于已完成运行的记录，不应在冻结目录中直接再次执行。重新实验需要另外建立输出目录并复原依赖、验证输入与门禁；此次发布不启动任何新实验。
