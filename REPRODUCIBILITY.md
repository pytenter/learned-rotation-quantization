# 结果读取、复绘与运行环境

## 2026-11-03：Native / Hadamard / C6R / C7

目录：[`results/2026-11-03-qwen35-9b-gdn-state-c6r-c7`](results/2026-11-03-qwen35-9b-gdn-state-c6r-c7)。所有原始文件保持原字节，[发布说明](results/2026-11-03-qwen35-9b-gdn-state-c6r-c7/PUBLICATION_README.md)解释Git/Release分工。

报告、主要CSV、100张主图和逐Key作图矩阵可直接读取。完整逐组、逐Key时间数组、验证NPZ和逐token明细需明确运行`scripts/fetch_release_data.py`下载。下载与校验只需Python3.9+标准库，不使用模型或GPU；读取NPZ用NumPy，绘图用Matplotlib。

| 项目 | 实际记录 |
|---|---|
| 模型与GPU | Qwen3.5-9B；.29物理GPU4 / RTX3090 |
| 环境 | PyTorch2.5.1+cu121，Transformers5.16.0.dev0历史Qwen3.5 overlay |
| modeling文件SHA | `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a` |
| 输入 | 上轮相同4条seed1 FP token轨迹；完整prompt+2,048 continuation |
| 第一阶段batch3 | A-Native FP reference、B-Native、B-Hadamard；A观察参考State的4种变换 |
| 第二阶段batch2 | B-C6R、B-C7；无持久FP reference字段 |
| GDN临时计算 / 指标归约 | FP32 / FP64 |
| 每个B持久State | INT8 codes+FP32 scales；24层合计12,976,128 bytes |
| C6R | update94/exposure752；SHA `a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6` |
| C7 | update19/exposure152；SHA `57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8` |

完整身份和生命周期在provenance、gates及operations中。模型/tokenizer17文件、原输入、24项冻结源和96行矩阵身份均核对；原GPU worker真实owning wait0并释放GPU。Native/Hadamard与旧归档的1,152个汇总字段精确一致。

run_gpu.py、rotation_adapter.py、物理bank/collector和analysis_cpu保留实际执行源码。它们含历史绝对路径、GPU UUID和一次性claim保护，属于已完成运行的记录；复绘应在独立工作副本适配路径，避免覆盖冻结结果。本次发布不重新执行实验脚本。

包含小型冻结旋转checkpoint；基础模型、完整历史Transformers overlay、Python环境和编译缓存不包含。普通`pip install transformers`不等于恢复了相同环境。原theta导出历史与本次直接读取冻结Delta的证据范围见报告和身份记录。

## 2026-11-02：Native / 固定H128

原环境、切片、物理bank和复绘说明原文保留在[REPRODUCIBILITY_20261102.md](REPRODUCIBILITY_20261102.md)，原发布说明在[README_20261102.md](README_20261102.md)。原命令`scripts/fetch_raw_data.py`及`scripts/verify_delivery.py`继续只对应上一轮，不改变旧Release及清单。
