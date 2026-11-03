# 2026-11-03 四旋转实验发布副本

先读[REPORT_zh.md](REPORT_zh.md)和[RESULTS_AT_A_GLANCE_zh.md](RESULTS_AT_A_GLANCE_zh.md)。A为同FP State严格配对，B为同token输入下四条独立、每token实际INT8写回的递推；所有8个实验/配置组合均PASS。

## 原始字节和布局

原交付2,638个文件、29,940,519,116 bytes：主清单2,636个成员、29,940,173,912 bytes，另外2个是清单和完成回执本身。

- Git：321个原始文件，76,184,248 bytes；含报告、100张主图、首版图和视觉复核、主要CSV、作图矩阵、源码、输入、checkpoint及JSON运行证据。
- Release：2,317个原始文件；含1,536个正式逐组/逐Key NPZ、774个验证NPZ、7个逐token明细表或采集日志，共34个TAR，附SHA256SUMS。
- 本说明、RELEASE_ASSETS.json、PUBLICATION_MANIFEST.sha256和发布核验回执属于额外发布元数据，不冒充原科学交付成员。

所有原始文件保持原字节。原主清单SHA256为`e1114ddb9b3504cc96f292b0cf490c6e0bcd575b329fcc64a8d8bdb24175f89e`；PUBLICATION_MANIFEST额外纳入清单和完成回执，共2,638项。

[同仓库Release](https://github.com/pytenter/learned-rotation-quantization/releases/tag/qwen35-9b-gdn-state-c6r-c7-20261103) · [逐包/逐文件索引](RELEASE_ASSETS.json) · [TAR SHA列表](RELEASE_SHA256SUMS)

32个正式包按实验、方法和题目命名，每包含全部24层的groups与key_summaries；另外2包为验证NPZ和逐token表/事件日志。解包恢复原目录布局。

```bash
# 在仓库根目录运行
python3 scripts/verify_release_data.py --git-only

# 可选：下载完整数据
python3 scripts/fetch_release_data.py --remove-archives
python3 scripts/verify_release_data.py
```

## 证据边界

每题完整prompt+2,048 continuation，采样`[0,15,31,...,2047]`，每题129个事件。题目只有4个，不以重复group观察制造显著性。

逐组NPZ为FP64指标，不是完整State。逐Key数据是跨Value最大绝对值和超阈计数，不能反推全部State。B的误差取更新后、真实写回前的FP32临时State，下一步读自身INT8 codes和FP32 scales，没有高精度窗口carrier。

C6R/C7实际q/k为BF16归一化后两次FP32乘法`@H @Delta`，State观察为`Delta.T @ (H.T @ S)`；有效矩阵FP64参考用于身份/正交性，不替代运行路径，未重新正交化checkpoint。

服务器原件保留在`/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_FOUR_ROTATIONS_C6R_C7_20261103_V1`。原报告“不自动推送”指实验结束时状态，本发布来自之后明确授权，没有追加生成、评分、训练或GPU实验。

原REPORT_CREATED记录的报告SHA保持有效。48张分布图和2张关系图的首版有显示刻度/文字重叠，首版均保留；独立CPU修复仅调整显示布局，统计、bin、数据点、坐标范围及冻结源码未变，证据见operations/CPU_TICK_LAYOUT_RESULT.json和visual_qa/VISUAL_REVIEW.json。
