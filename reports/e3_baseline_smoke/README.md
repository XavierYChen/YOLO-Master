# E3 baseline：CUDA 训练链路 smoke test

本报告记录 Tencent YOLO-Master E3 方向的最小训练链路验证。测试使用 `yolo-master-n`、COCO8、单个 epoch，验证本地安装、模型构建、MoE 层加载、CUDA/AMP、数据加载、训练、验证和权重保存能够端到端执行。

## 准入登记

- Owner：[@XavierYChen](https://github.com/XavierYChen)
- 仓库：`XavierYChen/YOLO-Master`
- 锁定 commit：`0854b5d72cc64b92ffa73f0d7000e3ec057d3b53`
- 数据：`coco8.yaml`（4 张训练图、4 张验证图）
- 模型：`ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml`
- 训练：1 epoch，`imgsz=320`，`batch=2`，`workers=0`，`seed=0`
- 设备：NVIDIA GeForce RTX 3060 Laptop GPU（6144 MiB），CUDA 12.8
- 结论：smoke test 通过，但存在可恢复警告，详见“结果与警告”。

## 从零复现

在 Windows Anaconda Prompt 中，从仓库根目录运行：

```cmd
reports\e3_baseline_smoke\reproduce.cmd
```

等价核心命令：

```cmd
yolo detect train model=ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml data=coco8.yaml epochs=1 imgsz=320 batch=2 device=0 workers=0 project=runs/e3_baseline name=detect_smoke_logged
```

## 通过标准

以下条件均满足时，判定基础训练链路 smoke 通过：

1. `yolo-master-n.yaml` 可解析，模型成功构建；
2. 日志中出现 `OptimizedMOEImproved` 层，证明 Master/MoE 配置已加载；
3. CUDA 设备可用且 AMP 检查通过；
4. COCO8 的 train/val 数据均成功扫描；
5. 至少一个 epoch 完成，随后完成验证；
6. 生成 `best.pt`、`last.pt`、指标 CSV 和可视化证据；
7. 进程最终正常结束并打印结果保存路径。

## 结果与警告

模型成功构建，共 442 层、7,546,984 个参数；日志确认三处 `OptimizedMOEImproved` 模块参与构图。训练使用 CUDA:0，AMP 检查通过，完成 1 epoch 并执行最终验证，成功保存 `best.pt` 和 `last.pt`。

本次只验证工程链路，不报告精度结论。COCO8 规模极小且只训练 1 epoch，mAP 为 0 不代表正式训练性能。

完整日志中存在两项需要保留的警告：

- 首次 epoch 检测到 `Gradient NaN/Inf`，框架恢复健康检查点后自动重放；重放成功并完成训练与验证。
- 最终验证的混淆矩阵绘图分配约 23 MiB 数组失败；数值验证和权重保存不受影响。另一次同配置 smoke 已生成混淆矩阵图片，作为独立结果证据保留。

因此本次准入结论是“端到端链路通过，带可恢复警告”。NaN 和绘图内存问题应在正式训练前继续排查，不能把本次结果当作稳定性或精度证明。

## 证据索引

- `configs/args.yaml`：第一次 smoke 的完整训练参数。
- `results/smoke.log`：同配置复跑的完整终端日志，已去除终端颜色控制字符。
- `results/results.csv`：第一次 smoke 的单 epoch 指标。
- `results/results.png`：第一次 smoke 的指标可视化。
- `results/train_batch0.jpg`：训练数据及标注抽样。
- `results/val_batch0_labels.jpg`：验证集真实标注。
- `results/val_batch0_pred.jpg`：验证预测可视化。
- `results/confusion_matrix.png`：第一次 smoke 生成的混淆矩阵。
- `env/environment.md`：软件、CUDA、GPU、commit 和 Git 状态。
- `limitations.md`：风险、边界和降级方案。

说明：指标与图片来自首次 `detect_smoke-3`；完整日志来自随后执行的同配置 `detect_smoke_logged`。两次运行均用于验证工程链路，不将二者混合作为同一次精度实验。

## 准入检查表

| 检查项 | 状态 | 证据 |
|---|---|---|
| 环境安装 | 已完成 | `env/environment.md`、`results/smoke.log` |
| 基线/最小任务 | 已完成 | COCO8 单 epoch 训练与验证 |
| 复现命令 | 已完成 | `reproduce.cmd` 与本 README |
| 配置文件 | 已完成 | `configs/args.yaml`、模型 YAML 路径 |
| 完整日志 | 已完成 | `results/smoke.log` |
| 结果证据 | 已完成 | CSV、曲线图、batch 图、预测图、混淆矩阵 |
| 设计说明 | 已完成 | 本 README 的目标、通过标准与证据索引 |
| 风险与降级 | 已完成 | `limitations.md` |
| 代码/方案链接 | 已完成 | 本目录所在分支/PR |

## 与 E3 正式工作的边界

本报告只证明基础 Master/MoE 检测训练链路可运行，不等于已经完成 E3 的路由透视、专家负载统计、路由熵可视化或开销测量。E3 专项实现应在后续分支中基于锁定 checkpoint、统一 schema 和重复测量单独验收。
