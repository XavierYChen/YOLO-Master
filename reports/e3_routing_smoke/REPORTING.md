# 中期汇报与 PR 材料

## 5 页中期汇报
### 1. 问题与假设
五族混合系统缺乏统一路由观测入口。本阶段验证：不改核心 forward，能否对 MoE、MoT、Latent 在真实 COCO8 上产出可解释、可复现的路由记录。

### 2. 实验设计
腾讯基线 246e79cfe418cfd90f4738bace56b02245dc38f8；本项目工具代码哈希见 environment.json。
COCO8 验证集首图，seed=0，320，batch=1，FP32，CPU；随机初始化，不训练。
关闭/开启工具采集对照，3 次预热、20 对交替前向。命令见 README 与结果中的 command.txt。

### 3. 已完成证据
三族 13 层结构化记录和静态图；输出一致性通过，hooks 清理通过；15 项回归测试通过。
CPU off/on 中位数 ms：MoE 92.834/93.849，MoT 106.955/108.202，Latent 99.654/98.111。
静态图直接标出 expert usage、normalized entropy 和 normalized Gini 三位小数，并明确每族指标语义。
保留原始计时、日志、配置、图及哈希；GPU 和官方 quick 失败如实记录。

### 4. 结论与不确定性
已验证本机 CPU 三族最小观测链路。MoE 选择份额与 MoT/Latent 概率明确区分；未发布辅助损失用 null。
不能据此判断训练效果、路由坍缩、稳定开销或训练减速 <10%；P1/P2 未完成。

### 5. 下一阶段与风险
P0：可复用采集接口、更多输入与训练态契约；P1：本地面板与完整训练 on/off；P2：真实空间路由叠图与两分钟视频。
先解决 PyTorch DLL/机器内存稳定性。若 GPU 不稳定则保留 CPU 路线；若 P1 不达标则降低采样频率并重测。

## PR 描述四节
### 改动摘要
新增独立 E3 smoke 工具与中文复现包，不改腾讯原文件。修正旧工具的内部 router 分数误标、aux 缺失填零、配置未读取及复现记录不完整问题。
MoE 读取最终 top-k 输出；MoT/Latent 读取原生 snapshot。结构化记录包含语义、状态、数据来源，失败显式可见。

### 测试证据
CPU 真实 COCO8 三族运行通过，13 层；15 项 pytest 通过；新增 Python 文件 Ruff/format/codespell 通过。
所有输出 on/off 一致；完整产物与脚本哈希已校验。
官方 quick 35/36，PyTorch DLL 初始化失败；仓库全量 lint 有既存问题，未修改腾讯代码。

### 消融数据
固定 seed=0、随机初始化、FP32、320、batch=1，3 warmup +20 对 AB/BA 推理前向。
MoE +1.09%、MoT +1.17%、Latent -1.55%；短测负数是噪声，不声称加速。原始样本见 routing_snapshot.json。
这不是训练 benchmark，不声称达到 P1；多 seed 与置信区间留待正式训练对照。

### 已知局限
仅锁定三份 YAML；MoA/MoLoRA unsupported；MoE eval aux 未发布；未实现实时面板、token overlay、视频、训练减速验收。
随机初始化单图只能证明链路。GPU/DLL 环境故障和全仓检查失败记录在 VALIDATION.md。
