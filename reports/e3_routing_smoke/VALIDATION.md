# 本机验证记录（2026-09-05）

## 已完成验证
- 最终入口：`run_smoke.cmd --output reports/e3_routing_smoke/results/verified-cpu-v4-annotated-20260905`。
- 真实 COCO8：`000000000036.jpg`；imgsz=320、FP32、batch=1、seed=0、CPU。
- MoE 6 层、MoT 4 层、Latent 3 层，13 条 JSONL，三族 passed。
- 所有模型 on/off 输出有限且一致（rtol=1e-5，atol=1e-6），hooks 已移除。
- pytest：15 passed，覆盖非法数据、stale snapshot、异常清理、MoE top-k 语义、YAML 覆盖和 manifest。
- 本次两个 Python 文件 Ruff check、format check、codespell 均通过。
- 最终运行的 14 个产物哈希验证通过；脚本 SHA-256 与 environment.json 一致。
- 对锁定腾讯基线做差异检查，没有修改/删除任何腾讯原文件。

## 最终 CPU on/off 数字
每组预热 3 次，20 对 AB/BA 交替；保留全部原始计时。

| 族 | off 中位数 ms | on 中位数 ms | 中位数比值减速 |
| --- | ---: | ---: | ---: |
| MoE | 84.013 | 88.987 | +5.92% |
| MoT | 101.805 | 100.978 | -0.81% |
| Latent | 91.539 | 89.231 | -2.52% |

负数表示本次计时噪声/运行波动，不能声称采集让模型加速。开发运行中的数字也有明显变化；该短测没有置信区间，不足以判断稳定性能，更不是 P1 训练门槛。

## 未通过和环境问题
- 首次 GPU 尝试：CUDA 内存分配失败，已终止，未得到有效 GPU 结果。
- 随后的并行启动尝试：WinError 1455（分页文件不足）；顺序 CPU 运行成功。
- 官方 Agent quick 两次均为 35/36。`pipeline_profile_dry_run` 在导入 PyTorch 的 shm.dll 时出现 WinError 1114，未完成环境总体验收。见 env/agent-quick.log 和 env/agent-quick-failure.json。
- 仓库全量 Ruff 检查当时报告 2692 项，其中包含当时新增脚本的一项格式相关提示；后续新增文件均已通过独立检查。原仓库有大量既存问题，未批量修改。
- 全量 format check 当时有 4 个文件需格式化；新增文件后续已单独修正。全仓 codespell 存在文档/词汇报告。没有宣称全仓 lint 通过。

## 证据位置
- [最终结果](results/verified-cpu-v4-annotated-20260905/summary.json)
- [带数值的总览图](results/verified-cpu-v4-annotated-20260905/routing_snapshot.png)
- [MoE 图](results/verified-cpu-v4-annotated-20260905/moe_expert_usage.png)
- [MoT 图](results/verified-cpu-v4-annotated-20260905/mot_expert_usage.png)
- [Latent 图](results/verified-cpu-v4-annotated-20260905/latent_expert_usage.png)
- [逐层记录](results/verified-cpu-v4-annotated-20260905/routing_snapshot.jsonl)
- [原始计时与一致性](results/verified-cpu-v4-annotated-20260905/routing_snapshot.json)
- [完整日志](results/verified-cpu-v4-annotated-20260905/full.log)
- [依赖版本](env/dependencies.json)

工作区当前 Git HEAD 仍为旧 smoke 提交；environment.json 的 dirty=true 与 source_sha256 如实表示本次未提交代码。上传分支中的脚本内容应与此 SHA-256 对得上。

