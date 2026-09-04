# Risks, limitations, and fallback plan

## Known limitations

1. COCO8 只有 4 张训练图和 4 张验证图，且只执行 1 epoch；结果不能用于判断精度或收敛性。
2. 首次 epoch 出现一次 Gradient NaN/Inf。框架的健康检查点恢复与 epoch 重放成功，但正式训练前仍需验证其复现频率和根因。
3. 完整日志对应的最终验证未能绘制混淆矩阵，错误是约 23 MiB NumPy 数组分配失败。数值验证、预测和权重保存完成；首次独立 smoke 已产生混淆矩阵证据。
4. 当前只在 Windows、单张 RTX 3060 Laptop GPU、`workers=0` 上验证；尚未覆盖 Linux、多卡、其他 CUDA/PyTorch 组合。
5. 当前测试仅验证基础 Master/MoE 检测训练链路，没有验证 E3 正式目标中的路由 snapshot、专家负载、路由熵或性能开销。

## Risk triggers

- 如果 NaN/Inf 在短程重复测试中再次出现或超过自动恢复次数，则不启动长程训练。
- 如果显存不足，则依次降低 batch、输入尺寸，并保持其他变量不变记录降级配置。
- 如果 Windows dataloader 多进程不稳定，则继续使用 `workers=0`；Linux 环境再单独测试多 worker。
- 如果绘图阶段内存分配失败，则保留 CSV 和日志作为主要数值证据，并在资源允许时单独离线重绘。

## Fallback plan

P0 保证单卡、COCO8、单 epoch 的训练—验证—保存链路稳定可复现。P1 再增加多次重复运行、NaN 定位和 E3 路由统计；实时面板、空间叠图及正式开销结论后置，不阻塞基础准入。
