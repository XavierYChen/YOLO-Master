# E3 字段字典 v2

JSON 根版本：`e3.routing_snapshot.v2`。JSONL 每行是一层一次 eval 观测；step=0、seed 与 family 随行记录。失败族留在 JSON 的 profiles 中，绝不生成伪造的成功行。

| 字段 | 类型/单位 | 来源与含义 |
| --- | --- | --- |
| layer_name / module_type / family | string | 模型 named_modules 路径、生产模块类型、族 |
| num_experts / top_k | int | 路由专家总数/本次 top-k；共享专家不计入 |
| expert_usage | float[E] | MoE：最终 top-k 索引计数/总选择数；MoT、Latent：原生 expert_usage（平均混合概率） |
| usage_semantics | enum | topk_selection_share 或 mean_mixture_probability；不可混为硬命中率 |
| mean_router_probs | float[E] 或 null | MoE：最终稀疏权重散射后均值；MoT/Latent：原生 mean_router_probs。不是跨族统一的 pre-softmax 分数 |
| mean_topk_weight | float[K] 或 null | MoE：被选位置的平均权重，按 top-k 槽位排序，不按 expert id |
| mixing_weights_status | string | available 或 unsupported；不支持时保留 null |
| normalized_entropy | float [0,1] | H(expert_usage)/ln(E)，自然对数；这是平均负载的熵，不是逐 token 熵的平均 |
| normalized_gini | float [0,1] | 有限专家数修正：普通 Gini × E/(E−1)，与未修正的参考图数字不必相等 |
| dominant_share | float [0,1] | max(expert_usage) |
| dead_experts | int[] | 当前单图 share ≤0.01 的索引，仅提示低负载，不能认定训练后死专家 |
| tensor_shape | int[] | 生产模块输出特征形状，不是 token 路由图形状 |
| aux_loss.status | string | not_published_eval / configured_inactive_eval / not_configured |
| aux_loss.observed | float 或 null | 原生本次 eval 值；MoE eval 未发布记 null，不能把 null 写成 0 |
| aux_loss.balance_loss_coeff / router_z_loss_coeff | float | 生产模块暴露的系数；MoE 不保证覆盖内部 loss 配置，状态仍为 not_published_eval |
| source | string | router_topk_output（MoE）或 last_routing_snapshot（MoT/Latent） |
| source_snapshot_keys | string[] | 本次实际读取的原生/适配记录键，前者 source 明确区分 |
| usage_scope / global_usage_available | string / bool | 默认 rank_local / false；不承诺 DDP 全局汇总 |
| dispatch_policy / executed_experts | string/int 或 null | 原生有则保留，不能用概率非零项推断实际计算专家数 |

严格检查：向量维数、有限性、非负、和为 1（绝对误差 1e-4）；top-k 合法；原生 finite 状态；snapshot 新鲜度。拒绝 NaN/Inf、空或过期 snapshot，不清洗后冒充通过。
MoE 适配器目前只支持锁定 YAML 内的 OptimizedMOEImproved 最终路由接口，未知模块报 unsupported。
MoT 和 Latent 选择带 snapshot 的叶生产模块，不重复统计包装层。

JSON profiles 还包含输出一致性结果（rtol=1e-5，atol=1e-6）、hook 清理状态以及每对原始计时。
原始采样张量不长期保存。本版未实现空间 token overlay。
