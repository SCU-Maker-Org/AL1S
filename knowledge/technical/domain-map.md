---
title: AL1S 技术 RAG 语料组织地图
domains: [sys, hpc, compile, distributed, db, storage, ai_workload, cloud, security]
product: AL1S
version: "1"
source_uri: local://knowledge/technical/domain-map.md
license: MIT
language: zh-CN
trust_level: 10
document_kind: organization_map
fact_corpus: false
index: false
---

# AL1S 技术 RAG 语料组织地图

> 本文档是语料建设导航，不是技术事实语料，也不应被当作产品行为、配置或最佳实践的权威来源。每个叶子主题都需要后续加入带来源、版本和许可信息的事实文档。

这张地图把 PostgreSQL、Cocoon Sandbox、大模型训练和大模型推理组织到 AL1S 的九个受控 domain。一个主题可以同时属于多个 domain。

## PostgreSQL

| 方向 | 受控 Domain | 待建设主题 |
| --- | --- | --- |
| DB | `db`, `storage` | 事务、索引、WAL、MVCC |
| Compile | `compile`, `db` | SQL 查询优化、代价模型、执行计划、执行器 |
| Sys | `sys`, `storage`, `db` | 文件系统、内存、并发、网络与数据库运行时的交互 |
| HPC | `hpc`, `compile`, `db` | 并行扫描、向量化执行、大规模分析 |

建议目录：`postgresql/transactions`、`postgresql/indexing`、`postgresql/wal`、`postgresql/mvcc`、`postgresql/query-planner`、`postgresql/executor` 和 `postgresql/performance`。

## Cocoon Sandbox

| 方向 | 受控 Domain | 待建设主题 |
| --- | --- | --- |
| Sys | `sys` | KVM、虚拟内存、UFFD、FUSE、virtio |
| Distributed Sys | `distributed`, `sys` | 节点控制、调度、资源状态 |
| Cloud | `cloud`, `distributed` | Kubernetes、CNI、镜像、快照编排 |
| Security | `security`, `sys` | 不可信代码隔离、权限边界、威胁模型 |
| DB/Storage | `db`, `storage` | 快照索引、镜像元数据、持久状态 |

建议目录：`cocoon-sandbox/virtualization`、`cocoon-sandbox/memory`、`cocoon-sandbox/io`、`cocoon-sandbox/control-plane`、`cocoon-sandbox/networking`、`cocoon-sandbox/isolation` 和 `cocoon-sandbox/state`。

## 大模型训练

| 方向 | 受控 Domain | 待建设主题 |
| --- | --- | --- |
| HPC | `hpc`, `ai_workload` | GPU 集群、数据/张量/流水线并行、集合通信 |
| Compile | `compile`, `ai_workload` | 算子融合、图编译、Kernel 生成 |
| Sys | `sys`, `ai_workload` | 驱动、调度、主机与设备内存管理 |
| Distributed Sys | `distributed`, `ai_workload` | 容错、检查点、节点协调 |
| DB/Storage | `db`, `storage`, `ai_workload` | 训练数据、模型权重、实验与作业元数据 |

建议目录：`llm-training/parallelism`、`llm-training/collectives`、`llm-training/compiler`、`llm-training/runtime`、`llm-training/fault-tolerance`、`llm-training/checkpointing` 和 `llm-training/data`。

## 大模型推理

| 方向 | 受控 Domain | 待建设主题 |
| --- | --- | --- |
| Sys | `sys`, `ai_workload` | GPU 内存管理、服务调度 |
| HPC | `hpc`, `ai_workload` | 矩阵计算、连续批处理、并行执行 |
| Compile | `compile`, `ai_workload` | 量化、算子融合、Kernel 优化 |
| DB-like | `db`, `storage`, `ai_workload` | KV Cache、前缀缓存、索引、淘汰与持久化 |
| Distributed Sys | `distributed`, `cloud`, `ai_workload` | 请求路由、扩缩容、Prefill/Decode 分离 |

建议目录：`llm-inference/memory`、`llm-inference/batching`、`llm-inference/quantization`、`llm-inference/kernels`、`llm-inference/kv-cache`、`llm-inference/routing`、`llm-inference/autoscaling` 和 `llm-inference/pd-disaggregation`。

## 横向能力矩阵

最终语料不应形成四个彼此孤立的产品岛。以下横向问题应通过同一术语、对比文档或交叉链接连接起来：

| 横向能力 | PostgreSQL | Cocoon Sandbox | 大模型训练 | 大模型推理 |
| --- | --- | --- | --- | --- |
| `sys` | 内存、并发、网络、I/O | KVM、UFFD、FUSE、virtio | 驱动、调度、内存 | GPU 内存、服务调度 |
| `hpc` | 并行与向量化执行 | 大规模节点资源视角 | GPU 并行与集合通信 | 矩阵计算与批处理 |
| `compile` | SQL 规划与执行 | 可选的策略/配置生成 | 图、算子与 Kernel 编译 | 量化、融合与 Kernel 优化 |
| `distributed` | 复制与分布式扩展的独立专题 | 控制面、调度、状态 | 容错、检查点、协调 | 路由、扩缩容、P/D 分离 |
| `db` / `storage` | 事务、WAL、索引、持久化 | 快照、镜像元数据、状态 | 数据、权重、元数据 | KV/前缀缓存、索引与淘汰 |
| `ai_workload` | 作为数据服务底座的交叉专题 | 作为隔离执行环境的交叉专题 | 训练工作负载本身 | 推理工作负载本身 |

## 建设顺序

1. 先为每个叶子主题选定官方文档、标准或原始论文，并记录版本和许可。
2. 再写边界明确的中文摘要，保留关键英文术语、约束、失败模式和原始链接。
3. 为容易混淆的主题加入版本差异和反例，例如 MVCC 可见性与锁语义、KV Cache 与数据库事务语义的区别。
4. 用真实问题建立检索评测集，检查语义召回、关键词召回、来源引用和跨 domain 问题。
5. 仅在语料和评测表明有明确瓶颈时，再评估更换嵌入模型。

总目标是覆盖 `Sys + HPC + Compile + Distributed + DB/Storage + AI Workload` 的交叉知识，同时保留可追溯来源，不把这张分类地图本身当作知识答案。
