---
title: AL1S 技术 RAG 语料规范
domains: [sys, hpc, compile, distributed, db, storage, ai_workload, cloud, security]
product: AL1S
version: "1"
source_uri: local://knowledge/technical/README.md
license: MIT
language: zh-CN
trust_level: 20
document_kind: corpus_policy
index: false
---

# AL1S 技术 RAG 语料规范

此目录用于维护可追溯、可更新的计算机技术语料。它不是一个把网页全文随意堆在一起的目录：每份事实文档都应说明来源、适用版本和许可，并尽量只覆盖一个边界清晰的主题。

本文件和 [`domain-map.md`](domain-map.md) 是语料治理与组织说明，不是 PostgreSQL、Linux、Kubernetes 或 AI 系统的事实依据。事实回答应引用产品官方文档、标准、论文或经过审阅的内部资料。

## 受控 Domain

frontmatter 和 `--domains` 只接受以下标签：

| Domain | 范围示例 |
| --- | --- |
| `sys` | 操作系统、文件系统、虚拟内存、并发、驱动、网络栈 |
| `hpc` | GPU/CPU 并行、集合通信、向量化、批处理、大规模分析 |
| `compile` | 编译器、查询优化、执行计划、算子融合、Kernel 生成与优化 |
| `distributed` | 一致性、节点协调、调度、容错、路由、扩缩容 |
| `db` | 事务、MVCC、索引、查询执行、缓存和数据库语义 |
| `storage` | WAL、持久化、快照、对象/块存储、权重和元数据 |
| `ai_workload` | 大模型训练、推理、Serving、KV Cache 等 AI 工作负载 |
| `cloud` | Kubernetes、容器、CNI、镜像、云资源编排 |
| `security` | 隔离、沙箱、权限边界、威胁模型和不可信代码 |

标签可以多选。例如 PostgreSQL WAL 文档可以是 `[db, storage, sys]`，GPU 集合通信故障排查可以是 `[hpc, distributed, ai_workload]`。未知标签会使文档入库失败，不要临时创造 `network`、`gpu` 或 `postgres` 等平行标签；这些概念应放入 `product`、`subdomain` 或正文。

## 支持的文件

入库器读取 UTF-8 文本，并支持：

- Markdown、纯文本、reStructuredText 和 HTML。
- JSON、YAML、TOML、INI、XML 和常见配置文件。
- C/C++、CUDA、Rust、Go、Python、Java、JavaScript/TypeScript、SQL、Shell、LLVM IR 等常见源码。
- `Dockerfile`、`Containerfile`、`Makefile`、`CMakeLists.txt` 等构建与部署文件。

HTML 会先移除脚本、样式等非正文内容；Markdown 标题用于章节感知分块。二进制文件、非 UTF-8 文件、超过 `[rag].max_document_bytes` 的文件，以及解析后落在入库根目录外的符号链接会被拒绝。

## 元数据

Markdown 文档推荐使用 YAML frontmatter：

```markdown
---
title: PostgreSQL MVCC 可见性规则
source_id: postgresql_mvcc
domains: [db, sys, storage]
subdomain: concurrency_control
product: PostgreSQL
version: "18"
source_uri: https://www.postgresql.org/docs/18/mvcc.html
license: PostgreSQL License
language: zh-CN
trust_level: 95
retrieved_at: "2026-08-03"
---

# PostgreSQL MVCC 可见性规则

正文从这里开始。
```

字段约定：

- `title`：准确描述本文档，不使用“笔记 1”之类无意义名称。
- `source_id`：来源的稳定唯一 ID；不要把可变版本号或 URL 放进 ID。
- `domains`：一个或多个受控标签；`domain` 单值也可接受。
- `product`：产品、项目或标准名称，例如 `PostgreSQL`、`Linux`、`Kubernetes`。
- `version`：事实适用的产品版本、标准版本或 Git revision；数字版本应加引号。
- `source_uri`：原始资料的规范 URL、论文 DOI，或稳定的内部文档标识。
- `license`：允许本地保存和使用该内容的许可。填写许可名称不代表自动取得转载权。
- `language`：例如 `zh-CN`、`en`。
- `trust_level`：语料维护者给出的相对可信度；官方版本化文档通常高于个人笔记。
- `retrieved_at`：抓取或人工核对日期，便于安排更新。

长期更新的资料至少维护 `source_id`、`title`、`domains`、`product`、`version`、`source_uri` 和 `license`。没有明确许可时，优先写自己的摘要并链接原文，不要复制受版权保护的整篇资料。

## 目录组织

推荐以系统或产品为第一层，以主题为第二层：

```text
knowledge/technical/
├── postgresql/
│   ├── transactions/
│   ├── indexing/
│   └── query-planner/
├── cocoon-sandbox/
│   ├── virtualization/
│   └── isolation/
├── llm-training/
└── llm-inference/
```

[`domain-map.md`](domain-map.md) 给出了这四类主题如何映射到受控 domain。它只是待建设语料的导航图，不代替事实文档。

## 入库

从仓库根目录运行：

```bash
uv run python scripts/ingest_rag.py knowledge/technical \
  --domains sys,hpc,compile,distributed,db,storage,ai_workload,cloud,security
```

`--domains` 只在文档没有声明 `domain`/`domains` 时提供目录级默认标签；frontmatter 一旦声明便完整覆盖默认值。更推荐按子目录精确设置默认标签：

```bash
uv run python scripts/ingest_rag.py knowledge/technical/postgresql \
  --domains db,storage,sys,compile

uv run python scripts/ingest_rag.py knowledge/technical/llm-training \
  --domains ai_workload,hpc,distributed,compile,sys,storage
```

入库是基于稳定内容哈希的幂等操作：未变化文档不会重复生成分块；更新后的文档会替换原分块。对长期维护的来源应填写稳定且唯一的 `source_id`，这样 URL 或版本变化会更新原记录。目录入库还会清理同一 source root 下已删除、改名或设为 `index: false` 的旧文档；扫描存在失败或不安全文件时跳过清理，避免误删。

官方资料清单维护在 [`../sources.toml`](../sources.toml)：

```bash
uv run python scripts/fetch_rag_sources.py --all --json
uv run python scripts/ingest_rag.py data/rag_sources --strict --json
```

命令完成后会从 SQLite 中的事实数据重建向量索引。需要手动修复派生索引时，Telegram 管理命令为 `/rebuild_index`，不要删除 `data/bot.db`。

## 检索与引用

AL1S 对技术文档执行两路召回：

1. 使用 `Qwen/Qwen3-Embedding-0.6B` 和 FAISS 查找语义相近分块。
2. 使用 SQLite FTS5 查找关键词、专有名词和版本号匹配；中文连续文本额外建立 bigram。
3. 使用 RRF 融合候选，并按上下文预算注入模型。

模型看到的技术片段带有 `[来源 N]`、标题、章节路径、产品版本和来源 URI。回答中的来源标记便于追溯，但仍需人工检查原始资料是否支持结论，尤其是版本差异、安全配置和生产事故处理。

当前 Qwen3 0.6B 是新一代多语言嵌入模型，且项目固定了 revision；这里不把它替换成另一个模型。后续优化优先级应是语料质量、增量更新、负例评测、分块边界和检索指标，而不是只比较模型参数量。

## 入库检查清单

- 内容能追溯到明确的 `source_uri`，并记录适用 `version`。
- 许可允许当前保存和使用方式；没有许可时只存原创摘要和链接。
- 标题、代码块、表格和公式在 UTF-8 文本中可读。
- 文档没有密码、Token、Cookie、私钥、个人数据或内部生产凭据。
- 跨版本行为被拆成不同文档，或在正文中明确列出差异。
- 对安全、并发、一致性和性能结论写明前提，不把经验值包装成普遍事实。
