# RAG 中台

向量检索 + 知识图谱双路融合的检索增强生成（RAG）平台，带 Gradio 可视化界面，支持文档上传入库与多轮问答。

## 功能特性

- **双路检索**：Milvus 向量库（稠密 + 稀疏混合检索）与 Neo4j 知识图谱并行召回，结果合并后送入大模型。
- **知识图谱自动构建**：LLM 从文档中抽取实体与关系，8 线程并发写入 Neo4j。
- **精排**：BGE-Reranker-v2-m3 对候选片段重排序。
- **两级缓存**：LRU + 语义相似度判重（阈值 0.9），命中直接返回，降低重复调用的模型开销。
- **多租户隔离**：按用户 ID 的 MD5 划分 Milvus collection 与图谱命名空间，互相不可见。
- **流式输出**：问答结果逐块返回。

## 技术栈

| 组件 | 选型 |
| --- | --- |
| 界面 | Gradio |
| 向量库 | Milvus 2.4（依赖 etcd + MinIO） |
| 图数据库 | Neo4j 5.20（含 APOC） |
| Embedding | BGE-M3 |
| 精排 | BGE-Reranker-v2-m3 |
| LLM | DeepSeek（问答）／Ollama + Qwen2.5（图谱抽取） |

## 目录结构

```
.
├── app/
│   ├── app_gradio.py           # Gradio 界面入口
│   ├── main.py                 # 双路检索编排 + 答案生成
│   ├── pipeline.py             # 向量入库流水线
│   ├── graph_pipeline.py       # 图谱构建流水线
│   ├── document_loader_xiugai.py  # 文档解析（PDF/DOCX/TXT/MD）
│   ├── embedding.py            # 切分与向量化
│   ├── vector_store.py         # Milvus 读写
│   ├── graph_extractor.py      # 实体关系抽取
│   ├── graph_store_xieru.py    # Neo4j 写入
│   ├── graph_retrieval.py      # Neo4j 检索
│   ├── get_comment.py          # 向量混合检索与融合
│   ├── reranker.py             # 精排
│   └── query_cache.py          # 两级缓存
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 准备模型文件

模型权重体积较大（合计约 6.5 GB），未纳入版本管理，需自行下载：

```bash
mkdir -p models
huggingface-cli download BAAI/bge-m3        --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

国内网络可使用镜像站 `HF_ENDPOINT=https://hf-mirror.com`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 `DEEPSEEK_API_KEY`（申请地址：<https://platform.deepseek.com>）。
若图谱抽取走本地 Ollama，需先拉起模型：`ollama pull qwen2.5:7b`。

> `.env` 已被 `.gitignore` 排除，请勿提交真实密钥。

### 3. 启动依赖与平台（推荐 Docker）

```bash
docker compose up -d        # etcd / MinIO / Milvus / Neo4j / rag-app
```

启动后：

- RAG 平台界面：<http://localhost:8000>
- Neo4j Browser：<http://localhost:7474>（账号 `neo4j`，密码见 compose）

### 4. 不用 Docker 的本地运行方式

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app/app_gradio.py
```

此方式需自行启动 Milvus 与 Neo4j，并把 `.env` 中的连接地址改为本机端口
（`MILVUS_URI=http://localhost:19530`、`NEO4J_URI=bolt://localhost:7687`）。

## 使用流程

1. 打开界面，填入用户 ID。
2. 「📁 上传文档」页选择 PDF / DOCX / TXT / MD，点击上传并入库。向量分块与图谱抽取串行执行，大文档耗时较长（500 KB PDF 约 5–8 分钟）。
3. 「💬 对话」页提问，回答附带 `[来源:xxx]` 出处标注。

## 已知限制

- 图谱抽取逐块调用 LLM，长文档耗时与 token 消耗都较高。
- Milvus collection 名按用户 ID 动态创建，首次查询不存在的用户会返回空知识库提示。
- 缓存为进程内内存缓存，重启后失效。
- `docker-compose.yml` 中 Neo4j 与 MinIO 使用了默认口令，**仅供本地开发，请勿直接用于生产环境**。

## License

个人学习与演示项目。
