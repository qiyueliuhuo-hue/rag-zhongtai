from FlagEmbedding import BGEM3FlagModel
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient
from reranker import reranker


# 定义一个函数，用于文本检索
def text_jiansuo(text: str) -> str:
    model = BGEM3FlagModel(model_name_or_path=r"../bge-m3", use_fp16=True)
    query_embedding = model.encode(
        [text], return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    query_dense = query_embedding["dense_vecs"][0].tolist()
    query_sparse = query_embedding["lexical_weights"][0]
    return query_dense, query_sparse


# 连接Milvus数据库
# client = MilvusClient(uri="http://localhost:19530", token="")
import os

client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"), token="")


def search(query_dense, query_sparse):
    # 构建搜索请求
    dense_req = AnnSearchRequest(
        data=[query_dense],
        anns_field="dense_vecs",
        param={"metric_type": "COSINE", "params": {"ef": 100}},
        limit=10,
    )
    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse_vecs",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=10,
    )
    return dense_req, sparse_req


# 执行搜索
def sousuo(
    dense_req, sparse_req, query_dense, collection_name="tenant_default_knowledge"
):
    """
    执行混合搜索,并返回一个列表
    """
    #   检查集合是否存在
    if not client.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return []
    # print("开始混合搜索...")
    # 混合搜索
    try:
        result = client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=10,
            output_fields=["text", "source", "chunk_id"],
        )
        documents = []
        for hits in result:
            for hit in hits:
                documents.append(
                    {
                        "text": hit["entity"]["text"],
                        "source": hit["entity"].get("source", "未知"),
                        "chunk_id": hit["entity"].get("chunk_id", ""),
                    }
                )
        return documents
    except Exception as e:
        print(f"搜索失败：{e}")
        print("进行降级搜索")
        result = client.search(
            collection_name=collection_name,
            data=[query_dense],
            limit=10,
            output_fields=["text"],
        )
        return result


def vector_search(
    dense_req,
    sparse_req,
    query_dense,
    query_text,
    collection_name="tenant_default_knowledge",
):
    """向量检索 + 精排，返回 [(score, text), ...]"""
    docs = sousuo(dense_req, sparse_req, query_dense, collection_name)
    if not docs:
        return []
    # print("开始向量检索...")
    print(f"  召回 {len(docs)} 条，精排中...")
    # return reranker.rerank(query_text, docs, top_n=3)
    # 提取纯文本给reranker打分

    texts = [d["text"] for d in docs]
    ranked = reranker.rerank(query_text, texts, top_n=3)

    #  # 把分数和原始字典（含 source）对应起来
    #  # 这里假设docs是一个包含字典的列表，每个字典包含text和source字段
    #  # 然后遍历ranked，找到每个text对应的原始字典，并把分数和source一起添加到results列表中
    results = []
    for score, text in ranked:
        for doc in docs:
            if doc["text"] == text:
                results.append({"score": score, "text": text, "source": doc["source"]})
                break
    return results


if __name__ == "__main__":
    # 先确认集合里有数据
    client = MilvusClient(uri="http://localhost:19530")
    count = client.query(
        collection_name="tenant_default_knowledge",
        filter="id > 0",
        output_fields=["count(*)"],
    )
    print(f"集合数据量: ...")

    # 用你文档里实际存在的内容提问
    query_text = "范赛超的简历里写了什么技能？"

    query_dense, query_sparse = text_jiansuo(query_text)
    dense_req, sparse_req = search(
        query_dense,
        query_sparse,
    )
    result = sousuo(dense_req, sparse_req, query_dense)

    # 打印结果
    for j, doc in enumerate(result):
        print(f"排名:{j+1}|来源:{doc['source']}")
        print(f"内容:{doc['text'][:150]}")
        print("---" * 20)
