from pymilvus import MilvusClient, DataType

# print("开始向Milvus存储数据...")
# client = MilvusClient(uri="http://localhost:19530", token="")
import os
client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"), token="")

def insert_to_milvus(
    dense_vecs,
    clean_sparse_vecs,
    texts,
    collection_name="mid_rag_data",
    source_name="unknown",
):

    # 创建集合
    if not client.has_collection(collection_name):
        print(f"集合不存在，创建集合：{collection_name}")
        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vecs", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vecs", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_id", DataType.INT64)
        client.create_collection(collection_name=collection_name, schema=schema)
    index_param = client.prepare_index_params()
    index_param.add_index(
        field_name="dense_vecs",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_param.add_index(
        field_name="sparse_vecs",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    data = [
        {
            "text": texts[i],
            "dense_vecs": dense_vecs[i],
            "sparse_vecs": clean_sparse_vecs[i],
            "source": source_name,
            "chunk_id": i,
        }
        for i in range(len(texts))
    ]
    client.insert(collection_name=collection_name, data=data)
    client.create_index(collection_name=collection_name, index_params=index_param)
    client.load_collection(collection_name=collection_name)
    print(f"集合创建成功：{collection_name}")


print("存储完成")
