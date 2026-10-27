from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

print("开始分割文本...")
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)


# 把文本进行切割
def split_text(text_list):
    """
    将文本列表分割成多个块
    """
    all_chunks = []
    for text in text_list:
        if text and len(text.strip()) > 0:
            chunk = chunks.split_text(text)
            all_chunks.extend(chunk)
    return all_chunks


print("分割完成")

# 向量化
# 引入模型
print("开始向量化数据...")

# all_chunks = split_text(text_list)


def get_embedding(all_chunks):
    # model = BGEM3FlagModel(r"../bge-m3")
    import os

    model = BGEM3FlagModel(os.getenv("BGE_M3_PATH", r"../bge-m3"))
    all_embeddings = model.encode(
        all_chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    dense_vecs = all_embeddings["dense_vecs"].tolist()
    sparse_vecs = all_embeddings["lexical_weights"]
    # 清洗系数限量
    clean_sparse_vecs = []
    for sparse_dict in sparse_vecs:
        clean_dict = {int(k): float(v) for k, v in sparse_dict.items() if v > 0}
        clean_sparse_vecs.append(clean_dict)
    return dense_vecs, clean_sparse_vecs


print("向量化完成")
