"""
Author: qiyueliuhuo-hue 19211420901@163.com
Date: 2026-07-20 11:28:17
LastEditors: qiyueliuhuo-hue 19211420901@163.com
LastEditTime: 2026-07-28 23:37:57
FilePath: \pypachong\123\RAG中台\app\pipeline.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

"""
Author: qiyueliuhuo-hue 19211420901@163.com
Date: 2026-07-20 11:28:17
LastEditors: qiyueliuhuo-hue 19211420901@163.com
LastEditTime: 2026-07-23 17:50:29
FilePath: \pypachong\123\RAG中台\app\pipeline.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from document_loader_xiugai import load_document
from embedding import split_text, get_embedding
from vector_store import insert_to_milvus


def run_vector_pipeline(file_path, collection_name="tenant_default_knowledge"):
    print("开始加载文档...")
    text_list = load_document(file_path)
    chunks = split_text(text_list)
    dense_vecs, sparse_vecs = get_embedding(chunks)
    insert_to_milvus(
        dense_vecs,
        sparse_vecs,
        chunks,
        collection_name=collection_name,
        source_name=file_path,
    )
    return len(chunks)


if __name__ == "__main__":
    run_vector_pipeline(r"../范赛超_AI.pdf")
