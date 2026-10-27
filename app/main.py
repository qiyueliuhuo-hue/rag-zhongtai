from get_comment import text_jiansuo, sousuo, search, vector_search
from graph_retrieval import graph_search
from dotenv import load_dotenv
from openai import OpenAI
import os
from query_cache import TwoLevelCache  # ← 加这行
from FlagEmbedding import BGEM3FlagModel

load_dotenv()  # 加载环境变量


model = BGEM3FlagModel(os.getenv("BGE_M3_PATH", r"../bge-m3"), use_fp16=True)
# model = BGEM3FlagModel(r"../bge-m3", use_fp16=True)
cache = TwoLevelCache(model=model, threshold=0.9, max_size=200)


def rag_search(query_text, user_id="default"):
    """
    双路检索，精排检索和图谱检索
    """
    # 路径A 向量检索加精排
    collection_name = f"tenant_{user_id}_knowledge"  # 替换为你的 Milvus 集合名称
    query_dense, query_sparse = text_jiansuo(query_text)
    dense_req, sparse_req = search(query_dense, query_sparse)
    top3 = vector_search(
        dense_req, sparse_req, query_dense, query_text, collection_name
    )

    # 路径B 图谱检索
    graph_results = graph_search(query_text, user_id)

    # 判断都不存在的时候返回不存在
    if not top3 and not graph_results:
        return f"暂无该用户的知识库，请在上传页面上传文档，当前用户ID：{user_id}"

    # 合并结果
    full_context = "【向量检索精排结果】\n"
    for i, item in enumerate(top3):
        full_context += (
            f"[{i+1}(来源：{item['source']}，相关度:{item['score']:.4f})]\n"
            f"{item['text'][:300]}\n\n"
        )

    if graph_results:
        full_context += "【图谱检索结果】\n"
        for i, fact in enumerate(graph_results):
            full_context += (
                f"[{i+1}]{fact['from']}--{fact['relation']}--{fact['to']}\n\n"
            )
    print("\n" + "=" * 50)
    print(full_context)
    print("\n" + "=" * 50)
    return full_context


def generate_answer(query, context):
    """
    用 DeepSeek 生成最终回答
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )
    prompt = f"""
        你是一个专业的RAG检索助手。请根据以下资料回答用户问题。
        要求：
        1. 只使用资料中的信息，不要编造
        2. 回答要结构化、有条理
        3. 如果资料不足，请如实说明
        4. 回答中引用资料时，用 [来源:xxx] 标注出处
        资料：
        {context}

        用户问题：{query}
    """
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_answer_stream(query, context):
    """流式，逐块返回"""
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )
    prompt = f"""
    你是一个专业的RAG检索助手。请根据以下资料回答用户问题。
        要求：
        1. 只使用资料中的信息，不要编造
        2. 回答要结构化、有条理
        3. 如果资料不足，请如实说明
        4. 回答中引用资料时，用 [来源:xxx] 标注出处
        资料：
        {context}

        用户问题：{query}
    """
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,  # 流式
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def ask(query_text, user_id="default"):
    """带两级缓存的问答入口"""
    answer, level, sim = cache.get(query_text)

    if answer is not None:
        print(f"缓存命中 [{level}] 相似度:{sim:.3f} 命中率:{cache.hit_rate():.1%}")
        return answer

    # 未命中 → 走原有检索 + 生成链路
    context = rag_search(query_text, user_id=user_id)
    answer = generate_answer(query_text, context)
    cache.set(query_text, answer)
    print(f"缓存未命中 → 已写入 | 命中率:{cache.hit_rate():.1%}")
    return answer


if __name__ == "__main__":
    query = "范赛超会什么技能？"
    context = rag_search(query)
    answer = generate_answer(query, context)
    print(f"\n{'='*50}")
    print(f"最终回答：\n{answer}")
    print(f"{'='*50}")

    # TODO: 把 context 送给 LLM 生成最终回答
    # answer = llm.generate(query, context)
    # print(f"\n最终回答：{answer}")
