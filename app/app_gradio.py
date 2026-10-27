"""RAG 中台 Gradio 前端 —— 支持文件上传 + 对话"""

import gradio as gr
import warnings
import traceback

# md5
import hashlib

warnings.filterwarnings("ignore")

print("=" * 50)
print("  RAG 中台 启动中...")
print("=" * 50)


def chat_fn(message, history, user_id):
    """对话"""
    if not message.strip():
        yield "请输入问题"
        return

    from main import rag_search, generate_answer_stream, cache

    cached, level, sim = cache.get(message)
    if cached is not None:
        yield f"缓存命中：{cached}"
        print(f"缓存命中：{cached}")
        return
    try:
        context = rag_search(message, user_id=user_id)
    except Exception as e:
        yield f"⚠ 检索失败：{e}"
        return
    response = ""
    try:
        for chunk in generate_answer_stream(message, context):
            response += chunk
            yield response
        if response:
            cache.set(message, response)
    except Exception as e:
        yield response + f"\n\n⚠ 生成中断：{e}"


def upload_file(file, user_id):
    """上传文件 → 写入 Milvus + Neo4j"""
    if file is None:
        return "⚠ 请先选择文件"
    try:
        from pipeline import run_vector_pipeline
        from graph_pipeline import build_graph

        collection_name = f"tenant_{user_id}_knowledge"
        n = run_vector_pipeline(file.name, collection_name=collection_name)
        build_graph(file.name, user_id=user_id)
        return f"✅ 完成！向量 {n} 块 + 图谱已入库"
    except Exception as e:
        traceback.print_exc()  # 打印异常堆栈
        return f"❌ 失败：{e}"


# ── Blocks 布局 ──
with gr.Blocks(title="RAG 中台") as demo:
    gr.Markdown("# RAG 中台")
    gr.Markdown("向量检索 + 知识图谱 双路融合")
    # 获取用户输入的id

    user_id = gr.Textbox(label="用户 ID", placeholder="用户id", interactive=True)

    with gr.Tab("📁 上传文档"):
        file_input = gr.File(
            label="选择 PDF / DOCX / TXT", file_types=[".pdf", ".docx", ".txt", ".md"]
        )
        upload_btn = gr.Button("上传并入库")
        upload_msg = gr.Textbox(label="状态", interactive=False)
        upload_btn.click(
            fn=lambda file, uid: upload_file(
                file, hashlib.md5(uid.encode()).hexdigest()[:8]
            ),
            inputs=[file_input, user_id],
            outputs=upload_msg,
        )

    with gr.Tab("💬 对话"):
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="输入问题，回车发送...", container=False)
        clear = gr.Button("清空对话")

        def respond(message, history, user_id):
            history = history or []
            # 新版返回值
            # history.append({"role": "user", "content": message})
            for chunk in chat_fn(
                message, history, hashlib.md5(user_id.encode()).hexdigest()[:8]
            ):
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": chunk},
                ]

        msg.submit(fn=respond, inputs=[msg, chatbot, user_id], outputs=chatbot)
        clear.click(fn=lambda: [], outputs=chatbot)

if __name__ == "__main__":
    print("\n  服务启动: http://localhost:8000\n")
    demo.launch(server_name="0.0.0.0", server_port=8000)
