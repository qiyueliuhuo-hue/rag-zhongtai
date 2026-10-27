"""
Author: qiyueliuhuo-hue 19211420901@163.com
Date: 2026-07-21 21:31:15
LastEditors: qiyueliuhuo-hue 19211420901@163.com
LastEditTime: 2026-07-30 10:48:08
FilePath: \pypachong\123\RAG中台\app\graph_pipeline.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from document_loader_xiugai import load_document
from embedding import split_text
from graph_extractor import extract_entities_and_relations
from graph_store_xieru import insert_entity, insert_relationship
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_graph(file_path, user_id="default", max_workers=8):
    """
    从文档构建知识图谱（多线程并发版）
    max_workers=5 约5倍提速，500KB PDF 约5-8分钟跑完
    """
    # 加载文档
    text_list = load_document(file_path)
    # 分割文本
    chunks = split_text(text_list)
    # 逐块抽取加写入
    print(f"开始构建知识图谱，共 {len(chunks)} 块，{max_workers} 线程并发...")
    all_enttities = {}  # 字典去重
    all_relationships = []
    seen_rels = set()  # 关系去重

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_entities_and_relations, chunk): i
            for i, chunk in enumerate(chunks)
            if chunk.split() and chunk.strip()  # 过滤空块  # 过滤空行
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  第 {i+1} 块抽取失败，跳过：{e}")
                continue
            if result is None:
                print(f"  第 {i+1} 块抽取失败，跳过")
                continue
            if not isinstance(result, dict):
                print(f"  第 {i+1} 块返回类型错误：{type(result).__name__}，跳过")
                continue
            for entity in result.get("entities", []):
                name = entity.get("name")
                if not name:
                    continue
                if name not in all_enttities:
                    all_enttities[name] = entity.get("type", "Concept")

            for rel in result.get("relationships", []):
                frm = rel.get("from")
                to = rel.get("to")
                if not frm or not to:
                    continue
                key = (frm, rel.get("relation", ""), to)
                if key not in seen_rels:
                    seen_rels.add(key)
                    all_relationships.append(rel)

    # 写入实体
    for name, etype in all_enttities.items():
        insert_entity(name, etype, user_id=user_id)
    # 写入关系
    for rel in all_relationships:
        insert_relationship(
            rel.get("from", ""),
            rel.get("from_type", "Concept"),
            rel.get("relation", "RELATED_TO"),
            rel.get("to", ""),
            rel.get("to_type", "Concept"),
            user_id=user_id,
        )
    print("知识图谱构建完成")


if __name__ == "__main__":
    file_path = r"123/RAG中台/范赛超_AI.pdf"
    build_graph(file_path)
