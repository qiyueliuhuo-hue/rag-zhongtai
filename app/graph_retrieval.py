from neo4j import GraphDatabase
from graph_extractor import extract_entities_and_relations
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def graph_search(quety_text, user_id="default"):
    """
    根据用户问题，从知识图谱中检索相关信息。
    1、从问题中提取关键实体。
    2、在图数据库中查找与这些实体相关的节点和关系。
    3、将结果格式化为文本列表返回。
    """

    kg_result = extract_entities_and_relations(quety_text)
    if not kg_result or not kg_result.get("entities"):
        print("未找到相关实体")
        return []
    entity_nemes = [entity["name"] for entity in kg_result["entities"]]
    print("找到的实体：", entity_nemes)

    # 图数据库查询
    cypher_query = """
    MATCH (n)-[r]->(m)
    WHERE n.name IN $entity_names AND n.user_id=$user_id
    RETURN n.name AS from_name, type(r) AS relation, m.name AS to_name
    LIMIT 20
    """
    retrieved_facts = []
    try:
        with driver.session() as session:
            result = session.run(
                cypher_query, entity_names=entity_nemes, user_id=user_id
            )
            for record in result:
                from_name = record["from_name"]
                relation = record["relation"]
                to_name = record["to_name"]
                retrieved_facts.append(
                    {
                        "from": from_name,
                        "relation": relation,
                        "to": to_name,
                    }
                )
    except Exception as e:
        print("查询失败：", e)
        return []

    print(f"图检索完成，共召回 {len(retrieved_facts)} 条事实。")
    return retrieved_facts


# --- 测试代码 ---
if __name__ == "__main__":
    test_query = "范赛超会什么技能？"
    results = graph_search(test_query)
    for res in results:
        print(f"- {res}")
