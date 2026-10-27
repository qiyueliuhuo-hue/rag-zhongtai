from neo4j import GraphDatabase
import re

# 连接Neo4j数据库
# driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j123"))
import os
from dotenv import load_dotenv

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j123")),
)


# 清理名称，只保留字母、数字、中文、下划线
def clean_name(name):
    """清理名称，只保留字母、数字、中文、下划线"""
    if name is None or str(name).strip() == "":
        return "Unknown"
    name = str(name)

    # 把特殊字符替换成下划线
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "_", name)
    # 不能以数字开头
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned if cleaned else "Unknown"


# 插入实体
def insert_entity(name, entity_type, user_id="default"):
    name = clean_name(name)
    entity_type = clean_name(entity_type)
    with driver.session() as session:
        session.run(
            f"""
            MERGE (a:{entity_type} {{name: $name,user_id:$user_id}})
            """,
            name=name,
            user_id=user_id,
        )


# 插入关系
def insert_relationship(
    from_name, from_type, relation, to_name, to_type, user_id="default"
):
    from_name = clean_name(from_name)
    from_type = clean_name(from_type)
    relation = clean_name(relation)
    to_name = clean_name(to_name)
    to_type = clean_name(to_type)
    # relation = clean_name(relation)
    with driver.session() as session:
        session.run(
            f"""
            MERGE (a:{from_type} {{name: $from_name,user_id:$user_id}})
            MERGE (b:{to_type} {{name: $to_name,user_id:$user_id}})
            MERGE (a)-[r:{relation}]->(b)
            """,
            from_name=from_name,
            to_name=to_name,
            user_id=user_id,
        )
    print(f"  写入关系: {from_name} --[{relation}]-> {to_name}")


# 测试
if __name__ == "__main__":
    insert_entity("范赛超", "Person")
    insert_entity("Python", "Skill")
    insert_relationship("范赛超", "Person", "精通", "Python", "Skill")
