"""
Author: qiyueliuhuo-hue 19211420901@163.com
Date: 2026-07-21 21:42:46
LastEditors: qiyueliuhuo-hue 19211420901@163.com
LastEditTime: 2026-08-10 09:19:21
FilePath: \pypachong\123\RAG中台\app\graph_extractor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
model = os.getenv("LLM_MODEL")

# 提示词
prompt = """
你是一个知识图谱提取专家，请从一下文本提取实体和关系。
实体类型：Person(人物)、Organization(组织)、Skill(技能)、Location(地点)、Product(产品)、Concept(概念)

严格按以下 JSON 格式返回，不要返回其他内容：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型"}}
  ],
  "relationships": [
    {{"from": "实体A", "from_type": "类型A", "relation": "关系", "to": "实体B", "to_type": "类型B"}}
  ]
}}
如果文本为空、过短或无法提取有效信息，直接返回：
{{"entities": [], "relationships": []}}
文本：
{text}
"""


# 提取实体和关系的函数
def extract_entities_and_relations(text):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt.format(text=text)},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()

    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n?```", content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"JSON 解析失败: {content[:300]}")
        return None
