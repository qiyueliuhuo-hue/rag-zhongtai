import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    def __init__(self, model_path, use_fp16=True):
        # 自动判断设备：有Gpu用Gpu，否则用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"精排模型将加载到设备：{self.device}")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 开始打分

        if use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    def rerank(self, query, documents, top_n=3):
        # 将问题和文档组合成模型需要的输入格式
        pairs = [[query, doc] for doc in documents]
        with torch.no_grad():  # 推理阶段，关闭梯度计算，节省内存和时间
            # 对输入进行分词和编码
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            # 关键一步：将输入数据移动到指定设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 模型推理，计算得分
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

            # 将得分和文档配对，并按得分排序
            ranked = sorted(zip(scores, documents), reverse=True)
            return ranked[:top_n]


# reranker = Reranker(model_path=r"../bge-reranker-v2-m3", use_fp16=True)
import os

reranker = Reranker(
    model_path=os.getenv("RERANKER_PATH", r"../bge-reranker-v2-m3"), use_fp16=True
)

if __name__ == "__main__":
    reranker = Reranker(model_path=r"123/RAG中台/bge-reranker-v2-m3", use_fp16=True)
    query_text = "范赛超会什么技能？"
    docs = ["范赛超擅长编程和数据分析", "范赛超擅长足球和篮球", "范赛超擅长摄影和音乐"]
    top3 = reranker.rerank(query_text, docs, top_n=3)
    for i, (score, text) in enumerate(top3):
        print(f"[{i+1}] {score:.4f} | {text}")
