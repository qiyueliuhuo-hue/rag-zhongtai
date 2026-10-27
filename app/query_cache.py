"""
Author: qiyueliuhuo-hue 19211420901@163.com
Date: 2026-08-07 11:05:09
LastEditors: qiyueliuhuo-hue 19211420901@163.com
LastEditTime: 2026-08-09 22:15:47
FilePath: \pypachong\123\RAG中台\app\query_cache.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

"""语义缓存层"""

import hashlib
import numpy as np
from collections import OrderedDict

# from query_cache import TwoLevelCache
# from embedding import get_embedding_model

# cache = TwoLevelCache(model=get_embedding_model(), threshold=0.9, max_size=200)


class TwoLevelCache:
    """两层缓存"""

    def __init__(self, model, threshold=0.9, max_size=200):
        self.model = model
        self.threshold = threshold
        self.max_size = max_size
        # L1精确匹配
        self.L1_cache = OrderedDict()
        # L2语义匹配
        self.L2_cache = OrderedDict()
        # L1历史匹配次数
        self.L1_hist = 0
        # L2历史匹配次数
        self.L2_hist = 0
        # 总匹配次数
        self.total = 0

    def get(self, query_text):
        self.total += 1
        # 获取文本输入哈希值
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        # L1精确匹配
        if query_hash in self.L1_cache:
            # 如果命中，更新L1历史匹配次数
            self.L1_hist += 1
            # 返回L1缓存中的结果
            return self.L1_cache[query_hash], "L1", 1.0
        # L2语义匹配
        query_vec = self.model.encode(
            [query_text], return_dense=True, return_sparse=False
        )["dense_vecs"][0]
        for _, (cache_vec, answer) in self.L2_cache.items():
            sim = np.dot(query_vec, cache_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cache_vec)
            )
            if sim >= self.threshold:
                self.L2_hist += 1
                return answer, "L2", sim
        # 如果都没有命中，返回None
        return None, "None", 0.0

    def set(self, query_text, answer, query_vec=None):
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        self.L1_cache[query_hash] = answer
        if query_vec is None:
            query_vec = self.model.encode(
                [query_text], return_dense=True, return_sparse=False
            )["dense_vecs"][0]
            self.L2_cache[query_hash] = (query_vec, answer)
        # 如果缓存满了，删除最旧的缓存
        if len(self.L1_cache) > self.max_size:
            self.L1_cache.popitem(last=False)
        if len(self.L2_cache) > self.max_size:
            self.L2_cache.popitem(last=False)
