import os
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
from tqdm import tqdm
from RAG.Embeddings import BaseEmbeddings, HuggingFaceEmbedding


# MindNLP的SentenceTransformer实现
class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings):
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            # 将 numpy.ndarray 转换为列表
            vectors_list = [vector.tolist() for vector in self.vectors]
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(vectors_list, f)

    def load_vector(self, EmbeddingModel: BaseEmbeddings, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            vectors_list = json.load(f)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

        # 查询 EmbeddingModel 的类别
        if isinstance(EmbeddingModel, HuggingFaceEmbedding):
            # 将列表重新变为 numpy.ndarray
            self.vectors = [np.array(vector) for vector in vectors_list]
        else:
            self.vectors = vectors_list

    def get_similarity(self, vector1, vector2, EmbeddingModel: BaseEmbeddings):
        return EmbeddingModel.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1):
        # 获取查询字符串的嵌入向量
        query_vector = EmbeddingModel.get_embedding(query)

        # 计算查询向量与数据库中每个向量的相似度
        similarities = [self.get_similarity(query_vector, vector, EmbeddingModel) for vector in self.vectors]

        # 将相似度、向量和文档存储在一个列表中
        results = []
        for similarity, vector, document in zip(similarities, self.vectors, self.document):
            results.append({
                'similarity': similarity,
                'vector': vector,
                'document': document
            })

        # 按相似度从高到低排序
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # 获取最相似的 k 个文档
        top_k_documents = [result['document'] for result in results[:k]]

        return top_k_documents

# MindNLP的Transformers库实现
# class VectorStore:
#     def __init__(self, document: List[str] = ['']) -> None:
#         self.document = document
#
#     def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
#
#         self.vectors = []
#         for doc in tqdm(self.document, desc="Calculating embeddings"):
#             self.vectors.append(EmbeddingModel.get_embedding(doc))
#         return self.vectors
#
#     def persist(self, EmbeddingModel: BaseEmbeddings, path: str = 'storage'):
#         if not os.path.exists(path):
#             os.makedirs(path)
#         with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
#             json.dump(self.document, f, ensure_ascii=False)
#         if self.vectors:
#             with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
#                 json.dump(self.vectors, f)
#
#     def load_vector(self, path: str = 'storage'):
#         with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
#             self.vectors = json.load(f)
#         with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
#             self.document = json.load(f)
#
#     def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
#         return BaseEmbeddings.cosine_similarity(vector1, vector2)
#
#     def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
#         query_vector = EmbeddingModel.get_embedding(query)
#         result = np.array([self.get_similarity(query_vector, vector)
#                           for vector in self.vectors])
#         return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
