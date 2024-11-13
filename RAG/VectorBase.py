import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BaseEmbeddings:
    """
    Base class for embeddings
    """

    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

# 使用MindNLP的SentenceTransformer实现
class MindNLPEmbedding(BaseEmbeddings):
    """
    class for MindNLP embeddings
    """
    def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model(path)

    def get_embedding(self, text: str):
        sentence_embedding = self._model.encode([text], normalize_embeddings=True)
        return sentence_embedding

    def load_model(self, path: str):
        from mindnlp.sentence import SentenceTransformer
        model = SentenceTransformer(path, mirror="huggingface")
        return model

    @classmethod
    def cosine_similarity(cls, sentence_embedding_1, sentence_embedding_2):
        """
        calculate cosine similarity between two vectors
        """
        similarity = sentence_embedding_1 @ sentence_embedding_2.T
        return similarity

# 使用MindNLP的Transformers实现
# class MindNLPEmbedding(BaseEmbeddings):
#
#     def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5', is_api: bool = False) -> None:
#         super().__init__(path, is_api)
#         self._model, self._tokenizer = self.load_model(path)
#
#     def get_embedding(self, text: str) -> List[float]:
#         encoded_input = self._tokenizer([text], padding=True, truncation=True, return_tensors="ms")
#         model_output = self._model(**encoded_input)
#         sentence_embeddings = model_output[0][:, 0]
#         norm = sentence_embeddings.norm(ord=2, dim=1, keepdim=True)
#         normalized_sentence_embeddings = sentence_embeddings / norm
#         return normalized_sentence_embeddings[0].tolist()
#
#     def load_model(self, path: str):
#         from mindnlp.transformers import AutoModel, AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(path)
#         model = AutoModel.from_pretrained(path)
#         return model, tokenizer

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError

