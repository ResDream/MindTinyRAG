from typing import List
import numpy as np


class BaseReranker:
    """
    Base class for reranker
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        raise NotImplementedError


class MindNLPReranker(BaseReranker):
    """
    class for MindNLP reranker
    """

    def __init__(self, path: str = 'BAAI/bge-reranker-base') -> None:
        super().__init__(path)
        self._model= self.load_model(path)

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        query_embedding = self._model.encode(text, normalize_embeddings=True)
        sentences_embedding = self._model.encode(sentences=content, normalize_embeddings=True)
        similarity = query_embedding @ sentences_embedding.T
        # 获取按相似度排序后的索引
        ranked_indices = np.argsort(similarity)[::-1]  # 按相似度降序排序
        # 选择前 k 个最相关的候选内容
        top_k_sentences = [content[i] for i in ranked_indices[:k]]
        return top_k_sentences

        # pairs = [(text, c) for c in content]
        # with torch.no_grad():
        #     inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        #     inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        #     scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
        #     index = np.argsort(scores.tolist())[-k:][::-1]
        # return [content[i] for i in index]

    def load_model(self, path: str):
        from mindnlp.sentence import SentenceTransformer
        model = SentenceTransformer(path)
        return model