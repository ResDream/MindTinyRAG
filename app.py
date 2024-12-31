# 在服务器上使用gradio准备：
# mv frpc_linux_aarch64_v0.2 /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/gradio
# cd /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/gradio
# chmod +x frpc_linux_aarch64_v0.2

import gradio as gr
from typing import List, Dict
import os
import PyPDF2
from mindspore import Tensor
from copy import copy
import numpy as np
from tqdm import tqdm
import json
import markdown
from bs4 import BeautifulSoup
import re
import tiktoken
from datetime import datetime

# 初始化 tiktoken 编码器
enc = tiktoken.get_encoding("cl100k_base")

class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()
    
    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # 避免跨单词分割
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                # 处理最后一个块
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                chunk_text.append(curr_chunk)
                
            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:]+line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")


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
        model = SentenceTransformer(path)
        return model

    @classmethod
    def cosine_similarity(cls, sentence_embedding_1, sentence_embedding_2):
        """
        calculate similarity between two vectors
        """
        similarity = sentence_embedding_1 @ sentence_embedding_2.T
        return similarity
    
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
        if isinstance(EmbeddingModel, MindNLPEmbedding):
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

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
        return tokenizer, model
    
PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，请输出我不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    MindNLP_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，请输出我不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)

class MindNLPChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str = '') -> str:
        prompt = PROMPT_TEMPLATE['MindNLP_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history, max_length=512)
        return response

    def load_model(self):
        import mindspore
        from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, mirror="huggingface")
        self.model = AutoModelForCausalLM.from_pretrained(self.path, ms_dtype=mindspore.float16, mirror="huggingface")

# 这里你可以定义你的 RAG 检索和回复生成函数
def rag_retrieval(query: str, embedding_model: 'MindNLPEmbedding', vector_store: 'VectorStore') -> List[Dict]:
    """
    实现 RAG 检索逻辑
    :param query: 用户查询
    :param documents: 上传的文档内容列表
    :param embedding_model: 嵌入模型
    :param vector_store: 向量存储
    :return: 检索结果，格式为 [{"content": "相关文本", "score": 相关性分数}, ...]
    """
    # 将文档内容存储到 VectorStore 中
    vector_store.get_vector(embedding_model)
    vector_store.persist(path='storage')
    
    # 查询最相关的文档
    retrieved_documents = vector_store.query(query, embedding_model, k=1)
    
    # 返回检索结果
    retrieved_results = [{"content": doc, "score": 1.0} for doc in retrieved_documents]
    print(retrieved_results)
    return retrieved_results

def generate_response(query: str, retrieved_results: List[Dict], chat_model: 'MindNLPChat') -> str:
    """
    实现回复生成逻辑
    :param query: 用户查询
    :param retrieved_results: RAG 检索结果
    :param chat_model: 聊天模型
    :return: 生成的回复文本
    """
    # 将检索到的内容拼接为上下文
    context = "\n".join([result["content"] for result in retrieved_results])
    
    # 使用聊天模型生成回复
    response = chat_model.chat(query, [], context)
    return response

# 处理上传的文件并提取文本内容
def process_uploaded_files(files: List[str]) -> None:
    """
    将上传的文件保存到 ./data/当前日期和时间 目录下。
    :param files: 上传的文件路径列表
    """
    # 创建保存目录，格式为 YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("./data", current_time)
    os.makedirs(save_dir, exist_ok=True)

    for file_path in files:
        # 获取文件名
        file_name = os.path.basename(file_path)
        # 构建保存路径
        save_path = os.path.join(save_dir, file_name)
        # 保存文件
        with open(file_path, 'rb') as src_file, open(save_path, 'wb') as dst_file:
            dst_file.write(src_file.read())

    documents = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
    return documents

# Gradio 应用的主函数
def rag_app(query: str, files: List[str]) -> str:
    """
    Gradio 应用的主函数
    :param query: 用户查询
    :param files: 上传的文件路径列表
    :return: 生成的回复
    """
    # 处理上传的文件
    documents = process_uploaded_files(files)
    
    # 初始化模型
    embedding_model = MindNLPEmbedding("BAAI/bge-base-zh-v1.5")
    vector_store = VectorStore(documents)
    chat_model = MindNLPChat(path='openbmb/MiniCPM-2B-dpo-bf16')
    
    # 调用 RAG 检索
    retrieved_results = rag_retrieval(query, embedding_model, vector_store)
    
    # 生成回复
    response = generate_response(query, retrieved_results, chat_model)
    
    return response

# 创建 Gradio 界面
interface = gr.Interface(
    fn=rag_app,
    inputs=[
        gr.Textbox(label="请输入你的问题"),
        gr.Files(label="上传文件（支持 .md, .txt, .pdf）")
    ],
    outputs=gr.Textbox(label="生成的回复"),
    title="RAG 应用",
    description="上传文件并提问，系统将基于文件内容生成回复。"
)

# 启动 Gradio 应用
interface.launch(share=True)