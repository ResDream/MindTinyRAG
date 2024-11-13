# MindTinyRAG
基于MindSpore的TinyRAG实现


# QuickStrat

```bash
git clone https://github.com/ResDream/MindTinyRAG
pip install -r requirements.txt
```

导入对应包

```python
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import MindNLPChat
from RAG.Embeddings import MindNLPEmbedding
from RAG.Reranker import MindNLPReranker
```

建立知识库

```python
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)  # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = MindNLPEmbedding("BAAI/bge-base-zh-v1.5")  # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库
```

读取知识库并完成RAG

```python
vector = VectorStore()
embedding = MindNLPEmbedding("BAAI/bge-base-zh-v1.5") # 创建EmbeddingModel
vector.load_vector(EmbeddingModel=embedding, path='./storage')  # 加载本地的数据库
question = 'git如何新建分支？'
content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
print(content)
chat = MindNLPChat(path='openbmb/MiniCPM-2B-dpo-bf16')
print(chat.chat(question, [], content))
```

使用Rerank

```python
embedding = MindNLPEmbedding("BAAI/bge-base-zh-v1.5") # 创建EmbeddingModel

# 创建RerankerModel
reranker = MindNLPReranker('BAAI/bge-reranker-base')

if have_created_db:
    # 保存数据库之后
    vector = VectorStore()
    vector.load_vector(EmbeddingModel=embedding, path='./storage')  # 加载本地的数据库
else:
    # 没有保存数据库
    docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)  # 获得data目录下的所有文件内容并分割
    vector = VectorStore(docs)
    vector.get_vector(EmbeddingModel=embedding)
    vector.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

question = '远程仓库的协作与贡献有哪些？'

# 从向量数据库中查询出最相似的3个文档
content = vector.query(question, EmbeddingModel=embedding, k=3)
print(content)
# 从一阶段查询结果中用Reranker再次筛选出最相似的2个文档
rerank_content = reranker.rerank(question, content, k=2)
print(rerank_content)
# 最后选择最相似的文档, 交给LLM作为可参考上下文
best_content = rerank_content[0]
chat = MindNLPChat(path='openbmb/MiniCPM-2B-dpo-bf16')
print(chat.chat(question, [], best_content))

```

