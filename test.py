from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import MindNLPChat
from RAG.Embeddings import MindNLPEmbedding

# 没有保存数据库
# docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)  # 获得data目录下的所有文件内容并分割
# vector = VectorStore(docs)
# embedding = MindNLPEmbedding("BAAI/bge-base-zh-v1.5")  # 创建EmbeddingModel
# vector.get_vector(EmbeddingModel=embedding)
# vector.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector.load_vector(EmbeddingModel=embedding, path='./storage')  # 加载本地的数据库

# question = 'git如何新建分支？'

# content = vector.query(question, EmbeddingModel=embedding, k=1)
# print(content)
# chat = MindNLPChat(path='THUDM/chatglm3-6b')
# print(chat.chat(question, [], content))


# 保存数据库之后
vector = VectorStore()
embedding = MindNLPEmbedding("BAAI/bge-base-zh-v1.5") # 创建EmbeddingModel
vector.load_vector(EmbeddingModel=embedding, path='./storage')  # 加载本地的数据库
question = '逆向纠错的原理是什么？'
content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
print(content)
chat = MindNLPChat(path='THUDM/chatglm-6b')
print(chat.chat(question, [], content))
