from flask import Blueprint, jsonify
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

langchain_module = Blueprint('langchain_module', __name__)

def _call_qwen_langchain(topic):
    prompt_template = PromptTemplate(input_variables=["topic"], template="{topic}")

    llm = ChatTongyi(model_name="qwen-plus", temperature=0.7)
    chain = prompt_template | llm
    response = chain.invoke({"topic": topic})
    return jsonify({"content": response.content})

def _call_langchain_rag(query):
    # 加载、切分文档
    loader = PyMuPDFLoader("static/python_guide.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    # 建立向量库
    print(f"{type(chunks[0])}")
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=chunks)
    # 检索器
    retriever = vector_store.as_retriever()
    # 大模型连接
    template = """
        使用以下提供的上下文来回答问题。如果答案包含在提供的上下文中，请直接使用上下文回答。
        如果答案无法从上下文中得出，请回复“无法找到答案”。
        上下文:
        {context}
        问题：{question}
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm = ChatTongyi(model_name="qwen-plus", temperature=0)
    # RAG链
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    response = rag_chain.invoke({"query": query})
    return jsonify({"content": response.get('result')})

@langchain_module.route('/test/qa')
def langchain_qa():
    prompt = "你和gpt-o3模型对比，优劣势是怎么样的"
    result = _call_qwen_langchain(prompt)
    if result:
        return result
    else:
        return '开小差啦。'
    
@langchain_module.route('/test/rag')
def langchain_rag():
    query = "什么是虚拟环境"
    result = _call_langchain_rag(query)
    if result:
        return result
    else:
        return '开小差啦。'