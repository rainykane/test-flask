from flask import Blueprint
from litellm import embedding
import fitz
import json

# 实现了文档分割以及文本嵌入的简单RAG示例
rag_simple = Blueprint('rag_simple', __name__)

def _extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def _chunk_text(text_input, chunk_size, overlap_size):
    text_chunks = []
    for i in range(0, len(text_input), chunk_size - overlap_size):
        text_chunks.append(text_input[i:i + chunk_size])  # 追加从i到i+chunk_size的文本块到text_chunks列表
    return text_chunks  # 返回文本块列表

def _create_embeddings(texts, model="dashscope/text-embedding-v4"):
    completion = embedding(
        model=model,
        input=texts,
        encoding_format="float"
    )
    data = json.loads(completion.model_dump_json())
    embeddings = [item["embedding"] for item in data["data"]]
    return embeddings

@rag_simple.route('/test/rag')
def openai_rag_simple():
    pdf_path = "static/python_guide.pdf"
    text = _extract_text_from_pdf(pdf_path)
    text_chunks = _chunk_text(text, 1000, 100)
    print("文本块数量:", len(text_chunks))
    # 打印第一个文本块
    print("\n第一个文本块:")
    print(text_chunks[0])
    _create_embeddings(text_chunks)
    result = text_chunks[0]
    return result




