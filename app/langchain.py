from flask import Blueprint, jsonify
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

langchain_module = Blueprint('langchain_module', __name__)

def _call_qwen_langchain(topic):
    prompt_template = PromptTemplate(input_variables=["topic"], template="{topic}")

    llm = ChatTongyi(model_name="qwen-plus", temperature=0.7)
    chain = prompt_template | llm
    response = chain.invoke({"topic": topic})
    return jsonify({"content": response.content})

@langchain_module.route('/test/qa')
def langchain_qa():
    prompt = "你和gpt-o3模型对比，优劣势是怎么样的"
    result = _call_qwen_langchain(prompt)
    if result:
        return result
    else:
        return '开小差啦。'