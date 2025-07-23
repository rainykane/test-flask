from flask import Blueprint
from litellm import completion
import openai

qa = Blueprint('qa_robot', __name__)

def _call_qwen_model_openai(prompt):
    try:
        response = openai.chat.completions.create(
            model="qwen-plus",  # 指定Qwen模型名称
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用Qwen模型时出错: {e}")
        return None
    
def _call_qwen_model_litellm(promt):
    response = completion(
        model="dashscope/qwen-plus",
        messages=[{"role": "user", "content": promt}],
    )
    return response.choices[0].message.content

@qa.route('/test/openai')
def openai_qa():
    prompt = "你和gpt-o3模型对比，优劣势是怎么样的"
    result = _call_qwen_model_openai(prompt)
    if result:
        return result
    else:
        return '开小差啦。'
    
@qa.route('/test/litellm')
def litellm_qa():
    prompt = "你在写代码上的能力怎么样呢"
    result = _call_qwen_model_litellm(prompt)
    if result:
        return result
    else:
        return '开小差啦。'
    