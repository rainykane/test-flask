from flask import Flask
from app.qa_robot import qa
from app.rag_simple import rag_simple
from app.langchain import langchain_module

def create_app():
    app = Flask(__name__)
    app.register_blueprint(qa, url_prefix='/qa')
    app.register_blueprint(rag_simple, url_prefix='/rag')
    app.register_blueprint(langchain_module, url_prefix='/langchain')
    return app