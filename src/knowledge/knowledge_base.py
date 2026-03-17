# src/knowledge/knowledge_base.py
import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions  # ✅ 添加这行

class KnowledgeBase:
    """三位一体知识库"""
    
    def __init__(self, config):
        self.config = config.knowledge_base
        self.embedding_model_name = config.model.embedding_model
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        self._init_embedding_function()  # ✅ 先初始化 embedding
        self._init_chroma()
        
        self.document_count = 0
        self.query_count = 0
    
    def _init_embedding_function(self):
        """初始化 Embedding 函数"""
        print(f"📌 加载 Embedding 模型：{self.embedding_model_name}")
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
                trust_remote_code=True
            )
            print("✅ Embedding 模型加载成功")
        except Exception as e:
            print(f"⚠️ Embedding 模型加载失败：{e}")
            print("🔄 回退到默认 embedding function...")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    def _init_chroma(self):
        """初始化 ChromaDB"""
        os.makedirs(self.config.chroma_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.config.chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # ✅ 关键修复：传入 embedding_function
        self.collection = self.client.get_or_create_collection(
            name="defense_knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
    
    # ... 其余方法保持不变 ...
