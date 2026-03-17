# src/knowledge/knowledge_base.py
import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings

class KnowledgeBase:
    """三位一体知识库"""
    
    def __init__(self, config):
        self.config = config.knowledge_base
        self.client = None
        self.collection = None
        
        self._init_chroma()
        
        self.document_count = 0
        self.query_count = 0
    
    def _init_chroma(self):
        """初始化ChromaDB"""
        os.makedirs(self.config.chroma_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.config.chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name="defense_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, text: str, metadata: Dict = None):
        """添加文档"""
        if self.document_count >= self.config.max_documents:
            self._cleanup_old_documents()
        
        doc_id = f"doc_{self.document_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        self.document_count += 1
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """批量添加文档"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        ids = [f"doc_{self.document_count + i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
               for i in range(len(texts))]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        self.document_count += len(texts)
    
    def query(self, query_text: str, k: int = None) -> List[str]:
        """查询知识库"""
        k = k or self.config.retrieval_k
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k
        )
        
        self.query_count += 1
        
        if results and results['documents']:
            return results['documents'][0]
        return []
    
    def query_with_metadata(self, query_text: str, k: int = None) -> List[Dict]:
        """查询知识库（带元数据）"""
        k = k or self.config.retrieval_k
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        self.query_count += 1
        
        if results and results['documents']:
            return [
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        return []
    
    def update(self, challenges: List[str], responses: List[str]):
        """更新知识库"""
        documents = []
        metadatas = []
        
        for challenge, response in zip(challenges, responses):
            doc_text = f"挑战：{challenge}\n回答：{response}"
            documents.append(doc_text)
            metadatas.append({
                "type": "qa_pair",
                "timestamp": datetime.now().isoformat()
            })
        
        self.add_documents(documents, metadatas)
        
        if self.document_count % self.config.save_frequency == 0:
            self._save_checkpoint()
    
    def _cleanup_old_documents(self):
        """清理旧文档"""
        # 简化：删除最早的10%文档
        num_to_delete = int(self.document_count * 0.1)
        
        if num_to_delete > 0:
            ids_to_delete = [f"doc_{i}" for i in range(num_to_delete)]
            try:
                self.collection.delete(ids=ids_to_delete)
                self.document_count -= num_to_delete
            except Exception:
                pass
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            "document_count": self.document_count,
            "query_count": self.query_count,
            "last_update": datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.config.chroma_path, 
            "checkpoint.json"
        )
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "document_count": self.document_count,
            "query_count": self.query_count,
            "max_documents": self.config.max_documents,
            "retrieval_k": self.config.retrieval_k
        }
    
    def clear(self):
        """清空知识库"""
        self.client.delete_collection("defense_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="defense_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self.document_count = 0
        self.query_count = 0
