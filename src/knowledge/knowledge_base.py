import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """三位一体知识库 - 已修复版本"""
    
    def __init__(self, config):
        self.config = config.knowledge_base if hasattr(config, 'knowledge_base') else {}
        self.embedding_model_name = getattr(config.model, 'embedding_model', 'all-MiniLM-L6-v2')
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        # 🔧 修复：设置默认值
        self.max_documents = getattr(self.config, 'max_documents', 10000)
        self.retrieval_k = getattr(self.config, 'retrieval_k', 5)
        
        self._init_embedding_function()
        self._init_chroma()
        
        self.document_count = 0
        self.query_count = 0
    
    def _init_embedding_function(self):
        """初始化 Embedding 函数"""
        logger.info(f"📌 加载 Embedding 模型：{self.embedding_model_name}")
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
                trust_remote_code=True
            )
            logger.info("✅ Embedding 模型加载成功")
        except Exception as e:
            logger.warning(f"⚠️ Embedding 模型加载失败：{e}")
            logger.info("🔄 回退到默认 embedding function...")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    def _init_chroma(self):
        """初始化 ChromaDB"""
        chroma_path = getattr(self.config, 'chroma_path', './chroma_data')
        os.makedirs(chroma_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name="defense_knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
        # 恢复文档计数
        self.document_count = self.collection.count()
    
    def add_document(self, document_id: str, content: str, metadata: Optional[Dict] = None):
        """添加文档到知识库"""
        try:
            # 🔧 修复：验证输入
            if not content or not content.strip():
                logger.warning("空内容，跳过添加")
                return False
            
            if not document_id or not document_id.strip():
                logger.warning("空文档ID，跳过添加")
                return False
            
            # 检查文档数量限制
            if self.document_count >= self.max_documents:
                logger.warning(f"达到最大文档数限制 ({self.max_documents})")
                return False
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata or {}],
                ids=[document_id]
            )
            self.document_count = self.collection.count()
            logger.info(f"文档 {document_id} 已添加到知识库 (总数：{self.document_count})")
            return True
        except Exception as e:
            logger.error(f"添加文档失败：{e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """批量添加文档"""
        success_count = 0
        for doc in documents:
            doc_id = doc.get('id', f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{success_count}")
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            if self.add_document(doc_id, content, metadata):
                success_count += 1
        
        logger.info(f"批量添加完成：{success_count}/{len(documents)}")
        return success_count
    
    def query(self, query_text: str, n_results: int = None) -> List[str]:
        """
        查询知识库中的相关文档
        
        🔧 修复：参数名统一为 n_results，添加输入验证
        """
        try:
            # 🔧 修复：验证输入
            if not query_text or not query_text.strip():
                logger.warning("空查询文本")
                return []
            
            # 🔧 修复：使用 n_results 参数，设置默认值
            n_results = n_results or self.retrieval_k
            n_results = max(1, min(n_results, 100))  # 限制范围
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            self.query_count += 1
            
            # 🔧 修复：安全检查空列表
            documents = results.get('documents', [[]])
            if not documents or not documents[0]:
                logger.debug(f"查询 '{query_text[:50]}...' 无结果")
                return []
            
            logger.info(f"知识库查询完成，找到 {len(documents[0])} 个结果")
            return documents[0]
            
        except Exception as e:
            logger.error(f"查询知识库失败：{e}")
            return []
    
    def query_with_metadata(self, query_text: str, n_results: int = None) -> List[Dict]:
        """查询文档及元数据"""
        try:
            if not query_text or not query_text.strip():
                return []
            
            n_results = n_results or self.retrieval_k
            n_results = max(1, min(n_results, 100))
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            ids = results.get('ids', [[]])[0]
            
            if not documents:
                return []
            
            return [
                {
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids)
            ]
            
        except Exception as e:
            logger.error(f"查询带元数据失败：{e}")
            return []
    
    def query_by_id(self, document_id: str) -> Optional[Dict]:
        """通过 ID 查询文档"""
        try:
            if not document_id:
                return None
            
            results = self.collection.get(ids=[document_id])
            if results.get('documents') and results['documents']:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results.get('metadatas') else {},
                    'id': results['ids'][0] if results.get('ids') else document_id
                }
            return None
        except Exception as e:
            logger.error(f"通过 ID 查询失败：{e}")
            return None
    
    def update(self, challenges: List[str], responses: List[str], 
               rewards: Optional[List[float]] = None):
        """用 QA 对更新知识库"""
        try:
            if len(challenges) != len(responses):
                logger.error("挑战和回答数量不匹配")
                return
            
            documents = []
            for i, (challenge, response) in enumerate(zip(challenges, responses)):
                reward = rewards[i] if rewards else 0.0
                doc_id = f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                
                content = f"挑战：{challenge}\n回答：{response}\n奖励：{reward}"
                metadata = {
                    'type': 'qa_pair',
                    'reward': reward,
                    'timestamp': datetime.now().isoformat()
                }
                
                documents.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': metadata
                })
            
            self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"更新知识库失败：{e}")
    
    def clear(self):
        """
        清空知识库
        
        🔧 修复：添加异常处理和状态重置
        """
        try:
            if self.client:
                self.client.delete_collection("defense_knowledge")
                logger.info("已删除知识库集合")
        except Exception as e:
            logger.warning(f"删除集合失败：{e}")
        finally:
            # 🔧 修复：确保状态重置
            try:
                self.collection = self.client.get_or_create_collection(
                    name="defense_knowledge",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_function
                )
            except Exception as e:
                logger.error(f"重新创建集合失败：{e}")
            
            self.document_count = 0
            self.query_count = 0
            logger.info("知识库已重置")
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            'document_count': self.document_count,
            'query_count': self.query_count,
            'max_documents': self.max_documents,
            'retrieval_k': self.retrieval_k,
            'collection_name': self.collection.name if self.collection else 'N/A'
        }
    
    def __repr__(self):
        return f"KnowledgeBase(docs={self.document_count}/{self.max_documents}, queries={self.query_count})"
