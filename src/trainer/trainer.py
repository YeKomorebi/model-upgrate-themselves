from typing import List, Dict, Any, Optional
import logging
import json
import os
from datetime import datetime
import fcntl  # 🔧 修复：文件锁

logger = logging.getLogger(__name__)

class MentorEvolutionTrainer:
    """训练器 - 已修复版本"""
    
    def __init__(self, config, knowledge_base=None):
        self.config = config
        self.knowledge_base = knowledge_base
        self.log_dir = getattr(config, 'log_dir', './logs')
        self.checkpoint_dir = getattr(config, 'checkpoint_dir', './checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_log(self, log_entry: Dict, log_path: str):
        """
        保存日志
        
        🔧 修复：原子写入、文件锁
        """
        try:
            # 🔧 修复：使用文件锁
            lock_path = log_path + '.lock'
            
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                try:
                    logs = []
                    if os.path.exists(log_path):
                        with open(log_path, 'r', encoding='utf-8') as f:
                            logs = json.load(f)
                    
                    logs.append(log_entry)
                    
                    # 🔧 修复：原子写入（先写临时文件，再重命名）
                    temp_path = log_path + '.tmp'
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(logs, f, ensure_ascii=False, indent=2)
                    
                    os.replace(temp_path, log_path)
                    
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            
            logger.debug(f"日志已保存：{log_path}")
            
        except Exception as e:
            logger.error(f"保存日志失败：{e}")
