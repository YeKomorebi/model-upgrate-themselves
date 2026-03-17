from typing import Dict, List, Any, Optional
import logging
import torch
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class KLConstraint:
    """
    KL 约束 - 已修复版本
    
    🔧 修复：缓冲区大小限制、内存管理
    """
    
    def __init__(self, config):
        self.config = config.ppo if hasattr(config, 'ppo') else {}
        
        self.kl_coefficient = getattr(self.config, 'kl_coefficient', 0.2)
        self.target_kl = getattr(self.config, 'target_kl', 0.01)
        
        # 🔧 修复：限制缓冲区大小
        self.max_buffer_size = getattr(self.config, 'kl_buffer_size', 1000)
        self.kl_buffer = deque(maxlen=self.max_buffer_size)
        
        self.min_kl_coeff = getattr(self.config, 'min_kl_coeff', 0.01)
        self.max_kl_coeff = getattr(self.config, 'max_kl_coeff', 1.0)
    
    def compute_kl_penalty(self, policy_logits, ref_logits) -> float:
        """
        计算 KL 散度惩罚
        
        🔧 修复：输入验证
        """
        try:
            # 🔧 修复：输入验证
            if policy_logits is None or ref_logits is None:
                logger.warning("logits 为空，返回 0 惩罚")
                return 0.0
            
            if not isinstance(policy_logits, torch.Tensor):
                policy_logits = torch.tensor(policy_logits)
            if not isinstance(ref_logits, torch.Tensor):
                ref_logits = torch.tensor(ref_logits)
            
            # 确保形状一致
            if policy_logits.shape != ref_logits.shape:
                logger.warning("logits 形状不匹配")
                min_len = min(policy_logits.shape[-1], ref_logits.shape[-1])
                policy_logits = policy_logits[..., :min_len]
                ref_logits = ref_logits[..., :min_len]
            
            policy_probs = torch.softmax(policy_logits, dim=-1)
            ref_probs = torch.softmax(ref_logits, dim=-1)
            
            # 🔧 修复：添加 epsilon 避免 log(0)
            epsilon = 1e-10
            kl_div = torch.sum(ref_probs * torch.log((ref_probs + epsilon) / (policy_probs + epsilon)), dim=-1)
            
            kl_value = kl_div.mean().item()
            
            # 记录到缓冲区
            self.kl_buffer.append(kl_value)
            
            return kl_value * self.kl_coefficient
            
        except Exception as e:
            logger.error(f"计算 KL 惩罚失败：{e}")
            return 0.0
    
    def update_kl_coefficient(self):
        """
        自适应更新 KL 系数
        
        🔧 修复：使用 deque 自动限制大小
        """
        try:
            if len(self.kl_buffer) < 10:
                return  # 数据不足
            
            # 🔧 修复：使用 deque，无需手动 pop
            avg_kl = sum(self.kl_buffer) / len(self.kl_buffer)
            
            if avg_kl < self.target_kl / 1.5:
                self.kl_coefficient = max(
                    self.min_kl_coeff,
                    self.kl_coefficient / 1.5
                )
                logger.debug(f"降低 KL 系数：{self.kl_coefficient:.4f}")
            elif avg_kl > self.target_kl * 1.5:
                self.kl_coefficient = min(
                    self.max_kl_coeff,
                    self.kl_coefficient * 1.5
                )
                logger.debug(f"提高 KL 系数：{self.kl_coefficient:.4f}")
            
        except Exception as e:
            logger.error(f"更新 KL 系数失败：{e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取 KL 统计"""
        if not self.kl_buffer:
            return {
                'avg_kl': 0.0,
                'kl_coefficient': self.kl_coefficient,
                'buffer_size': 0,
                'max_buffer_size': self.max_buffer_size
            }
        
        kl_values = list(self.kl_buffer)
        
        return {
            'avg_kl': sum(kl_values) / len(kl_values),
            'min_kl': min(kl_values),
            'max_kl': max(kl_values),
            'kl_coefficient': self.kl_coefficient,
            'target_kl': self.target_kl,
            'buffer_size': len(kl_values),
            'max_buffer_size': self.max_buffer_size
        }
    
    def reset(self):
        """重置 KL 约束"""
        self.kl_buffer.clear()
        self.kl_coefficient = getattr(self.config, 'kl_coefficient', 0.2)
        logger.info("KL 约束已重置")
    
    def get_buffer_memory_usage(self) -> int:
        """
        获取缓冲区内存占用
        
        🔧 修复：新增方法，监控内存
        """
        return len(self.kl_buffer) * 8  # 每个 float 约 8 字节
