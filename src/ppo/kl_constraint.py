# src/ppo/kl_constraint.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class KLConstraintConfig:
    """KL约束配置"""
    kl_coefficient: float = 0.2
    kl_target: float = 0.02
    kl_clip_min: float = 0.001
    kl_clip_max: float = 1.0
    adaptive_kl: bool = True
    horizon: int = 10000

class KLConstraint:
    """KL散度约束管理器"""
    
    def __init__(self, config: KLConstraintConfig):
        self.config = config
        self.current_kl_coeff = config.kl_coefficient
        
        self.kl_buffer = []
        self.update_count = 0
        
        self.stats = {
            "total_updates": 0,
            "kl_coeff_adjustments": 0,
            "kl_violations": 0
        }
    
    def compute_kl_penalty(self, policy_logits: torch.Tensor,
                          ref_logits: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算KL惩罚项"""
        
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        
        kl_div = torch.sum(
            torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs),
            dim=-1
        )
        
        if mask is not None:
            kl_div = kl_div * mask
        
        kl_penalty = kl_div.mean() * self.current_kl_coeff
        
        return kl_penalty
    
    def compute_ppo_loss(self, old_log_probs: torch.Tensor,
                        new_log_probs: torch.Tensor,
                        advantages: torch.Tensor,
                        clip_epsilon: float = 0.2) -> torch.Tensor:
        """计算PPO策略损失"""
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        ratio_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        loss_unclipped = ratio * advantages
        loss_clipped = ratio_clipped * advantages
        
        policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()
        
        return policy_loss
    
    def adaptive_kl_update(self, current_kl: float):
        """自适应调整KL系数"""
        if not self.config.adaptive_kl:
            return
        
        self.kl_buffer.append(current_kl)
        
        if len(self.kl_buffer) > self.config.horizon:
            self.kl_buffer.pop(0)
        
        avg_kl = np.mean(self.kl_buffer)
        
        if avg_kl < self.config.kl_target / 1.5:
            self.current_kl_coeff = max(
                self.config.kl_clip_min,
                self.current_kl_coeff / 1.5
            )
            self.stats["kl_coeff_adjustments"] += 1
        
        elif avg_kl > self.config.kl_target * 1.5:
            self.current_kl_coeff = min(
                self.config.kl_clip_max,
                self.current_kl_coeff * 1.5
            )
            self.stats["kl_coeff_adjustments"] += 1
            self.stats["kl_violations"] += 1
        
        self.stats["total_updates"] += 1
    
    def compute_value_loss(self, predicted_values: torch.Tensor,
                          target_values: torch.Tensor,
                          clip_value: bool = True,
                          value_clip_range: float = 0.2) -> torch.Tensor:
        """计算价值函数损失"""
        
        if clip_value:
            value_pred_clipped = torch.clamp(
                predicted_values,
                target_values - value_clip_range,
                target_values + value_clip_range
            )
            loss_clipped = (value_pred_clipped - target_values) ** 2
            loss_unclipped = (predicted_values - target_values) ** 2
            
            value_loss = torch.max(loss_unclipped, loss_clipped).mean()
        else:
            value_loss = ((predicted_values - target_values) ** 2).mean()
        
        return value_loss
    
    def clip_gradients(self, model, max_grad_norm: float = 1.0) -> float:
        """梯度裁剪"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_grad_norm
        )
        return total_norm.item()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "current_kl_coeff": self.current_kl_coeff,
            "avg_kl_buffer": np.mean(self.kl_buffer) if self.kl_buffer else 0.0,
            "kl_buffer_size": len(self.kl_buffer)
        }
