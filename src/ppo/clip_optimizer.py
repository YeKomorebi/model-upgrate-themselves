# src/ppo/clip_optimizer.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .kl_constraint import KLConstraint

@dataclass
class ClipOptimizerConfig:
    """Clip优化器配置"""
    learning_rate: float = 2e-5
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4

class ClipOptimizer:
    """PPO-style Clip优化器"""
    
    def __init__(self, model, config: ClipOptimizerConfig, 
                 kl_constraint: KLConstraint):
        self.model = model
        self.config = config
        self.kl_constraint = kl_constraint
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=config.learning_rate * 0.1
        )
        
        self.step_count = 0
        self.loss_history = []
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """计算熵"""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        return entropy
    
    def optimization_step(self, batch: Dict, ref_model, 
                         generation: int) -> Dict:
        """执行一次优化步骤"""
        
        self.model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_penalty = 0.0
        total_entropy = 0.0
        
        for accum_step in range(self.config.gradient_accumulation_steps):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels")
            )
            
            policy_logits = outputs.logits
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            target_values = batch["target_values"]
            
            with torch.no_grad():
                ref_logits = ref_model.get_logits(
                    batch["input_ids"],
                    batch["attention_mask"]
                )
            
            new_log_probs = torch.log_softmax(policy_logits[:, -1, :], dim=-1)
            
            policy_loss = self.kl_constraint.compute_ppo_loss(
                old_log_probs,
                new_log_probs,
                advantages,
                self.config.clip_epsilon
            )
            
            kl_penalty = self.kl_constraint.compute_kl_penalty(
                policy_logits[:, -1, :],
                ref_logits
            )
            
            value_loss = self.kl_constraint.compute_value_loss(
                outputs.logits.mean(dim=-1),
                target_values,
                self.config.clip_value,
                self.config.clip_epsilon
            )
            
            entropy = self.compute_entropy(policy_logits[:, -1, :])
            
            loss = (
                policy_loss +
                self.config.value_loss_coeff * value_loss +
                kl_penalty -
                self.config.entropy_coeff * entropy
            )
            
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl_penalty += kl_penalty.item()
            total_entropy += entropy.item()
        
        grad_norm = self.kl_constraint.clip_gradients(
            self.model, 
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.step_count += 1
        
        avg_kl = total_kl_penalty / self.config.gradient_accumulation_steps
        self.kl_constraint.adaptive_kl_update(avg_kl)
        
        self.loss_history.append({
            "step": self.step_count,
            "policy_loss": total_policy_loss / self.config.gradient_accumulation_steps,
            "value_loss": total_value_loss / self.config.gradient_accumulation_steps,
            "kl_penalty": avg_kl,
            "entropy": total_entropy / self.config.gradient_accumulation_steps,
            "grad_norm": grad_norm,
            "kl_coeff": self.kl_constraint.current_kl_coeff
        })
        
        return {
            "policy_loss": total_policy_loss / self.config.gradient_accumulation_steps,
            "value_loss": total_value_loss / self.config.gradient_accumulation_steps,
            "kl_penalty": avg_kl,
            "entropy": total_entropy / self.config.gradient_accumulation_steps,
            "grad_norm": grad_norm,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        if not self.loss_history:
            return {}
        
        recent = self.loss_history[-100:]
        
        return {
            "total_steps": self.step_count,
            "avg_policy_loss": sum(l["policy_loss"] for l in recent) / len(recent),
            "avg_value_loss": sum(l["value_loss"] for l in recent) / len(recent),
            "avg_kl_penalty": sum(l["kl_penalty"] for l in recent) / len(recent),
            "avg_entropy": sum(l["entropy"] for l in recent) / len(recent),
            "avg_grad_norm": sum(l["grad_norm"] for l in recent) / len(recent),
            "current_lr": self.loss_history[-1]["learning_rate"]
        }
