# src/models/reference_model.py
import torch
import copy
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict
from datetime import datetime

class ReferenceModel:
    """冻结的参考模型（用于KL约束）"""
    
    def __init__(self, model_id: str, device: str = "cuda", 
                 update_type: str = "frozen"):
        self.model_id = model_id
        self.device = device
        self.update_type = update_type
        
        self.model, self.tokenizer = self._load_frozen_model()
        
        self.ema_decay = 0.995
        self.last_update_time = datetime.now()
        
        self.update_count = 0
        self.kl_history = []
    
    def _load_frozen_model(self):
        """加载冻结的参考模型"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map={"": self.device},
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()
        return model, tokenizer
    
    def get_logits(self, input_ids: torch.Tensor, 
                   attention_mask: torch.Tensor = None) -> torch.Tensor:
        """获取参考模型logits"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits[:, -1, :]
        return logits
    
    def get_distribution(self, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor = None) -> torch.Tensor:
        """获取参考模型概率分布"""
        logits = self.get_logits(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)
    
    def update_from_model(self, current_model, update_type: str = None):
        """从当前模型更新参考模型"""
        update_type = update_type or self.update_type
        
        if update_type == "frozen":
            return
        elif update_type == "ema":
            self._ema_update(current_model)
        elif update_type == "periodic":
            self._periodic_update(current_model)
        
        self.update_count += 1
        self.last_update_time = datetime.now()
    
    def _ema_update(self, current_model):
        """EMA更新"""
        with torch.no_grad():
            for ref_param, curr_param in zip(
                self.model.parameters(), 
                current_model.model.parameters()
            ):
                ref_param.data = (
                    self.ema_decay * ref_param.data +
                    (1 - self.ema_decay) * curr_param.data
                )
    
    def _periodic_update(self, current_model):
        """定期完全复制"""
        with torch.no_grad():
            for ref_param, curr_param in zip(
                self.model.parameters(),
                current_model.model.parameters()
            ):
                ref_param.data.copy_(curr_param.data)
    
    def compute_kl_divergence(self, policy_logits: torch.Tensor,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor = None) -> torch.Tensor:
        """计算KL散度"""
        ref_logits = self.get_logits(input_ids, attention_mask)
        
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        
        kl_div = torch.sum(
            torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs),
            dim=-1
        )
        
        self.kl_history.append(kl_div.mean().item())
        return kl_div
    
    def get_kl_stats(self) -> Dict:
        """获取KL统计"""
        if not self.kl_history:
            return {"avg_kl": 0.0, "min_kl": 0.0, "max_kl": 0.0, "update_count": 0}
        
        recent = self.kl_history[-100:]
        return {
            "avg_kl": sum(recent) / len(recent),
            "min_kl": min(recent),
            "max_kl": max(recent),
            "update_count": self.update_count,
            "last_update": self.last_update_time.isoformat()
        }
    
    def save_checkpoint(self, path: str):
        """保存参考模型"""
        checkpoint = {
            "model_id": self.model_id,
            "update_type": self.update_type,
            "update_count": self.update_count,
            "kl_history": self.kl_history[-1000:],
            "ema_decay": self.ema_decay
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
