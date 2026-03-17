# src/models/defender.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import json
import os

class DefenderModel:
    """防御者模型（支持导师制和PPO约束）"""
    
    def __init__(self, model_id: str, config, device: str = "cuda"):
        self.id = f"defender_{model_id.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_id = model_id
        self.config = config
        self.device = device
        
        # 模型加载
        self.model, self.tokenizer = self._load_model()
        
        # LoRA配置
        if config.model.use_qlora:
            self._setup_lora()
        
        # 性能指标
        self.avg_reward = 0.0
        self.reward_history: List[float] = []
        self.diversity_score = 0.5
        self.generation_count = 0
        self.kb_coverage = 0.0
        
        # 攻击类型表现
        self.attack_type_performance: Dict[str, float] = {}
        self.top_expertise_topics: List[str] = []
        
        # 导师制相关属性
        self.is_mentor = False
        self.mentor_since: Optional[int] = None
        self.mentees: List["DefenderModel"] = []
        self.max_mentees = config.mentor.max_mentees_per_mentor
        self.mentor_score = 0.0
        self.has_mentor = False
        self.current_mentor: Optional["DefenderModel"] = None
        self.learning_history: List[Dict] = []
        self.warning_count = 0
        self.last_evaluation_score = 0.0
        self.initial_gap: float = 0.5
        self.current_gap: float = 0.5
        
        # PPO相关
        self.use_ppo = config.ppo.enabled if hasattr(config, 'ppo') else False
        self.experience_buffer: Optional[Dict] = None
        self.update_step_count = 0
        self.max_update_steps_per_gen = 10
        self.kl_history: List[float] = []
        
        # 缓存
        self._logits_cache = None
        self._reasoning_trace = ""
        self._last_input_ids = None
        self._last_attention_mask = None
    
    def _load_model(self):
        """加载模型"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        if self.config.model.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map={"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map={"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    
    def _setup_lora(self):
        """设置LoRA"""
        lora_config = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def answer(self, challenge: str, context: List[str] = None) -> Tuple[str, float]:
        """生成回答"""
        prompt = self._build_prompt(challenge, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.config.model.max_seq_len)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 缓存input
        self._last_input_ids = inputs["input_ids"]
        self._last_attention_mask = inputs.get("attention_mask")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], 
                                       skip_special_tokens=True)
        
        confidence = self._calculate_confidence(outputs, inputs)
        self.generation_count += 1
        
        return answer, confidence
    
    def answer_with_logits(self, challenge: str, context: List[str] = None) -> Tuple[str, np.ndarray]:
        """生成回答并返回logits"""
        prompt = self._build_prompt(challenge, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=self.config.model.max_seq_len)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self._last_input_ids = inputs["input_ids"]
        self._last_attention_mask = inputs.get("attention_mask")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].cpu().numpy()
        
        answer = self.tokenizer.decode(
            torch.argmax(outputs.logits[:, -1, :], dim=-1)[0],
            skip_special_tokens=True
        )
        
        self._logits_cache = logits
        return answer, logits
    
    def extract_reasoning_trace(self) -> str:
        """提取推理过程"""
        return self._reasoning_trace
    
    def extract_key_knowledge(self, context: List[str]) -> List[str]:
        """提取关键知识点"""
        if not context:
            return []
        return context[:3]
    
    def _calculate_confidence(self, outputs, inputs) -> float:
        """计算回答置信度"""
        logits = outputs.logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        confidence = torch.max(probs).item()
        return confidence
    
    def _build_prompt(self, challenge: str, context: List[str] = None) -> str:
        """构建提示词"""
        prompt = "你是一个安全防御助手。请回答以下挑战：\n\n"
        if context:
            prompt += "参考知识：\n" + "\n".join(context[:3]) + "\n\n"
        prompt += f"挑战：{challenge}\n\n回答："
        return prompt
    
    def update_reward(self, reward: float, attack_type: str = None):
        """更新奖励"""
        self.reward_history.append(reward)
        
        window_size = min(20, len(self.reward_history))
        self.avg_reward = np.mean(self.reward_history[-window_size:])
        
        if attack_type:
            if attack_type not in self.attack_type_performance:
                self.attack_type_performance[attack_type] = reward
            else:
                alpha = 0.3
                self.attack_type_performance[attack_type] = (
                    alpha * reward + (1 - alpha) * self.attack_type_performance[attack_type]
                )
    
    def update_from_loss(self, loss: float):
        """从损失更新模型"""
        self.learning_history.append({
            "loss": loss,
            "timestamp": datetime.now().isoformat()
        })
    
    def store_experience(self, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        log_probs: torch.Tensor,
                        rewards: torch.Tensor,
                        values: torch.Tensor = None):
        """存储经验（用于PPO更新）"""
        self.experience_buffer = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_log_probs": log_probs,
            "rewards": rewards,
            "values": values
        }
    
    def compute_advantages(self, rewards: torch.Tensor, 
                          values: torch.Tensor = None,
                          gamma: float = 0.99,
                          lam: float = 0.95) -> torch.Tensor:
        """计算优势函数（GAE）"""
        if values is None:
            advantages = rewards - rewards.mean()
        else:
            deltas = rewards + gamma * values[1:] - values[:-1]
            advantages = []
            gae = 0
            for delta in reversed(deltas):
                gae = delta + gamma * lam * gae
                advantages.insert(0, gae)
            advantages = torch.tensor(advantages)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def ppo_update(self, ref_model, kl_constraint, optimizer, 
                   generation: int) -> Dict:
        """执行PPO更新"""
        if not self.use_ppo or not hasattr(self, 'experience_buffer') or not self.experience_buffer:
            return {}
        
        batch = self.experience_buffer
        advantages = self.compute_advantages(batch["rewards"], batch.get("values"))
        
        if batch.get("values") is not None:
            target_values = batch["values"] + advantages
        else:
            target_values = batch["rewards"]
        
        opt_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "old_log_probs": batch["old_log_probs"],
            "advantages": advantages,
            "target_values": target_values
        }
        
        update_stats = optimizer.optimization_step(opt_batch, ref_model, generation)
        
        if "kl_penalty" in update_stats:
            self.kl_history.append(update_stats["kl_penalty"])
        
        self.update_step_count += 1
        return update_stats
    
    def can_take_more_mentees(self) -> bool:
        """检查是否能接收更多学生"""
        return len(self.mentees) < self.max_mentees
    
    def add_mentee(self, mentee: "DefenderModel"):
        """添加学生"""
        if self.can_take_more_mentees():
            self.mentees.append(mentee)
            mentee.has_mentor = True
            mentee.current_mentor = self
            mentee.initial_gap = self._calculate_gap(mentee)
    
    def remove_mentee(self, mentee: "DefenderModel"):
        """移除学生"""
        if mentee in self.mentees:
            self.mentees.remove(mentee)
            mentee.has_mentor = False
            mentee.current_mentor = None
    
    def _calculate_gap(self, mentee: "DefenderModel") -> float:
        """计算与学生的性能差距"""
        return abs(self.avg_reward - mentee.avg_reward)
    
    def get_mentor_feedback(self) -> Dict:
        """获取导师评价"""
        if not self.current_mentor:
            return {"recommend_promotion": False}
        
        if len(self.reward_history) >= 20:
            before = np.mean(self.reward_history[:10])
            after = np.mean(self.reward_history[-10:])
            improvement = (after - before) / (before + 1e-10)
            
            recommend = improvement > 0.1 and self.avg_reward > 0.7
            return {
                "recommend_promotion": recommend,
                "improvement_rate": improvement,
                "mentor_id": self.current_mentor.id
            }
        
        return {"recommend_promotion": False}
    
    def check_update_budget(self) -> bool:
        """检查是否还有更新预算"""
        return self.update_step_count < self.max_update_steps_per_gen
    
    def reset_update_budget(self):
        """重置更新预算"""
        self.update_step_count = 0
    
    def get_kl_stats(self) -> Dict:
        """获取KL统计"""
        if not self.kl_history:
            return {"avg_kl": 0.0, "min_kl": 0.0, "max_kl": 0.0}
        
        recent = self.kl_history[-100:]
        return {
            "avg_kl": sum(recent) / len(recent),
            "min_kl": min(recent),
            "max_kl": max(recent),
            "total_updates": len(self.kl_history)
        }
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            "id": self.id,
            "model_id": self.model_id,
            "avg_reward": self.avg_reward,
            "reward_history": self.reward_history[-100:],
            "diversity_score": self.diversity_score,
            "generation_count": self.generation_count,
            "is_mentor": self.is_mentor,
            "mentor_since": self.mentor_since,
            "mentees": [m.id for m in self.mentees],
            "has_mentor": self.has_mentor,
            "current_mentor_id": self.current_mentor.id if self.current_mentor else None,
            "attack_type_performance": self.attack_type_performance,
            "learning_history": self.learning_history[-100:],
            "kl_history": self.kl_history[-100:]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        model_path = path.replace('.json', '_model')
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(model_path)
    
    @classmethod
    def load_checkpoint(cls, path: str, config, device: str = "cuda") -> "DefenderModel":
        """从检查点加载"""
        with open(path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        defender = cls(checkpoint["model_id"], config, device)
        defender.id = checkpoint["id"]
        defender.avg_reward = checkpoint["avg_reward"]
        defender.reward_history = checkpoint["reward_history"]
        defender.diversity_score = checkpoint["diversity_score"]
        defender.generation_count = checkpoint["generation_count"]
        defender.is_mentor = checkpoint["is_mentor"]
        defender.mentor_since = checkpoint["mentor_since"]
        defender.has_mentor = checkpoint["has_mentor"]
        defender.attack_type_performance = checkpoint["attack_type_performance"]
        defender.learning_history = checkpoint["learning_history"]
        defender.kl_history = checkpoint.get("kl_history", [])
        
        model_path = path.replace('.json', '_model')
        if os.path.exists(model_path):
            defender.model = PeftModel.from_pretrained(defender.model, model_path)
        
        return defender
    
    def __repr__(self):
        mentor_status = "🎓导师" if self.is_mentor else "📚学员"
        if self.has_mentor:
            mentor_status += f"(导师:{self.current_mentor.id[:20]}...)"
        return f"{mentor_status} {self.id[:30]} (奖励:{self.avg_reward:.3f}, 代数:{self.generation_count})"
