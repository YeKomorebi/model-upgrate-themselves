# src/mentor/distillation.py
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import torch

class KnowledgeDistillation:
    """知识蒸馏系统"""
    
    def __init__(self, config, kl_constraint=None):
        self.config = config.mentor
        self.kl_constraint = kl_constraint
        self.distillation_history: List[Dict] = []
    
    def generate_mentor_guidance(self, mentor, challenge: str, 
                                context: List[str] = None) -> Dict:
        """导师生成指导意见"""
        
        mentor_answer, mentor_logits = mentor.answer_with_logits(challenge, context)
        reasoning_trace = mentor.extract_reasoning_trace()
        key_knowledge = mentor.extract_key_knowledge(context)
        soft_labels = self._apply_temperature(mentor_logits, self.config.distillation_temperature)
        confidence = float(np.max(soft_labels))
        
        return {
            "answer": mentor_answer,
            "reasoning": reasoning_trace,
            "knowledge": key_knowledge,
            "soft_labels": soft_labels,
            "confidence": confidence,
            "mentor_id": mentor.id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """应用温度缩放"""
        exp_logits = np.exp(logits / temperature)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def student_learn(self, student, mentor_guidance: Dict, 
                     student_answer: str, ref_model=None,
                     true_label: float = None) -> float:
        """学生从导师指导中学习"""
        
        student_logits = student._logits_cache
        if student_logits is None:
            student_logits = np.zeros_like(mentor_guidance["soft_labels"])
        
        kd_loss = self._calculate_kd_loss(student_logits, mentor_guidance["soft_labels"])
        reasoning_loss = self._calculate_reasoning_loss(
            student.extract_reasoning_trace(),
            mentor_guidance["reasoning"]
        )
        
        kl_penalty = 0.0
        if self.kl_constraint and ref_model and student._last_input_ids is not None:
            with torch.no_grad():
                ref_logits = ref_model.get_logits(
                    student._last_input_ids,
                    student._last_attention_mask
                )
            kl_penalty = self.kl_constraint.compute_kl_penalty(
                torch.tensor(student_logits),
                ref_logits
            ).item()
        
        total_loss = (
            self.config.distillation_alpha * kd_loss +
            self.config.distillation_beta * reasoning_loss +
            kl_penalty
        )
        
        student.update_from_loss(total_loss)
        
        if student.current_mentor:
            student.current_gap = student.current_mentor._calculate_gap(student)
        
        student.learning_history.append({
            "mentor_id": mentor_guidance["mentor_id"],
            "kd_loss": float(kd_loss),
            "reasoning_loss": float(reasoning_loss),
            "kl_penalty": float(kl_penalty),
            "total_loss": float(total_loss),
            "timestamp": datetime.now().isoformat()
        })
        
        self.distillation_history.append({
            "mentor_id": mentor_guidance["mentor_id"],
            "student_id": student.id,
            "kd_loss": float(kd_loss),
            "kl_penalty": float(kl_penalty),
            "total_loss": float(total_loss),
            "timestamp": datetime.now().isoformat()
        })
        
        return total_loss
    
    def _calculate_kd_loss(self, student_logits: np.ndarray, 
                          mentor_soft_labels: np.ndarray) -> float:
        """计算知识蒸馏损失"""
        student_soft = self._apply_temperature(student_logits, self.config.distillation_temperature)
        
        student_soft = np.clip(student_soft, 1e-10, 1.0)
        mentor_soft_labels = np.clip(mentor_soft_labels, 1e-10, 1.0)
        
        kl_div = np.sum(mentor_soft_labels * np.log(mentor_soft_labels / student_soft))
        
        return float(kl_div * (self.config.distillation_temperature ** 2))
    
    def _calculate_reasoning_loss(self, student_trace: str, mentor_trace: str) -> float:
        """计算推理过程一致性损失"""
        if not student_trace or not mentor_trace:
            return 0.5
        
        words1 = set(student_trace.lower().split())
        words2 = set(mentor_trace.lower().split())
        
        if not words1 or not words2:
            return 0.5
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0.0
        return 1.0 - similarity
    
    def get_distillation_stats(self) -> Dict:
        """获取蒸馏统计"""
        if not self.distillation_history:
            return {"total_distillations": 0}
        
        losses = [d["total_loss"] for d in self.distillation_history]
        
        return {
            "total_distillations": len(self.distillation_history),
            "avg_loss": float(np.mean(losses)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "recent_trend": float(np.mean(losses[-10:])) if len(losses) >= 10 else float(np.mean(losses))
        }
