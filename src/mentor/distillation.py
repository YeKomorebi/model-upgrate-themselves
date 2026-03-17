from typing import Dict, List, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)

class KnowledgeDistillation:
    """
    知识蒸馏 - 已修复版本
    
    🔧 修复：输入验证、None 检查
    """
    
    def __init__(self, config):
        self.config = config.distillation if hasattr(config, 'distillation') else {}
        
        self.temperature = getattr(self.config, 'temperature', 2.0)
        self.alpha = getattr(self.config, 'alpha', 0.7)  # 导师损失权重
        self.beta = getattr(self.config, 'beta', 0.3)    # 真实标签权重
    
    def student_learn(self, student, mentor_guidance: Dict, 
                     student_answer: str, ref_model=None,
                     true_label: float = None) -> float:
        """
        学生学习过程
        
        🔧 修复：输入验证、None 检查
        """
        try:
            # 🔧 修复：输入验证
            if not mentor_guidance:
                logger.warning("导师指导为空")
                return 0.0
            
            if not student_answer:
                logger.warning("学生回答为空")
                return 0.0
            
            # 🔧 修复：验证 student 对象
            if not student:
                logger.error("学生对象为空")
                return 0.0
            
            # 🔧 修复：安全检查 logits 缓存
            student_logits = getattr(student, '_logits_cache', None)
            if student_logits is None:
                logger.warning("学生 logits 缓存为空，使用默认值")
                student_logits = torch.zeros(1, 1000)
            
            mentor_logits = mentor_guidance.get('logits', None)
            if mentor_logits is None:
                logger.warning("导师 logits 为空")
                mentor_logits = torch.zeros(1, 1000)
            
            # 计算蒸馏损失
            distillation_loss = self._compute_distillation_loss(
                student_logits, mentor_logits
            )
            
            # 计算真实标签损失
            label_loss = 0.0
            if true_label is not None:
                label_loss = self._compute_label_loss(student_logits, true_label)
            
            # 组合损失
            total_loss = self.alpha * distillation_loss + self.beta * label_loss
            
            # 更新学生
            self._update_student(student, total_loss, mentor_guidance)
            
            logger.debug(f"蒸馏损失：{total_loss:.4f} (蒸馏：{distillation_loss:.4f}, 标签：{label_loss:.4f})")
            
            return 1.0 / (1.0 + total_loss)  # 转换为奖励
            
        except Exception as e:
            logger.error(f"知识蒸馏失败：{e}")
            return 0.0
    
    def _compute_distillation_loss(self, student_logits, mentor_logits) -> float:
        """计算蒸馏损失"""
        try:
            # 🔧 修复：张量验证
            if not isinstance(student_logits, torch.Tensor):
                student_logits = torch.tensor(student_logits)
            if not isinstance(mentor_logits, torch.Tensor):
                mentor_logits = torch.tensor(mentor_logits)
            
            # 确保形状一致
            if student_logits.shape != mentor_logits.shape:
                logger.warning("logits 形状不匹配，进行裁剪")
                min_len = min(student_logits.shape[-1], mentor_logits.shape[-1])
                student_logits = student_logits[..., :min_len]
                mentor_logits = mentor_logits[..., :min_len]
            
            # KL 散度
            student_probs = torch.softmax(student_logits / self.temperature, dim=-1)
            mentor_probs = torch.softmax(mentor_logits / self.temperature, dim=-1)
            
            # 🔧 修复：添加 epsilon 避免 log(0)
            epsilon = 1e-10
            kl_loss = torch.sum(mentor_probs * torch.log((mentor_probs + epsilon) / (student_probs + epsilon)))
            
            return kl_loss.item()
            
        except Exception as e:
            logger.error(f"计算蒸馏损失失败：{e}")
            return 1.0
    
    def _compute_label_loss(self, student_logits, true_label: float) -> float:
        """计算真实标签损失"""
        try:
            # 🔧 修复：标签验证
            if not isinstance(true_label, (int, float)):
                logger.warning(f"无效的标签类型：{type(true_label)}")
                return 0.5
            
            true_label = max(0.0, min(1.0, true_label))  # 限制范围
            
            # BCE 损失
            prob = torch.sigmoid(student_logits).mean().item()
            loss = -(true_label * torch.log(torch.tensor(prob + 1e-10)) + 
                    (1 - true_label) * torch.log(torch.tensor(1 - prob + 1e-10)))
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"计算标签损失失败：{e}")
            return 0.5
    
    def _update_student(self, student, loss: float, mentor_guidance: Dict):
        """更新学生模型"""
        try:
            # 🔧 修复：安全检查
            if hasattr(student, 'update_from_distillation'):
                student.update_from_distillation(loss, mentor_guidance)
            
            # 记录学习历史
            if hasattr(student, 'learning_history'):
                student.learning_history.append({
                    'loss': loss,
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.warning(f"更新学生失败：{e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta
        }
