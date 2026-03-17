from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MentorEvaluator:
    """
    导师评估器 - 已修复版本
    
    🔧 修复：除零风险、硬编码魔法数字
    """
    
    def __init__(self, config):
        self.config = config.mentor if hasattr(config, 'mentor') else {}
        
        # 🔧 修复：从配置读取权重，避免硬编码
        mentor_config = self.config
        self.evaluation_metrics = {
            "mentee_improvement_rate": getattr(mentor_config, 'weight_improvement', 0.4),
            "mentee_retention": getattr(mentor_config, 'weight_retention', 0.2),
            "knowledge_transfer_efficiency": getattr(mentor_config, 'weight_transfer', 0.25),
            "mentee_satisfaction": getattr(mentor_config, 'weight_satisfaction', 0.15),
        }
        
        # 🔧 修复：验证权重总和
        total_weight = sum(self.evaluation_metrics.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"评估权重总和不为 1: {total_weight}，正在归一化...")
            for key in self.evaluation_metrics:
                self.evaluation_metrics[key] /= total_weight
        
        # 🔧 修复：除零保护常数
        self.epsilon = getattr(mentor_config, 'epsilon', 1e-6)
        
        self.evaluation_history: List[Dict] = []
    
    def evaluate_mentor(self, mentor, mentees: List, 
                       before_rewards: Dict[str, float],
                       after_rewards: Dict[str, float]) -> Dict[str, Any]:
        """
        评估导师表现
        
        🔧 修复：添加输入验证和除零保护
        """
        try:
            # 🔧 修复：输入验证
            if not mentees:
                logger.warning("没有学生，无法评估导师")
                return self._create_empty_evaluation(mentor)
            
            if not before_rewards or not after_rewards:
                logger.warning("奖励数据不完整")
                return self._create_empty_evaluation(mentor)
            
            # 计算各项指标
            improvement_rate = self._calculate_improvement_rate(
                mentees, before_rewards, after_rewards
            )
            retention_rate = self._calculate_retention_rate(mentees)
            transfer_efficiency = self._calculate_transfer_efficiency(
                mentor, mentees, before_rewards, after_rewards
            )
            satisfaction = self._calculate_satisfaction(mentees)
            
            # 计算加权总分
            overall_score = (
                improvement_rate * self.evaluation_metrics["mentee_improvement_rate"] +
                retention_rate * self.evaluation_metrics["mentee_retention"] +
                transfer_efficiency * self.evaluation_metrics["knowledge_transfer_efficiency"] +
                satisfaction * self.evaluation_metrics["mentee_satisfaction"]
            )
            
            evaluation = {
                'mentor_id': mentor.id if hasattr(mentor, 'id') else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'scores': {
                    'improvement_rate': improvement_rate,
                    'retention_rate': retention_rate,
                    'transfer_efficiency': transfer_efficiency,
                    'satisfaction': satisfaction,
                    'overall': overall_score
                },
                'metrics_weights': self.evaluation_metrics.copy(),
                'mentee_count': len(mentees),
                'qualified': overall_score >= getattr(self.config, 'min_score', 0.6)
            }
            
            self.evaluation_history.append(evaluation)
            logger.info(f"导师评估完成：{mentor.id if hasattr(mentor, 'id') else 'unknown'} - 总分：{overall_score:.3f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"导师评估失败：{e}")
            return self._create_empty_evaluation(mentor)
    
    def _calculate_improvement_rate(self, mentees: List, 
                                   before_rewards: Dict[str, float],
                                   after_rewards: Dict[str, float]) -> float:
        """
        计算学生进步率
        
        🔧 修复：安全的除零保护
        """
        improvements = []
        
        for mentee in mentees:
            mentee_id = mentee.id if hasattr(mentee, 'id') else str(mentee)
            before = before_rewards.get(mentee_id, 0.0)
            after = after_rewards.get(mentee_id, 0.0)
            
            # 🔧 修复：安全的除法
            divisor = abs(before) + self.epsilon
            if divisor < self.epsilon:
                improvement = 0.0 if after <= 0 else 1.0
            else:
                improvement = (after - before) / divisor
            
            # 限制范围 [-1, 2]
            improvement = max(-1.0, min(2.0, improvement))
            improvements.append(improvement)
        
        if not improvements:
            return 0.0
        
        # 归一化到 [0, 1]
        avg_improvement = sum(improvements) / len(improvements)
        return (avg_improvement + 1) / 3  # [-1,2] -> [0,1]
    
    def _calculate_retention_rate(self, mentees: List) -> float:
        """计算学生保留率"""
        if not mentees:
            return 0.0
        
        # 假设活跃学生是有最近交互的
        active_count = sum(
            1 for m in mentees 
            if hasattr(m, 'last_interaction') and m.last_interaction
        )
        
        return active_count / len(mentees)
    
    def _calculate_transfer_efficiency(self, mentor, mentees: List,
                                      before_rewards: Dict[str, float],
                                      after_rewards: Dict[str, float]) -> float:
        """计算知识传递效率"""
        if not mentees:
            return 0.0
        
        efficiency_scores = []
        
        for mentee in mentees:
            mentee_id = mentee.id if hasattr(mentee, 'id') else str(mentee)
            before = before_rewards.get(mentee_id, 0.0)
            after = after_rewards.get(mentee_id, 0.0)
            
            # 计算能力差距缩小程度
            mentor_reward = getattr(mentor, 'avg_reward', 0.5)
            
            # 🔧 修复：安全的除法
            gap_before = max(0, mentor_reward - before)
            gap_after = max(0, mentor_reward - after)
            
            if gap_before < self.epsilon:
                efficiency = 1.0 if gap_after <= 0 else 0.5
            else:
                efficiency = 1.0 - (gap_after / gap_before)
            
            efficiency = max(0.0, min(1.0, efficiency))
            efficiency_scores.append(efficiency)
        
        return sum(efficiency_scores) / len(efficiency_scores)
    
    def _calculate_satisfaction(self, mentees: List) -> float:
        """计算学生满意度"""
        if not mentees:
            return 0.0
        
        satisfaction_scores = []
        
        for mentee in mentees:
            # 从学生属性获取满意度
            if hasattr(mentee, 'satisfaction_score'):
                score = mentee.satisfaction_score
            elif hasattr(mentee, 'avg_reward'):
                score = mentee.avg_reward
            else:
                score = 0.5  # 默认值
            
            satisfaction_scores.append(max(0.0, min(1.0, score)))
        
        return sum(satisfaction_scores) / len(satisfaction_scores)
    
    def _create_empty_evaluation(self, mentor) -> Dict[str, Any]:
        """创建空评估结果"""
        return {
            'mentor_id': mentor.id if hasattr(mentor, 'id') else 'unknown',
            'timestamp': datetime.now().isoformat(),
            'scores': {
                'improvement_rate': 0.0,
                'retention_rate': 0.0,
                'transfer_efficiency': 0.0,
                'satisfaction': 0.0,
                'overall': 0.0
            },
            'metrics_weights': self.evaluation_metrics.copy(),
            'mentee_count': 0,
            'qualified': False,
            'error': '评估失败或无数据'
        }
    
    def get_evaluation_history(self) -> List[Dict]:
        """获取评估历史"""
        return self.evaluation_history
    
    def get_top_mentors(self, n: int = 5) -> List[Dict]:
        """获取评分最高的导师"""
        if not self.evaluation_history:
            return []
        
        sorted_evals = sorted(
            self.evaluation_history,
            key=lambda x: x['scores']['overall'],
            reverse=True
        )
        
        return sorted_evals[:n]
    
    def reset(self):
        """重置评估历史"""
        self.evaluation_history = []
        logger.info("评估历史已重置")
