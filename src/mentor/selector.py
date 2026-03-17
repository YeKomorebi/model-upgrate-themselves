from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MentorSelector:
    """
    导师选拔器 - 已修复版本
    
    🔧 修复：属性存在性检查
    """
    
    def __init__(self, config):
        self.config = config.mentor if hasattr(config, 'mentor') else {}
        
        # 选拔标准
        self.min_avg_reward = getattr(self.config, 'min_avg_reward', 0.7)
        self.min_diversity = getattr(self.config, 'min_diversity', 0.5)
        self.min_stability = getattr(self.config, 'min_stability', 0.6)
        self.min_generations = getattr(self.config, 'min_generations', 5)
        self.min_experience = getattr(self.config, 'min_experience', 3)
    
    def select_mentors(self, defenders: List, 
                      generation: int,
                      existing_mentors: Optional[List] = None) -> List:
        """
        选拔导师
        
        🔧 修复：添加属性存在性检查
        """
        try:
            qualified_mentors = []
            
            for defender in defenders:
                if self._is_qualified(defender, generation):
                    qualified_mentors.append(defender)
            
            logger.info(f"从 {len(defenders)} 个防御者中选拔出 {len(qualified_mentors)} 个导师")
            
            # 如果有现有导师，优先考虑
            if existing_mentors:
                qualified_mentors = self._prioritize_existing(qualified_mentors, existing_mentors)
            
            return qualified_mentors
            
        except Exception as e:
            logger.error(f"导师选拔失败：{e}")
            return []
    
    def _is_qualified(self, defender, generation: int) -> bool:
        """
        检查防御者是否符合导师资格
        
        🔧 修复：安全的属性访问
        """
        try:
            # 🔧 修复：使用 getattr 安全获取属性
            avg_reward = getattr(defender, 'avg_reward', 0.0)
            diversity_score = getattr(defender, 'diversity_score', 0.0)
            stability = getattr(defender, 'stability', 0.0)
            experience = getattr(defender, 'experience', 0)
            generations_active = getattr(defender, 'generations_active', 0)
            
            # 🔧 修复：安全访问 attack_type_performance
            attack_perf = getattr(defender, 'attack_type_performance', {})
            if isinstance(attack_perf, dict):
                kb_coverage = len(attack_perf) / max(10, len(attack_perf))
            else:
                kb_coverage = 0.0
            
            qualified = (
                avg_reward >= self.min_avg_reward and
                diversity_score >= self.min_diversity and
                stability >= self.min_stability and
                generations_active >= self.min_generations and
                experience >= self.min_experience
            )
            
            if qualified:
                logger.debug(f"防御者 {getattr(defender, 'id', 'unknown')} 符合导师资格")
            
            return qualified
            
        except Exception as e:
            logger.warning(f"检查防御者资格失败：{e}")
            return False
    
    def _calculate_stability(self, defender) -> float:
        """
        计算防御者稳定性
        
        🔧 修复：安全的属性访问和除零保护
        """
        try:
            reward_history = getattr(defender, 'reward_history', [])
            
            if len(reward_history) < 2:
                return 0.0
            
            # 计算最近 5 代的标准差
            recent_rewards = reward_history[-5:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            variance = sum((r - avg_reward) ** 2 for r in recent_rewards) / len(recent_rewards)
            std_dev = variance ** 0.5
            
            # 稳定性 = 1 / (1 + std_dev)
            stability = 1.0 / (1.0 + std_dev)
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.warning(f"计算稳定性失败：{e}")
            return 0.0
    
    def _prioritize_existing(self, qualified: List, existing: List) -> List:
        """优先考虑现有导师"""
        if not existing:
            return qualified
        
        existing_ids = {getattr(m, 'id', str(m)) for m in existing}
        
        # 现有导师排在前面
        prioritized = [m for m in qualified if getattr(m, 'id', str(m)) in existing_ids]
        prioritized += [m for m in qualified if getattr(m, 'id', str(m)) not in existing_ids]
        
        return prioritized
    
    def get_selection_criteria(self) -> Dict[str, Any]:
        """获取选拔标准"""
        return {
            'min_avg_reward': self.min_avg_reward,
            'min_diversity': self.min_diversity,
            'min_stability': self.min_stability,
            'min_generations': self.min_generations,
            'min_experience': self.min_experience
        }
