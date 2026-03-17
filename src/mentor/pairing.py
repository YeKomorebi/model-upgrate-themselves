from typing import List, Dict, Any, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)

class MentorPairing:
    """
    师徒配对器 - 已修复版本
    
    🔧 修复：策略验证、默认值处理
    """
    
    VALID_STRATEGIES = ['best_match', 'complementary', 'diverse', 'random']
    
    def __init__(self, config):
        self.config = config.mentor if hasattr(config, 'mentor') else {}
        
        # 🔧 修复：验证策略
        strategy = getattr(self.config, 'pairing_strategy', 'best_match')
        if strategy not in self.VALID_STRATEGIES:
            logger.warning(f"无效的配对策略 '{strategy}'，使用默认 'best_match'")
            strategy = 'best_match'
        
        self.strategy = strategy
        
        # 策略权重
        self.strategy_weights = {
            'best_match': {'similarity': 0.7, 'mentor_score': 0.3},
            'complementary': {'complementarity': 0.6, 'mentor_score': 0.4},
            'diverse': {'diversity': 0.5, 'mentor_score': 0.5},
            'random': {'random': 1.0}
        }
    
    def pair_mentors_mentees(self, mentors: List, mentees: List,
                            max_mentees_per_mentor: int = 3) -> Dict[str, List]:
        """
        配对导师和学生
        
        🔧 修复：输入验证、策略验证
        """
        try:
            # 🔧 修复：输入验证
            if not mentors:
                logger.warning("没有可用导师")
                return {}
            
            if not mentees:
                logger.warning("没有可用学生")
                return {}
            
            # 🔧 修复：验证策略
            if self.strategy not in self.VALID_STRATEGIES:
                logger.warning(f"无效策略 '{self.strategy}'，使用默认")
                self.strategy = 'best_match'
            
            pairings = {}
            assigned_mentees = set()
            
            # 按导师分数排序
            sorted_mentors = sorted(
                mentors,
                key=lambda m: getattr(m, 'avg_reward', 0),
                reverse=True
            )
            
            for mentor in sorted_mentors:
                mentor_id = getattr(mentor, 'id', str(mentor))
                available_mentees = [m for m in mentees if getattr(m, 'id', str(m)) not in assigned_mentees]
                
                if not available_mentees:
                    break
                
                # 选择最佳匹配的学生
                selected = self._select_mentees(
                    mentor, 
                    available_mentees, 
                    min(max_mentees_per_mentor, len(available_mentees))
                )
                
                pairings[mentor_id] = selected
                assigned_mentees.update(getattr(m, 'id', str(m)) for m in selected)
            
            logger.info(f"完成配对：{len(pairings)} 个导师，{len(assigned_mentees)} 个学生")
            return pairings
            
        except Exception as e:
            logger.error(f"配对失败：{e}")
            return {}
    
    def _select_mentees(self, mentor, mentees: List, count: int) -> List:
        """选择学生"""
        if not mentees or count <= 0:
            return []
        
        # 🔧 修复：验证策略
        if self.strategy not in self.strategy_weights:
            logger.warning(f"未知策略 '{self.strategy}'，使用 best_match")
            self.strategy = 'best_match'
        
        weights = self.strategy_weights.get(self.strategy, self.strategy_weights['best_match'])
        
        # 计算每个学生的匹配分数
        scores = []
        for mentee in mentees:
            score = self._calculate_pairing_score(mentor, mentee, weights)
            scores.append((score, mentee))
        
        # 按分数排序
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # 选择前 count 个
        selected = [mentee for _, mentee in scores[:count]]
        return selected
    
    def _calculate_pairing_score(self, mentor, mentee, weights: Dict) -> float:
        """
        计算配对分数
        
        🔧 修复：安全的属性访问
        """
        score = 0.0
        
        if 'similarity' in weights:
            similarity = self._calculate_similarity(mentor, mentee)
            score += similarity * weights['similarity']
        
        if 'complementarity' in weights:
            complementarity = self._calculate_complementarity(mentor, mentee)
            score += complementarity * weights['complementarity']
        
        if 'diversity' in weights:
            diversity = self._calculate_diversity(mentor, mentee)
            score += diversity * weights['diversity']
        
        if 'mentor_score' in weights:
            mentor_score = getattr(mentor, 'avg_reward', 0.5)
            score += mentor_score * weights['mentor_score']
        
        if 'random' in weights:
            score += random.random() * weights['random']
        
        return score
    
    def _calculate_similarity(self, mentor, mentee) -> float:
        """计算相似度"""
        # 🔧 修复：安全访问属性
        mentor_types = getattr(mentor, 'attack_type_performance', {})
        mentee_types = getattr(mentee, 'attack_type_performance', {})
        
        if not mentor_types or not mentee_types:
            return 0.5
        
        common_types = set(mentor_types.keys()) & set(mentee_types.keys())
        all_types = set(mentor_types.keys()) | set(mentee_types.keys())
        
        return len(common_types) / max(1, len(all_types))
    
    def _calculate_complementarity(self, mentor, mentee) -> float:
        """计算互补性"""
        mentor_weaknesses = getattr(mentor, 'weaknesses', [])
        mentee_needs = getattr(mentee, 'learning_needs', [])
        
        if not mentor_weaknesses or not mentee_needs:
            return 0.5
        
        # 导师的弱点是否是学生需要的
        complement = set(mentor_weaknesses) & set(mentee_needs)
        return len(complement) / max(1, len(mentee_needs))
    
    def _calculate_diversity(self, mentor, mentee) -> float:
        """计算多样性"""
        mentor_style = getattr(mentor, 'defense_style', 'default')
        mentee_style = getattr(mentee, 'defense_style', 'default')
        
        return 1.0 if mentor_style != mentee_style else 0.3
    
    def get_pairing_statistics(self, pairings: Dict) -> Dict[str, Any]:
        """获取配对统计"""
        if not pairings:
            return {'total_pairs': 0, 'mentors_count': 0, 'mentees_count': 0}
        
        total_mentees = sum(len(mentees) for mentees in pairings.values())
        
        return {
            'total_pairs': total_mentees,
            'mentors_count': len(pairings),
            'mentees_count': total_mentees,
            'avg_mentees_per_mentor': total_mentees / max(1, len(pairings)),
            'strategy': self.strategy
        }
