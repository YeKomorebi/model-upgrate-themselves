# src/mentor/selector.py
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

class MentorSelector:
    """导师选拔器"""
    
    def __init__(self, config):
        self.config = config.mentor
        self.selection_history: List[Dict] = []
    
    def evaluate_candidate(self, defender) -> Dict:
        """评估防御者是否具备导师资格"""
        
        stability = self._calculate_stability(defender)
        kb_coverage = len(defender.attack_type_performance) / 10.0
        kb_coverage = min(1.0, kb_coverage)
        
        metrics = {
            "avg_reward": defender.avg_reward,
            "diversity_score": defender.diversity_score,
            "stability": stability,
            "experience": defender.generation_count,
            "kb_coverage": kb_coverage,
        }
        
        qualified = (
            metrics["avg_reward"] >= self.config.min_avg_reward and
            metrics["diversity_score"] >= self.config.min_diversity_score and
            metrics["stability"] >= self.config.min_stability and
            metrics["experience"] >= self.config.min_generations
        )
        
        mentor_score = (
            metrics["avg_reward"] * 0.35 +
            metrics["diversity_score"] * 0.25 +
            metrics["stability"] * 0.25 +
            metrics["kb_coverage"] * 0.15
        )
        
        return {
            "qualified": qualified,
            "mentor_score": mentor_score,
            "metrics": metrics,
            "defender_id": defender.id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_stability(self, defender) -> float:
        """计算稳定性"""
        if len(defender.reward_history) < 10:
            return 0.5
        
        recent_rewards = defender.reward_history[-10:]
        std = np.std(recent_rewards)
        stability = 1.0 / (1.0 + std * 5)
        return stability
    
    def select_mentors(self, defenders: List, num_mentors: int = None) -> List:
        """从防御者池中选拔导师"""
        num_mentors = num_mentors or self.config.num_mentors
        
        candidates = []
        for defender in defenders:
            evaluation = self.evaluate_candidate(defender)
            if evaluation["qualified"]:
                candidates.append({
                    "defender": defender,
                    "score": evaluation["mentor_score"]
                })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        selected = []
        for candidate in candidates:
            defender = candidate["defender"]
            if len(defender.mentees) < defender.max_mentees:
                selected.append(defender)
                if len(selected) >= num_mentors:
                    break
        
        self.selection_history.append({
            "generation": defenders[0].generation_count if defenders else 0,
            "num_selected": len(selected),
            "num_candidates": len(candidates),
            "timestamp": datetime.now().isoformat()
        })
        
        return selected
    
    def get_selection_stats(self) -> Dict:
        """获取选拔统计"""
        if not self.selection_history:
            return {"total_selections": 0}
        
        return {
            "total_selections": len(self.selection_history),
            "avg_mentors_selected": np.mean([s["num_selected"] for s in self.selection_history]),
            "avg_candidates": np.mean([s["num_candidates"] for s in self.selection_history]),
            "last_selection": self.selection_history[-1] if self.selection_history else None
        }
