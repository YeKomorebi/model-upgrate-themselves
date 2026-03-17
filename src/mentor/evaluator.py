# src/mentor/evaluator.py
from typing import List, Dict
import numpy as np
from datetime import datetime

class MentorEvaluator:
    """导师绩效考核系统"""
    
    def __init__(self, config):
        self.config = config.mentor
        self.evaluation_history: List[Dict] = []
        self.evaluation_metrics = {
            "mentee_improvement_rate": 0.4,
            "mentee_retention": 0.2,
            "knowledge_transfer_efficiency": 0.25,
            "mentee_satisfaction": 0.15,
        }
    
    def evaluate_mentor(self, mentor) -> Dict:
        """评估导师绩效"""
        
        mentees = mentor.mentees
        if not mentees:
            return {
                "mentor_id": mentor.id,
                "score": 0.0,
                "status": "no_mentees",
                "metrics": {},
                "num_mentees": 0
            }
        
        improvement_rates = []
        for mentee in mentees:
            if len(mentee.reward_history) >= 10:
                before = np.mean(mentee.reward_history[:5])
                after = np.mean(mentee.reward_history[-5:])
                improvement = (after - before) / (before + 1e-10)
                improvement_rates.append(improvement)
        
        avg_improvement = float(np.mean(improvement_rates)) if improvement_rates else 0.0
        
        active_mentees = sum(1 for m in mentees if m.generation_count > 0)
        retention = active_mentees / len(mentees)
        
        transfer_efficiency = self._calculate_transfer_efficiency(mentor, mentees)
        satisfaction = min(1.0, max(0.0, avg_improvement + 0.5))
        
        mentor_score = (
            avg_improvement * self.evaluation_metrics["mentee_improvement_rate"] +
            retention * self.evaluation_metrics["mentee_retention"] +
            transfer_efficiency * self.evaluation_metrics["knowledge_transfer_efficiency"] +
            satisfaction * self.evaluation_metrics["mentee_satisfaction"]
        )
        
        if mentor_score >= 0.8:
            status = "excellent"
        elif mentor_score >= 0.6:
            status = "qualified"
        elif mentor_score >= 0.4:
            status = "needs_improvement"
        else:
            status = "at_risk"
        
        mentor.last_evaluation_score = mentor_score
        mentor.mentor_score = mentor_score
        
        result = {
            "mentor_id": mentor.id,
            "score": float(mentor_score),
            "status": status,
            "metrics": {
                "improvement_rate": avg_improvement,
                "retention": float(retention),
                "transfer_efficiency": float(transfer_efficiency),
                "satisfaction": float(satisfaction)
            },
            "num_mentees": len(mentees),
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluation_history.append(result)
        return result
    
    def _calculate_transfer_efficiency(self, mentor, mentees) -> float:
        """计算知识传递效率"""
        efficiencies = []
        
        for mentee in mentees:
            if hasattr(mentee, 'initial_gap') and hasattr(mentee, 'current_gap'):
                if mentee.initial_gap > 0:
                    gap_reduction = (mentee.initial_gap - mentee.current_gap) / mentee.initial_gap
                    efficiencies.append(max(0, gap_reduction))
        
        return float(np.mean(efficiencies)) if efficiencies else 0.5
    
    def review_mentors(self, mentors: List, generation: int) -> Dict:
        """定期审查导师"""
        
        review_results = {
            "excellent": [],
            "qualified": [],
            "needs_improvement": [],
            "dismissed": [],
            "generation": generation
        }
        
        for mentor in mentors:
            evaluation = self.evaluate_mentor(mentor)
            
            if evaluation["status"] == "excellent":
                review_results["excellent"].append(mentor)
                mentor.max_mentees = min(5, mentor.max_mentees + 1)
            elif evaluation["status"] == "qualified":
                review_results["qualified"].append(mentor)
            elif evaluation["status"] == "needs_improvement":
                review_results["needs_improvement"].append(mentor)
                mentor.max_mentees = max(1, mentor.max_mentees - 1)
                mentor.warning_count += 1
            else:
                review_results["dismissed"].append(mentor)
                mentor.is_mentor = False
                for mentee in mentor.mentees[:]:
                    mentor.remove_mentee(mentee)
        
        return review_results
    
    def get_evaluation_stats(self) -> Dict:
        """获取评估统计"""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        scores = [e["score"] for e in self.evaluation_history]
        status_counts = {}
        for e in self.evaluation_history:
            status = e["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "avg_score": float(np.mean(scores)),
            "status_distribution": status_counts,
            "recent_avg": float(np.mean(scores[-10:])) if len(scores) >= 10 else float(np.mean(scores))
        }
