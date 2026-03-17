# src/mentor/pairing.py
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime

class MentorPairing:
    """师徒配对系统"""
    
    def __init__(self, config):
        self.config = config.mentor
        self.pairing_history: List[Dict] = []
    
    def pair_mentors_mentees(self, mentors: List, mentees: List, 
                            strategy: str = None) -> List[Tuple]:
        """配对导师和学生"""
        strategy = strategy or self.config.pairing_strategy
        
        pairs = []
        used_mentors = set()
        
        for mentee in mentees:
            if mentee.has_mentor:
                continue
            
            best_mentor = None
            best_score = -1
            
            for mentor in mentors:
                if mentor.id in used_mentors:
                    continue
                
                if mentor.can_take_more_mentees():
                    score = self._calculate_pair_score(mentor, mentee, strategy)
                    
                    if score > best_score:
                        best_score = score
                        best_mentor = mentor
            
            if best_mentor:
                pairs.append((best_mentor, mentee))
                used_mentors.add(best_mentor.id)
                best_mentor.add_mentee(mentee)
        
        if pairs:
            self.pairing_history.append({
                "num_pairs": len(pairs),
                "strategy": strategy,
                "avg_score": np.mean([self._calculate_pair_score(m, e, strategy) for m, e in pairs]),
                "timestamp": datetime.now().isoformat()
            })
        
        return pairs
    
    def _calculate_pair_score(self, mentor, mentee, strategy: str) -> float:
        """计算配对得分"""
        
        if strategy == "best_match":
            similarity = self._calculate_similarity(mentor, mentee)
            return similarity * 0.7 + mentor.mentor_score * 0.3
        
        elif strategy == "complementary":
            complementarity = self._calculate_complementarity(mentor, mentee)
            return complementarity * 0.6 + mentor.mentor_score * 0.4
        
        elif strategy == "diverse":
            diversity = self._calculate_diversity(mentor, mentee)
            return diversity * 0.5 + mentor.mentor_score * 0.5
        
        return 0.0
    
    def _calculate_similarity(self, mentor, mentee) -> float:
        """计算相似度"""
        mentor_profile = mentor.attack_type_performance
        mentee_profile = mentee.attack_type_performance
        
        common_types = set(mentor_profile.keys()) & set(mentee_profile.keys())
        if not common_types:
            return 0.5
        
        mentor_vec = [mentor_profile[t] for t in common_types]
        mentee_vec = [mentee_profile[t] for t in common_types]
        
        dot = sum(a * b for a, b in zip(mentor_vec, mentee_vec))
        norm_mentor = sum(a * a for a in mentor_vec) ** 0.5
        norm_mentee = sum(a * a for a in mentee_vec) ** 0.5
        
        if norm_mentor * norm_mentee == 0:
            return 0.5
        
        return dot / (norm_mentor * norm_mentee)
    
    def _calculate_complementarity(self, mentor, mentee) -> float:
        """计算互补性"""
        mentor_profile = mentor.attack_type_performance
        mentee_profile = mentee.attack_type_performance
        
        scores = []
        for attack_type in mentor_profile:
            if attack_type in mentee_profile:
                if mentor_profile[attack_type] > 0.8 and mentee_profile[attack_type] < 0.5:
                    scores.append(1.0)
                elif mentor_profile[attack_type] < mentee_profile[attack_type]:
                    scores.append(0.0)
                else:
                    scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_diversity(self, mentor, mentee) -> float:
        """计算多样性"""
        mentor_topics = set(mentor.top_expertise_topics[:5])
        mentee_topics = set(mentee.top_expertise_topics[:5])
        
        intersection = len(mentor_topics & mentee_topics)
        union = len(mentor_topics | mentee_topics)
        
        return 1.0 - (intersection / union) if union > 0 else 0.5
    
    def get_pairing_stats(self) -> Dict:
        """获取配对统计"""
        if not self.pairing_history:
            return {"total_pairings": 0}
        
        return {
            "total_pairings": len(self.pairing_history),
            "total_pairs": sum(p["num_pairs"] for p in self.pairing_history),
            "avg_pairs_per_round": np.mean([p["num_pairs"] for p in self.pairing_history]),
            "strategies_used": list(set(p["strategy"] for p in self.pairing_history))
        }
