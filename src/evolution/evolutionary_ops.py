# src/evolution/evolutionary_ops.py
from typing import List, Dict
import random
import copy
import torch
from datetime import datetime

class EvolutionaryOperations:
    """进化操作"""
    
    def __init__(self, config):
        self.config = config.evolution
        self.evolution_history: List[Dict] = []
    
    def evolve(self, defenders: List, generation: int) -> List:
        """执行进化操作"""
        
        # 1. 评估适应度
        fitness_scores = [d.avg_reward for d in defenders]
        
        # 2. 选择精英
        elite_indices = self._select_elite(defenders, fitness_scores)
        elites = [defenders[i] for i in elite_indices]
        
        # 3. 杂交
        offspring = self._crossover(elites)
        
        # 4. 突变
        offspring = self._mutate(offspring)
        
        # 5. 选择下一代
        next_generation = self._select_next_generation(
            defenders, offspring, fitness_scores
        )
        
        # 记录进化历史
        self.evolution_history.append({
            "generation": generation,
            "num_elites": len(elites),
            "num_offspring": len(offspring),
            "avg_fitness": sum(fitness_scores) / len(fitness_scores),
            "best_fitness": max(fitness_scores),
            "timestamp": datetime.now().isoformat()
        })
        
        return next_generation
    
    def _select_elite(self, defenders: List, fitness_scores: List) -> List[int]:
        """选择精英"""
        sorted_indices = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i],
            reverse=True
        )
        return sorted_indices[:self.config.elite_count]
    
    def _crossover(self, elites: List) -> List:
        """杂交操作"""
        offspring = []
        
        if len(elites) < 2:
            return offspring
        
        num_offspring = self.config.pool_size - self.config.elite_count
        
        for _ in range(num_offspring):
            if random.random() < self.config.crossover_rate:
                parent1, parent2 = random.sample(elites, 2)
                child = self._create_offspring(parent1, parent2)
                offspring.append(child)
            else:
                offspring.append(copy.deepcopy(random.choice(elites)))
        
        return offspring
    
    def _create_offspring(self, parent1, parent2):
        """创建子代"""
        child = copy.deepcopy(parent1)
        child.id = f"defender_offspring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        child.generation_count = 0
        child.reward_history = []
        child.avg_reward = (parent1.avg_reward + parent2.avg_reward) / 2
        
        # 合并攻击类型表现
        child.attack_type_performance = {}
        all_types = set(parent1.attack_type_performance.keys()) | set(parent2.attack_type_performance.keys())
        for attack_type in all_types:
            score1 = parent1.attack_type_performance.get(attack_type, 0.5)
            score2 = parent2.attack_type_performance.get(attack_type, 0.5)
            child.attack_type_performance[attack_type] = (score1 + score2) / 2
        
        return child
    
    def _mutate(self, defenders: List) -> List:
        """突变操作"""
        for defender in defenders:
            if random.random() < self.config.mutation_rate:
                self._apply_mutation(defender)
        
        return defenders
    
    def _apply_mutation(self, defender):
        """应用突变"""
        # 轻微调整奖励历史
        if defender.reward_history:
            mutation_factor = random.gauss(1.0, 0.1)
            defender.reward_history[-1] *= mutation_factor
            defender.avg_reward = sum(defender.reward_history[-20:]) / min(20, len(defender.reward_history))
        
        # 随机调整攻击类型表现
        for attack_type in defender.attack_type_performance:
            if random.random() < 0.3:
                defender.attack_type_performance[attack_type] *= random.gauss(1.0, 0.15)
                defender.attack_type_performance[attack_type] = max(0.0, min(1.0, defender.attack_type_performance[attack_type]))
    
    def _select_next_generation(self, current: List, offspring: List, 
                               fitness_scores: List) -> List:
        """选择下一代"""
        all_defenders = current + offspring
        all_fitness = fitness_scores + [o.avg_reward for o in offspring]
        
        sorted_indices = sorted(
            range(len(all_fitness)),
            key=lambda i: all_fitness[i],
            reverse=True
        )
        
        next_generation = [all_defenders[i] for i in sorted_indices[:self.config.pool_size]]
        
        # 重置更新预算
        for defender in next_generation:
            defender.reset_update_budget()
        
        return next_generation
    
    def get_evolution_stats(self) -> Dict:
        """获取进化统计"""
        if not self.evolution_history:
            return {"total_generations": 0}
        
        recent = self.evolution_history[-100:]
        
        return {
            "total_generations": len(self.evolution_history),
            "avg_fitness": sum(e["avg_fitness"] for e in recent) / len(recent),
            "best_fitness": max(e["best_fitness"] for e in self.evolution_history),
            "fitness_trend": recent[-1]["avg_fitness"] - recent[0]["avg_fitness"] if len(recent) > 1 else 0
        }
