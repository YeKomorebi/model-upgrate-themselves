from typing import List, Dict, Any, Optional
import logging
import copy
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class EvolutionaryOperations:
    """
    进化操作 - 已修复版本
    
    🔧 修复：深拷贝问题、属性清理
    """
    
    def __init__(self, config):
        self.config = config.evolution if hasattr(config, 'evolution') else {}
        
        self.mutation_rate = getattr(self.config, 'mutation_rate', 0.1)
        self.crossover_rate = getattr(self.config, 'crossover_rate', 0.7)
        self.elite_count = getattr(self.config, 'elite_count', 5)
        
        # 🔧 修复：定义不应继承的属性
        self.non_inheritable_attrs = [
            'learning_history',
            'mentor_id',
            'mentee_ids',
            'last_interaction',
            'satisfaction_score',
            '_logits_cache',
            'generation_created',
            'experience'
        ]
    
    def evolve(self, population: List, generation: int,
              elite_count: Optional[int] = None) -> List:
        """
        执行进化操作
        
        🔧 修复：深拷贝时清理不应继承的属性
        """
        try:
            if not population:
                logger.warning("空种群，无法进化")
                return []
            
            elite_count = elite_count or self.elite_count
            elite_count = min(elite_count, len(population))
            
            # 选择精英
            elites = self._select_elites(population, elite_count)
            
            # 生成新个体
            offspring = []
            while len(offspring) < len(population) - elite_count:
                parent1, parent2 = self._select_parents(population)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                    offspring.append(child1)
                    if len(offspring) < len(population) - elite_count:
                        offspring.append(child2)
                else:
                    child = self._mutate(copy.deepcopy(parent1))
                    offspring.append(child)
            
            # 🔧 修复：清理不应继承的属性
            for individual in offspring:
                self._clean_inherited_attrs(individual, generation)
            
            new_population = elites + offspring
            
            logger.info(f"进化完成：{len(elites)} 精英 + {len(offspring)} 后代 = {len(new_population)}")
            return new_population
            
        except Exception as e:
            logger.error(f"进化操作失败：{e}")
            return population
    
    def _select_elites(self, population: List, count: int) -> List:
        """选择精英"""
        sorted_pop = sorted(
            population,
            key=lambda x: getattr(x, 'avg_reward', 0),
            reverse=True
        )
        return sorted_pop[:count]
    
    def _select_parents(self, population: List) -> tuple:
        """选择父母"""
        # 锦标赛选择
        tournament_size = 3
        
        def tournament():
            candidates = random.sample(population, min(tournament_size, len(population)))
            return max(candidates, key=lambda x: getattr(x, 'avg_reward', 0))
        
        parent1 = tournament()
        parent2 = tournament()
        
        return parent1, parent2
    
    def _crossover(self, parent1, parent2) -> tuple:
        """杂交"""
        try:
            # 🔧 修复：深拷贝时清理属性
            child1 = self._clean_copy(parent1)
            child2 = self._clean_copy(parent2)
            
            # 单点交叉
            if hasattr(parent1, 'parameters') and hasattr(parent2, 'parameters'):
                params1 = getattr(parent1, 'parameters', {})
                params2 = getattr(parent2, 'parameters', {})
                
                if params1 and params2:
                    keys = list(params1.keys())
                    if keys:
                        crossover_point = random.randint(0, len(keys))
                        
                        # 🔧 修复：安全交换参数
                        for i, key in enumerate(keys):
                            if i >= crossover_point:
                                if key in params2:
                                    params1[key] = params2[key]
                                if key in params1:
                                    params2[key] = params1[key]
            
            # 生成新 ID
            child1.id = f"defender_offspring_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            child2.id = f"defender_offspring_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"杂交失败：{e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def _mutate(self, individual) -> Any:
        """突变"""
        try:
            # 🔧 修复：深拷贝时清理属性
            mutant = self._clean_copy(individual)
            
            # 参数突变
            if hasattr(mutant, 'parameters'):
                params = getattr(mutant, 'parameters', {})
                for key in params:
                    if random.random() < self.mutation_rate:
                        # 🔧 修复：安全的突变
                        if isinstance(params[key], (int, float)):
                            mutation_factor = random.gauss(1.0, 0.1)
                            params[key] = params[key] * mutation_factor
            
            # 生成新 ID
            mutant.id = f"defender_mutant_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            return mutant
            
        except Exception as e:
            logger.error(f"突变失败：{e}")
            return copy.deepcopy(individual)
    
    def _clean_copy(self, individual) -> Any:
        """
        深拷贝并清理不应继承的属性
        
        🔧 修复：新增方法，解决漏洞 10
        """
        try:
            copied = copy.deepcopy(individual)
            
            # 清理不应继承的属性
            for attr in self.non_inheritable_attrs:
                if hasattr(copied, attr):
                    setattr(copied, attr, None)
            
            # 重置世代信息
            if hasattr(copied, 'generation_created'):
                copied.generation_created = 0
            if hasattr(copied, 'experience'):
                copied.experience = 0
            
            return copied
            
        except Exception as e:
            logger.warning(f"清理拷贝失败：{e}")
            return copy.deepcopy(individual)
    
    def _clean_inherited_attrs(self, individual, generation: int):
        """清理继承的属性"""
        if hasattr(individual, 'generation_created'):
            individual.generation_created = generation
        if hasattr(individual, 'experience'):
            individual.experience = 0
        if hasattr(individual, 'learning_history'):
            individual.learning_history = []
        if hasattr(individual, 'reward_history'):
            # 保留部分历史
            individual.reward_history = individual.reward_history[-5:]
    
    def get_statistics(self, population: List) -> Dict[str, Any]:
        """获取种群统计"""
        if not population:
            return {'count': 0}
        
        rewards = [getattr(p, 'avg_reward', 0) for p in population]
        
        return {
            'count': len(population),
            'avg_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'std_reward': (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5
        }
