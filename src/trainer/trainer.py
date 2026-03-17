# src/trainer/trainer.py
import os
import json
import torch
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from config.config import SystemConfig
from src.models.defender import DefenderModel
from src.models.attacker import AttackerModel
from src.models.judge import JudgeModel
from src.models.reference_model import ReferenceModel
from src.mentor.selector import MentorSelector
from src.mentor.pairing import MentorPairing
from src.mentor.distillation import KnowledgeDistillation
from src.mentor.evaluator import MentorEvaluator
from src.ppo.kl_constraint import KLConstraint, KLConstraintConfig
from src.ppo.clip_optimizer import ClipOptimizer, ClipOptimizerConfig
from src.knowledge.knowledge_base import KnowledgeBase
from src.evolution.evolutionary_ops import EvolutionaryOperations

class MentorEvolutionTrainer:
    """整合导师制+PPO约束的训练器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = config.device
        
        self.defenders: List[DefenderModel] = []
        self.attacker: Optional[AttackerModel] = None
        self.judge: Optional[JudgeModel] = None
        self.knowledge_base: Optional[KnowledgeBase] = None
        
        self.mentor_selector = MentorSelector(config)
        self.mentor_pairing = MentorPairing(config)
        self.knowledge_distillation = KnowledgeDistillation(config)
        self.mentor_evaluator = MentorEvaluator(config)
        
        self.evolution_ops = EvolutionaryOperations(config)
        
        self.current_mentors: List[DefenderModel] = []
        
        self.use_ppo = config.ppo.enabled
        self.ref_model: Optional[ReferenceModel] = None
        self.kl_constraint: Optional[KLConstraint] = None
        self.defender_optimizers: Dict = {}
        
        if self.use_ppo:
            self._init_ppo_components()
        
        self.training_stats = {
            "start_time": None,
            "generations_completed": 0,
            "best_reward": 0.0,
            "mentor_selections": 0,
            "distillation_sessions": 0,
            "mentor_evaluations": 0
        }
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _init_ppo_components(self):
        """初始化PPO组件"""
        print("🔒 初始化PPO约束组件...")
        
        kl_config = KLConstraintConfig(
            kl_coefficient=self.config.ppo.kl_coefficient,
            kl_target=self.config.ppo.kl_target,
            kl_clip_min=self.config.ppo.kl_clip_min,
            kl_clip_max=self.config.ppo.kl_clip_max,
            adaptive_kl=self.config.ppo.adaptive_kl
        )
        self.kl_constraint = KLConstraint(kl_config)
        
        self.ref_model = ReferenceModel(
            self.config.model.defender_model,
            self.config.device,
            self.config.ppo.ref_model_type
        )
    
    def initialize(self):
        """初始化系统"""
        print("🚀 初始化训练系统...")
        
        print(f"📚 初始化 {self.config.evolution.pool_size} 个防御者...")
        for i in range(self.config.evolution.pool_size):
            defender = DefenderModel(self.config.model.defender_model, self.config, self.device)
            self.defenders.append(defender)
        
        print("⚔️ 初始化攻击者...")
        self.attacker = AttackerModel(self.config.model.attacker_model, self.config, self.device)
        
        print("⚖️ 初始化法官...")
        self.judge = JudgeModel(self.config.model.judge_model, self.config, self.device)
        
        print("📖 初始化知识库...")
        self.knowledge_base = KnowledgeBase(self.config)
        
        if self.use_ppo:
            print("🔒 初始化PPO优化器...")
            for defender in self.defenders:
                opt_config = ClipOptimizerConfig(
                    learning_rate=2e-5,
                    clip_epsilon=self.config.ppo.clip_epsilon,
                    value_loss_coeff=self.config.ppo.value_loss_coeff,
                    max_grad_norm=self.config.ppo.max_grad_norm,
                    gradient_accumulation_steps=self.config.ppo.gradient_accumulation_steps
                )
                self.defender_optimizers[defender.id] = ClipOptimizer(
                    defender.model, opt_config, self.kl_constraint
                )
        
        if self.config.mentor.enabled:
            print("🎓 初始导师选拔...")
            self._select_mentors()
        
        self.training_stats["start_time"] = datetime.now().isoformat()
        print("✅ 初始化完成！\n")
    
    def _select_mentors(self):
        """选拔导师"""
        self.current_mentors = self.mentor_selector.select_mentors(self.defenders)
        
        for mentor in self.current_mentors:
            mentor.is_mentor = True
            mentor.mentor_since = self.training_stats["generations_completed"]
        
        mentees = [d for d in self.defenders if not d.is_mentor and not d.has_mentor]
        self.mentor_pairing.pair_mentors_mentees(self.current_mentors, mentees)
        
        self.training_stats["mentor_selections"] += 1
        
        print(f"🎓 选出 {len(self.current_mentors)} 名导师")
    
    def _evaluate_mentors(self):
        """评估导师"""
        review_results = self.mentor_evaluator.review_mentors(
            self.current_mentors, 
            self.training_stats["generations_completed"]
        )
        
        self.training_stats["mentor_evaluations"] += 1
        
        print(f"📊 导师评估: 优秀:{len(review_results['excellent'])} 合格:{len(review_results['qualified'])} 需改进:{len(review_results['needs_improvement'])} 淘汰:{len(review_results['dismissed'])}")
        
        self._promote_new_mentors()
    
    def _promote_new_mentors(self):
        """晋升新导师"""
        for defender in self.defenders:
            if not defender.is_mentor and defender.generation_count >= self.config.mentor.min_generations:
                if defender.has_mentor:
                    feedback = defender.get_mentor_feedback()
                    if feedback.get("recommend_promotion", False) and defender.avg_reward >= 0.75:
                        defender.is_mentor = True
                        defender.mentor_since = self.training_stats["generations_completed"]
                        self.current_mentors.append(defender)
                        print(f"🎓 晋升新导师: {defender.id[:40]}...")
    
    def _knowledge_distillation(self):
        """执行知识蒸馏"""
        if not self.current_mentors:
            return
        
        distillation_count = 0
        
        for mentor in self.current_mentors:
            for mentee in mentor.mentees:
                challenge = f"安全挑战_{distillation_count}"
                
                guidance = self.knowledge_distillation.generate_mentor_guidance(
                    mentor, challenge, self.knowledge_base.query(challenge, k=3)
                )
                
                student_answer, _ = mentee.answer(challenge, self.knowledge_base.query(challenge, k=3))
                
                loss = self.knowledge_distillation.student_learn(
                    mentee, guidance, student_answer, 
                    self.ref_model if self.use_ppo else None
                )
                
                distillation_count += 1
        
        self.training_stats["distillation_sessions"] += 1
        print(f"📖 完成 {distillation_count} 次知识蒸馏")
    
    def _ppo_update_step(self, generation: int):
        """执行PPO更新"""
        if not self.use_ppo:
            return
        
        print(f"🔒 执行PPO约束更新...")
        
        for defender in self.defenders:
            if defender.check_update_budget():
                optimizer = self.defender_optimizers.get(defender.id)
                if optimizer:
                    update_stats = defender.ppo_update(
                        self.ref_model,
                        self.kl_constraint,
                        optimizer,
                        generation
                    )
                    
                    if update_stats and 'kl_penalty' in update_stats:
                        print(f"   {defender.id[:30]}... | KL:{update_stats.get('kl_penalty', 0):.4f}")
        
        for defender in self.defenders:
            defender.reset_update_budget()
    
    def _update_reference_model(self):
        """更新参考模型"""
        if not self.use_ppo or not self.ref_model:
            return
        
        print(f"🔄 更新参考模型...")
        
        best_defender = max(self.defenders, key=lambda d: d.avg_reward)
        self.ref_model.update_from_model(best_defender, self.config.ppo.ref_model_type)
        
        kl_stats = self.ref_model.get_kl_stats()
        print(f"   参考模型更新次数: {kl_stats['update_count']}, 平均KL: {kl_stats['avg_kl']:.4f}")
    
    def train_generation(self, generation: int):
        """训练一代"""
        
        print(f"\n{'='*60}")
        print(f"📊 第 {generation}/{self.config.evolution.num_generations} 代")
        print(f"{'='*60}")
        
        challenges = self.attacker.generate_challenges(
            self.defenders, 
            num_challenges=len(self.defenders) * 3
        )
        
        for defender in self.defenders:
            for challenge in challenges[:3]:
                context = self.knowledge_base.query(challenge, k=3)
                answer, confidence = defender.answer(challenge, context)
                
                reward, feedback = self.judge.evaluate(
                    challenge, answer, context, defender.id
                )
                
                defender.update_reward(reward, attack_type=feedback.get("attack_type", "general"))
        
        self.knowledge_base.update(challenges, [d.answer(c)[0] for c, d in zip(challenges, self.defenders)])
        
        self.defenders = self.evolution_ops.evolve(self.defenders, generation)
        
        if self.config.mentor.enabled:
            if generation % self.config.mentor.guidance_frequency == 0:
                self._knowledge_distillation()
            
            if generation % self.config.mentor.evaluation_frequency == 0:
                self._evaluate_mentors()
            
            if generation % self.config.mentor.mentor_selection_frequency == 0:
                self._select_mentors()
        
        if self.use_ppo:
            self._ppo_update_step(generation)
            
            if generation % self.config.ppo.ref_model_update_frequency == 0:
                self._update_reference_model()
        
        self.training_stats["generations_completed"] = generation
        current_best = max(d.avg_reward for d in self.defenders)
        if current_best > self.training_stats["best_reward"]:
            self.training_stats["best_reward"] = current_best
        
        if generation % self.config.log_frequency == 0:
            self._log_generation(generation)
        
        if generation % self.config.save_checkpoint_frequency == 0:
            self._save_checkpoint(generation)
    
    def _log_generation(self, generation: int):
        """记录日志"""
        log_entry = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "avg_reward": sum(d.avg_reward for d in self.defenders) / len(self.defenders),
            "best_reward": max(d.avg_reward for d in self.defenders),
            "num_mentors": len(self.current_mentors),
            "num_mentees": sum(len(d.mentees) for d in self.current_mentors),
            "training_stats": self.training_stats
        }
        
        if self.use_ppo and self.kl_constraint:
            log_entry["ppo_stats"] = {
                "kl_coefficient": self.kl_constraint.current_kl_coeff,
                "kl_constraint_stats": self.kl_constraint.get_stats()
            }
        
        log_path = os.path.join(self.config.log_dir, f"training_log.json")
        
        logs = []
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        print(f"📈 平均奖励: {log_entry['avg_reward']:.3f} | 最佳奖励: {log_entry['best_reward']:.3f}")
        print(f"🎓 导师: {log_entry['num_mentors']} | 学生: {log_entry['num_mentees']}")
    
    def _save_checkpoint(self, generation: int):
        """保存检查点"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_gen_{generation}.json")
        
        checkpoint = {
            "generation": generation,
            "config": self.config.__dict__,
            "training_stats": self.training_stats,
            "defenders": [d.id for d in self.defenders],
            "mentors": [m.id for m in self.current_mentors]
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        for defender in self.defenders:
            defender.save_checkpoint(
                os.path.join(self.config.checkpoint_dir, f"defender_{defender.id[:20]}_gen_{generation}.json")
            )
        
        print(f"💾 检查点已保存: {checkpoint_path}")
    
    def train(self):
        """开始训练"""
        self.initialize()
        
        print(f"\n🎯 开始训练 {self.config.evolution.num_generations} 代...\n")
        
        for generation in range(1, self.config.evolution.num_generations + 1):
            self.train_generation(generation)
        
        print(f"\n{'='*60}")
        print("✅ 训练完成！")
        print(f"{'='*60}")
        print(f"📊 总代数: {self.training_stats['generations_completed']}")
        print(f"🏆 最佳奖励: {self.training_stats['best_reward']:.3f}")
        print(f"🎓 导师选拔次数: {self.training_stats['mentor_selections']}")
        print(f"📖 知识蒸馏次数: {self.training_stats['distillation_sessions']}")
        print(f"📋 导师评估次数: {self.training_stats['mentor_evaluations']}")
        
        final_stats_path = os.path.join(self.config.output_dir, "final_training_stats.json")
        with open(final_stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 最终统计已保存: {final_stats_path}")
