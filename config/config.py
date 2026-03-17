# config/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import yaml
import os

class ModelSize(Enum):
    """模型尺寸枚举"""
    TINY = "Qwen/Qwen2.5-0.5B"
    SMALL = "Qwen/Qwen2.5-1.5B"
    MEDIUM = "Qwen/Qwen2.5-3B"
    LARGE = "Qwen/Qwen2.5-7B"

@dataclass
class ModelConfig:
    """模型配置"""
    defender_model: str = ModelSize.SMALL.value
    attacker_model: str = ModelSize.SMALL.value
    judge_model: str = ModelSize.MEDIUM.value
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_len: int = 512

@dataclass
class PPOConfig:
    """PPO-style约束配置"""
    enabled: bool = True
    ref_model_update_frequency: int = 100
    ref_model_type: str = "ema"  # frozen, ema, periodic
    kl_coefficient: float = 0.2
    kl_target: float = 0.02
    kl_clip_min: float = 0.001
    kl_clip_max: float = 1.0
    adaptive_kl: bool = True
    clip_epsilon: float = 0.2
    clip_value: bool = True
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4

@dataclass
class MentorConfig:
    """导师制配置"""
    enabled: bool = True
    min_avg_reward: float = 0.75
    min_diversity_score: float = 0.7
    min_stability: float = 0.8
    min_generations: int = 50
    max_mentees_per_mentor: int = 3
    num_mentors: int = 3
    pairing_strategy: str = "best_match"
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.7
    distillation_beta: float = 0.3
    guidance_frequency: int = 5
    evaluation_frequency: int = 20
    mentor_selection_frequency: int = 50
    use_ppo_constraint: bool = True

@dataclass
class EvolutionConfig:
    """进化配置"""
    pool_size: int = 5
    elite_count: int = 2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    batch_size: int = 2
    num_generations: int = 500

@dataclass
class KnowledgeBaseConfig:
    """知识库配置"""
    chroma_path: str = "./data/chroma_db"
    max_documents: int = 10000
    retrieval_k: int = 3
    save_frequency: int = 10

@dataclass
class SystemConfig:
    """系统总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    mentor: MentorConfig = field(default_factory=MentorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    
    # 训练配置
    output_dir: str = "./output"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"
    
    # 监控配置
    log_frequency: int = 1
    save_checkpoint_frequency: int = 50
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 递归创建dataclass
        def create_dataclass(cls, data_dict):
            if hasattr(cls, '__dataclass_fields__'):
                kwargs = {}
                for field_name, field_obj in cls.__dataclass_fields__.items():
                    if field_name in data_dict:
                        field_type = field_obj.type
                        if hasattr(field_type, '__dataclass_fields__'):
                            kwargs[field_name] = create_dataclass(field_type, data_dict[field_name])
                        else:
                            kwargs[field_name] = data_dict[field_name]
                return cls(**kwargs)
            return data_dict
        
        return create_dataclass(cls, data)
    
    def to_yaml(self, path: str):
        """保存配置到YAML文件"""
        def convert_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(convert_to_dict(self), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def get_default(cls) -> "SystemConfig":
        """获取默认配置"""
        return cls()
