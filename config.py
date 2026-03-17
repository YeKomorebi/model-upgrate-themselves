# config/config.py - 新增PPO配置

@dataclass
class PPOConfig:
    """PPO-style约束配置"""
    enabled: bool = True
    
    # 参考模型配置
    ref_model_update_frequency: int = 100  # 每100代更新参考模型
    ref_model_type: str = "frozen"  # frozen, ema, periodic
    
    # KL散度约束
    kl_coefficient: float = 0.2      # KL惩罚系数
    kl_target: float = 0.02          # 目标KL散度
    kl_clip_min: float = 0.001       # 最小KL系数
    kl_clip_max: float = 1.0         # 最大KL系数
    adaptive_kl: bool = True         # 自适应KL系数
    
    # PPO Clip机制
    clip_epsilon: float = 0.2        # PPO clip范围
    clip_value: bool = True          # 是否clip价值函数
    value_loss_coeff: float = 0.5    # 价值函数损失系数
    
    # 梯度约束
    max_grad_norm: float = 1.0       # 最大梯度范数
    gradient_accumulation_steps: int = 4

@dataclass
class MentorConfig:
    """导师制配置"""
    enabled: bool = True
    # ... 原有配置 ...
    # 新增：与PPO的集成
    use_ppo_constraint: bool = True  # 是否使用PPO约束
    ppo: PPOConfig = field(default_factory=PPOConfig)

@dataclass
class SystemConfig:
    """系统总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    mentor: MentorConfig = field(default_factory=MentorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)  # 顶层PPO配置
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    # ... 其他配置 ...
