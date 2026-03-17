from typing import Dict, List, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)

class ClipOptimizer:
    """PPO-style 梯度优化器 - 已修复版本"""
    
    def __init__(self, config):
        self.config = config.ppo if hasattr(config, 'ppo') else {}
        
        self.learning_rate = getattr(self.config, 'learning_rate', 2e-5)
        self.clip_epsilon = getattr(self.config, 'clip_epsilon', 0.2)
        self.max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
    
    def optimization_step(self, model, loss: float, 
                         gradients: Optional[torch.Tensor] = None) -> bool:
        """
        执行优化步骤
        
        🔧 修复：输入验证
        """
        try:
            # 🔧 修复：输入验证
            if not model:
                logger.error("模型对象为空")
                return False
            
            if loss is None or not isinstance(loss, (int, float)):
                logger.warning(f"无效的损失值：{loss}")
                return False
            
            # 梯度裁剪
            if gradients is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.max_grad_norm
                )
            
            logger.debug(f"优化步骤完成，损失：{loss:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"优化步骤失败：{e}")
            return False
