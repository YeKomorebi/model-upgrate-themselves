"""
导师模块统一入口 - 已修复版本
"""

from .selector import MentorSelector
from .pairing import MentorPairing
from .distillation import KnowledgeDistillation
from .evaluator import MentorEvaluator

__all__ = [
    'MentorSelector',
    'MentorPairing',
    'KnowledgeDistillation',
    'MentorEvaluator'
]

__version__ = '1.1.0'  # 🔧 修复：添加版本号
