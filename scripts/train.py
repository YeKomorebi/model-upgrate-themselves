#!/usr/bin/env python3
# scripts/train.py
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import SystemConfig
from src.trainer.trainer import MentorEvolutionTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="防御者导师制进化训练系统")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="配置文件路径")
    parser.add_argument("--generations", type=int, default=None, help="训练代数")
    parser.add_argument("--pool-size", type=int, default=None, help="防御者池大小")
    parser.add_argument("--device", type=str, default=None, help="计算设备")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if os.path.exists(args.config):
        config = SystemConfig.from_yaml(args.config)
    else:
        config = SystemConfig.get_default()
    
    if args.generations:
        config.evolution.num_generations = args.generations
    if args.pool_size:
        config.evolution.pool_size = args.pool_size
    if args.device:
        config.device = args.device
    
    print("📋 配置信息:")
    print(f"   训练代数: {config.evolution.num_generations}")
    print(f"   防御者池: {config.evolution.pool_size}")
    print(f"   设备: {config.device}")
    print(f"   导师制: {'启用' if config.mentor.enabled else '禁用'}")
    print(f"   PPO约束: {'启用' if config.ppo.enabled else '禁用'}")
    
    trainer = MentorEvolutionTrainer(config)
    
    if args.resume:
        print(f"🔄 从检查点恢复: {args.resume}")
    
    trainer.train()

if __name__ == "__main__":
    main()
