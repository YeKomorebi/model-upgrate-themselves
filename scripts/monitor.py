import argparse
import json
import os
import logging
from typing import Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_log(log_dir: str) -> List[Dict]:
    """加载训练日志"""
    log_path = os.path.join(log_dir, 'training_log.json')
    
    if not os.path.exists(log_path):
        logger.warning(f"日志文件不存在：{log_path}")
        return []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载日志失败：{e}")
        return []

def show_summary(logs: List[Dict]):
    """显示训练摘要"""
    if not logs:
        print("无日志数据")
        return
    
    print("=" * 60)
    print("训练摘要")
    print("=" * 60)
    print(f"总代数：{len(logs)}")
    print(f"开始时间：{logs[0].get('timestamp', 'N/A')}")
    print(f"结束时间：{logs[-1].get('timestamp', 'N/A')}")
    
    if logs and 'reward' in logs[-1]:
        rewards = [log.get('reward', 0) for log in logs]
        print(f"平均奖励：{sum(rewards)/len(rewards):.4f}")
        print(f"最终奖励：{rewards[-1]:.4f}")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='日志目录')
    parser.add_argument('--live', action='store_true',
                       help='实时监控')
    parser.add_argument('--mentors', action='store_true',
                       help='显示导师统计')
    parser.add_argument('--ppo', action='store_true',
                       help='显示 PPO 统计')
    
    args = parser.parse_args()
    
    logs = load_training_log(args.log_dir)
    show_summary(logs)

if __name__ == "__main__":
    main()
