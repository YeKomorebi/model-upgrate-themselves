#!/usr/bin/env python3
# scripts/monitor.py
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="训练监控工具")
    parser.add_argument("--log-dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--mentors", action="store_true", help="查看导师统计")
    parser.add_argument("--ppo", action="store_true", help="查看PPO统计")
    parser.add_argument("--evolution", action="store_true", help="查看进化统计")
    parser.add_argument("--live", action="store_true", help="实时监控")
    return parser.parse_args()

def load_training_log(log_dir: str):
    log_path = os.path.join(log_dir, "training_log.json")
    if not os.path.exists(log_path):
        return []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def show_summary(logs: list):
    if not logs:
        print("❌ 暂无训练日志")
        return
    
    latest = logs[-1]
    print(f"\n{'='*60}")
    print("📊 训练摘要")
    print(f"{'='*60}")
    print(f"当前代数: {latest.get('generation', 'N/A')}")
    print(f"平均奖励: {latest.get('avg_reward', 0):.3f}")
    print(f"最佳奖励: {latest.get('best_reward', 0):.3f}")
    print(f"导师数量: {latest.get('num_mentors', 0)}")
    print(f"学生数量: {latest.get('num_mentees', 0)}")
    
    if 'ppo_stats' in latest:
        ppo = latest['ppo_stats']
        print(f"KL系数: {ppo.get('kl_coefficient', 0):.4f}")

def show_mentor_stats(logs: list):
    print(f"\n{'='*60}")
    print("🎓 导师统计")
    print(f"{'='*60}")
    
    if not logs:
        print("暂无数据")
        return
    
    for log in logs[-10:]:
        print(f"代{log.get('generation', '?')}: 导师={log.get('num_mentors', 0)}, 学生={log.get('num_mentees', 0)}")

def show_ppo_stats(logs: list):
    print(f"\n{'='*60}")
    print("🔒 PPO约束统计")
    print(f"{'='*60}")
    
    ppo_logs = [l for l in logs if 'ppo_stats' in l]
    
    if not ppo_logs:
        print("暂无PPO数据")
        return
    
    for log in ppo_logs[-10:]:
        ppo = log['ppo_stats']
        print(f"代{log.get('generation', '?')}: KL系数={ppo.get('kl_coefficient', 0):.4f}")

def main():
    args = parse_args()
    
    logs = load_training_log(args.log_dir)
    
    if args.mentors:
        show_mentor_stats(logs)
    elif args.ppo:
        show_ppo_stats(logs)
    elif args.evolution:
        show_summary(logs)
    else:
        show_summary(logs)
        show_mentor_stats(logs)
        show_ppo_stats(logs)

if __name__ == "__main__":
    main()
