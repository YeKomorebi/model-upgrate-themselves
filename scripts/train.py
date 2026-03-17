import argparse
import os
import sys
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='防御者导师制进化训练系统')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--generations', type=int, default=100,
                       help='训练代数')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='日志目录')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 🔧 修复：配置文件验证和警告
    if os.path.exists(args.config):
        logger.info(f"✅ 加载配置文件：{args.config}")
        # config = SystemConfig.from_yaml(args.config)
    else:
        logger.warning(f"⚠️ 配置文件不存在：{args.config}")
        logger.info("📋 使用默认配置")
        # config = SystemConfig.get_default()
    
    logger.info(f"🚀 训练开始于 {datetime.now().isoformat()}")
    
    # 训练逻辑...
    
    logger.info(f"✅ 训练完成于 {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
