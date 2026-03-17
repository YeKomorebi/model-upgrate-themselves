# 🛡️ 防御者导师制进化训练系统

基于PPO约束的多智能体对抗进化训练系统，融合导师制知识蒸馏与KL稳定性约束。

## ✨ 核心特性

- 🎓 **导师制机制**: 优秀防御者指导新防御者，加速知识传承
- 🔒 **PPO约束**: KL散度约束 + Clip机制，防止训练崩溃
- 📚 **知识库检索**: 三位一体知识库支持RAG增强
- 🧬 **进化操作**: 杂交、突变、选择，持续优化防御策略
- 📊 **完整监控**: 训练过程全记录，支持断点续训

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/evolution_defense_system.git
cd evolution_defense_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
