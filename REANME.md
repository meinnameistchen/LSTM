# CNN Project

## 项目结构
该项目实现了一个基于 CNN 的分类模型，包含以下功能：
- 数据加载与预处理
- 模型定义与训练
- 测试与评估
- 结果保存（模型、混淆矩阵、评估指标）
Predictive_Maintaince/ 
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── config.py                # 各类配置文件
│
├── models/
│   ├── __init__.py
│   └── cnn1d.py                 # 模型结构
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # 自定义Dataset类（如果有）
│   └── data_loader.py          # 封装DataLoader逻辑
│
├── train/
│   ├── __init__.py
│   ├── train.py                # 训练函数
│   └── train_main.py           # 可执行的主训练入口
│
├── test/
│   ├── __init__.py
│   ├── test.py                 # 测试与评估函数
│   └── test_main.py            # 测试主入口
│
├── utils/
│   ├── __init__.py
│   └── utils.py                # 通用工具，如绘图、日志
│
├── saved_models/
│   └── best_model.pt           # 保存模型
│
├── results/
│   ├── confusion_matrix.png
│   └── metrics.txt
│
└── run.py                      # 顶层统一运行入口（可选）


## 运行方式
### 安装依赖
```bash
pip install -r requirements.txt
### 训练
```bash
python run.py --mode train

### 测试
```bash
python run.py --mode test

