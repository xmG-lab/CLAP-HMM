# CLAP-HMM: A Hybrid Deep Learning and HMM Framework for Gene Prediction

CLAP-HMM（**C**NN-**L**STM-**A**ttention with **P**rotHint and **H**idden **M**arkov **M**odel）是一种集成深度神经网络与传统隐马尔可夫模型的基因结构预测框架，特别适用于植物全基因组的抗性基因预测等任务。

该模型融合了 CNN 提取局部序列特征、BiLSTM 建模长程依赖、注意力机制进行权重增强、ProtHint 提供蛋白质同源支持信息，以及 HMM 精炼边界结构，为基因功能与结构注释提供了一体化解决方案。

---

## 📁 项目结构

```text
CLAP-HMM/
├── data/                         # 存放输入序列、标签、预测输出
│   ├── input/                    # 原始FASTA输入序列
│   ├── labels/                   # 真实标签（GFF/BED）
│   └── output/                   # 预测结果输出
│
├── models/                       # 模型结构定义（CNN-LSTM-Attention + HMM）
│   ├── cnn_lstm_attention.py     # 特征提取主模型
│   ├── hmm_module.py             # HMM后处理模块
│   └── fusion.py                 # 特征融合策略
│
├── prothint/                     # 与ProtHint对接的脚本或结果
│   └── hints.gff                 # 蛋白同源比对输出
│
├── scripts/                      # 数据处理、评估脚本
│   ├── evaluate.py               # 模型评估指标计算
│   └── preprocess.py             # FASTA转模型输入等预处理
│
├── configs/                      # YAML配置文件
│   └── default.yaml              # 默认参数设置
│
├── notebooks/                    # 示例Jupyter Notebook
│   └── demo.ipynb                # 使用范例
│
├── main.py                       # 主运行入口
├── requirements.txt              # 所需Python依赖
├── README.md                     # 项目说明文档
└── LICENSE                       # 开源协议
