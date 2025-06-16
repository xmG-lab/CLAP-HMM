# 🌱 CLAP-HMM: A Hybrid Deep Learning and HMM Framework for Gene Prediction

CLAP-HMM（**C**NN-**L**STM-**A**ttention with **P**rotHint and **H**idden **M**arkov **M**odel）is a gene structure prediction framework integrating deep neural networks and traditional hidden Markov models, and is particularly suitable for tasks such as the prediction of resistance genes in the entire plant genome.

This model integrates CNN for extracting local sequence features, LSTM for modeling long-range dependencies, attention mechanism for weight enhancement, ProtHint for providing protein homology support information, and HMM for prediction, providing an integrated solution for gene function prediction and annotation.

---

## 🚀 快速开始
### 1. 克隆项目
git clone https://github.com/xmG-lab/CLAP-HMM.git

cd CLAP-HMM

pip install -r requirements.txt

### 2. 准备数据
输入格式：FASTA 格式序列（.fna / .fa）

标签格式：GFF3 或 BED

同源蛋白文件：从 UniProt / OrthoDB / 自建数据库中提取

ProtHint 安装与执行：https://github.com/gatech-genemark/ProtHint

### 3. 运行模型预测
python main.py --config configs/default.yaml

### 4. 模型评估
python scripts/evaluate.py --pred data/output/pred.gff --true data/labels/popular.gff

---

## 🧠 模型架构
CLAP-HMM 由三大部分组成：

序列特征提取模块（CNN → LSTM → Attention）

外源蛋白信息融合模块（ProtHint hints）

结构优化模块（HMM）

![figure1](https://github.com/user-attachments/assets/ad16263c-d7e6-4eae-8b85-4dcaffc95577)

---

## 📊 示例结果
| 模型                  | Accuracy  | Precision | Recall    | MCC       |
| ------------------- | --------- | --------- | --------- | --------- |
| Baseline (CNN-LSTM) | 91.2%     | 0.889     | 0.881     | 0.871     |
| +ProtHint           | 92.8%     | 0.913     | 0.901     | 0.894     |
| +HMM后处理             | **94.5%** | **0.936** | **0.928** | **0.917** |

---

## 🧬 数据来源
Populus alba 全基因组序列（NCBI / GigaDB）

抗性基因注释数据：PlantRGA, PRGdb

蛋白数据库：UniProtKB/SwissProt, RefSeq

ProtHint：提供内含子-外显子边界预测信息

---

## 💬 联系与支持
欢迎提交 Issues 或联系：

📧 your.email@example.com

🧑‍💻 Your GitHub

---

## 📄 License
本项目基于 MIT 开源协议发布，详见 LICENSE 文件。

