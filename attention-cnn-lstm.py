import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from Bio import SeqIO
import joblib
from torch.cuda.amp import autocast, GradScaler

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, lstm_output):
        # LSTM output shape: (batch_size, seq_len, hidden_dim)
        attention_scores = torch.matmul(lstm_output, self.attention_weights)
        attention_scores = torch.softmax(attention_scores, dim=1)  # Shape: (batch_size, seq_len, 1)
        context_vector = torch.sum(lstm_output * attention_scores, dim=1)  # Shape: (batch_size, hidden_dim)
        return context_vector, attention_scores

class CNNLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size):
        super(CNNLSTMWithAttention, self).__init__()
        # 第一层卷积和池化
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 第二层卷积和池化
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=kernel_size)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 第三层卷积和池化
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=kernel_size)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, batch_first=True)

        # Attention层
        self.attention = Attention(hidden_dim)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 第一层卷积和池化
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二层卷积和池化
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # 第三层卷积和池化
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        # 调整维度顺序以适应LSTM输入 (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)  # 交换维度: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)

        # LSTM层
        lstm_out, (h_n, _) = self.lstm(x)

        # Attention层
        attention_out, attention_scores = self.attention(lstm_out)

        # 全连接层输出
        output = self.fc(attention_out)
        return output, attention_out, attention_scores

# 序列转化为one-hot编码
def seq_to_one_hot(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping[base] for base in seq if base in mapping])

# 加载数据的函数，并统一填充为最大长度
def load_data(file, label, max_len=6000):
    sequences = []
    labels = []

    # 读取序列文件并打上标签
    for record in SeqIO.parse(file, "fasta"):
        seq = seq_to_one_hot(str(record.seq))
        sequences.append(seq)
        labels.append(label)  # 抗性为1，非抗性为0

    # 如果未提供 max_len，找到所有序列的最大长度
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
        print(f"Max sequence length: {max_len}")  # 调试打印最大序列长度

    # 填充所有序列到相同长度
    padded_sequences = []
    for seq in sequences:
        # 如果序列长度小于 max_len，则填充
        if len(seq) < max_len:
            padded_seq = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant')
        else:
            padded_seq = seq[:max_len]  # 如果序列过长，截断到 max_len
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences), np.array(labels), max_len

# 初始化模型、损失函数和优化器
input_dim = 4  # one-hot编码的4个维度
hidden_dim = 128  # 减小隐藏层维度
output_dim = 1  # 二分类输出
kernel_size = 3
learning_rate = 0.0001  # 降低学习率
num_epochs = 200

model = CNNLSTMWithAttention(input_dim, hidden_dim, output_dim, kernel_size).to(device)
criterion = nn.BCEWithLogitsLoss()  # 二分类损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scaler = GradScaler()  # 用于混合精度训练

# 存储特征向量的列表
feature_vectors = []
all_labels = []  # 用于存储标签

# 加载抗性和非抗性数据，统一填充为相同的最大长度
resistant_data, resistant_labels, max_len_resistant = load_data("D:\desktop/2024\CNNLSTM/rseqs156.fna", 1)
non_resistant_data, non_resistant_labels, _ = load_data("D:\desktop/2024\CNNLSTM/nrseqs39.fna", 0, max_len=max_len_resistant)

# 将数据转换为 torch 张量并移动到指定设备
resistant_data = torch.tensor(resistant_data, dtype=torch.float32).permute(0, 2, 1).to(device)
non_resistant_data = torch.tensor(non_resistant_data, dtype=torch.float32).permute(0, 2, 1).to(device)
resistant_labels = torch.tensor(resistant_labels, dtype=torch.float32).unsqueeze(1).to(device)
non_resistant_labels = torch.tensor(non_resistant_labels, dtype=torch.float32).unsqueeze(1).to(device)

# 合并抗性和非抗性数据
data = torch.cat((resistant_data, non_resistant_data), dim=0)
labels = torch.cat((resistant_labels, non_resistant_labels), dim=0)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 使用混合精度训练
    with autocast():  # 自动混合精度上下文
        outputs, features, attention_scores = model(data)
        loss = criterion(outputs, labels)

    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    # 调整梯度以适应混合精度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 每20轮打印一次损失并清理GPU内存
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        feature_vectors.append(features.detach().cpu().numpy())  # 保存特征向量
        all_labels.append(labels.detach().cpu().numpy())  # 保存标签


    # 清理GPU内存
    torch.cuda.empty_cache()

# 将特征向量和标签保存为文件
feature_vectors = np.concatenate(feature_vectors, axis=0)  # 合并特征向量
all_labels = np.concatenate(all_labels, axis=0)  # 合并标签

# 保存为 .npy 文件
joblib.dump((feature_vectors, all_labels), 'gene_features_labels_with_attention1.npy')  # 保存特征向量和标签

# 保存为 CSV 文件
df = pd.DataFrame(feature_vectors)  # 创建 DataFrame
df['label'] = all_labels  # 添加标签列
df.to_csv('gene_features_labels_with_attention1.csv', index=False)  # 保存为 CSV 文件

print("CUDA available:", torch.cuda.is_available())
print("Current device:", device)
print("特征向量和标签已保存为 .npy 和 .csv 文件，可用于后续分析。")

# 训练结束后保存模型权重
torch.save(model.state_dict(), "model_cnnlstm_attention.pth")
print("模型权重保存完成！")
