import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Bio import SeqIO

# 定义CNN-LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # 调整维度顺序以适应LSTM输入
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.fc(x)
        return x

# 序列转化为one-hot编码
def seq_to_one_hot(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping[base] for base in seq if base in mapping])

# 加载数据的函数
def load_data(resistant_file, non_resistant_file):
    sequences = []
    labels = []

    # 读取抗性序列文件并打上标签1
    for record in SeqIO.parse(resistant_file, "fasta"):
        seq = seq_to_one_hot(str(record.seq))
        sequences.append(seq)
        labels.append(1)  # 抗性为1

    # 读取非抗性序列文件并打上标签0
    for record in SeqIO.parse(non_resistant_file, "fasta"):
        seq = seq_to_one_hot(str(record.seq))
        sequences.append(seq)
        labels.append(0)  # 非抗性为0

    # 将序列填充为相同长度
    max_len = max([len(seq) for seq in sequences])
    sequences = [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant') for seq in sequences]

    return np.array(sequences), np.array(labels)

# 初始化模型、损失函数和优化器
input_dim = 4  # one-hot编码的4个维度
hidden_dim = 64
output_dim = 1  # 二分类输出
kernel_size = 3
learning_rate = 0.001
num_epochs = 200

model = CNNLSTM(input_dim, hidden_dim, output_dim, kernel_size)
criterion = nn.BCEWithLogitsLoss()  # 二分类损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 加载数据（替换成你自己的路径）
    data, labels = load_data('D:/desktop/CNNLSTM/R40.fna', 'D:/desktop/CNNLSTM/NR10.fna')

    # 将numpy数组转换为torch张量
    data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)  # 需要调整维度 (batch_size, channels, seq_len)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # 前向传播
    outputs = model(data)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        predictions = torch.sigmoid(outputs).round()
        print(f'Predictions: {predictions.detach().cpu().numpy().flatten()}')
        print(f'Actual Labels: {labels.detach().cpu().numpy().flatten()}')

# 评估模型（可以在这里加载测试数据进行评估）

