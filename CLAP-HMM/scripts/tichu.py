import numpy as np

# 加载 .npy 文件，允许 pickle
data = np.load("D:/desktop/CNNLSTM/gene_features_labels1000.npy", allow_pickle=True)

# 查看数据内容
print(data)
