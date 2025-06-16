import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 读取CSV文件
data = pd.read_csv("D:/pythonproject/cnn-lstm/gene_features_labels_with_attention1596.csv", header=None)

# 将特征和标签分离，假设最后一列为标签
features = data.iloc[:, :-1].values  # 提取特征列，转换为NumPy数组
labels = data.iloc[:, -1].values    # 提取标签列

# 2. 数据预处理
# 对特征进行标准化（均值为0，标准差为1）
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. 特征降维（可选，基于PCA）
# 如果特征维度较高，可以使用PCA减少维度（根据需要）
pca = PCA(n_components=50)  # 选择一个适当的维度
features_reduced = pca.fit_transform(features_scaled)

# 4. 定义并初始化HMM模型
# 降低模型复杂度，减少状态数
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)

# 5. 训练HMM模型
model.fit(features_reduced)

# 6. 预测抗性基因状态
predicted_states = model.predict(features_reduced)

# 7. 将预测结果保存到CSV文件
output_df = pd.DataFrame(predicted_states, columns=['Predicted_State'])
output_df.to_csv('predicted_gene_resistance1596.csv', index=False)

print("HMM模型已训练完成，预测结果已保存至 'predicted_gene_resistance1596.csv'")
