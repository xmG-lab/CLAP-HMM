import joblib
from hmmlearn import hmm
import numpy as np
import pandas as pd

# 1. 使用 joblib 加载保存的 .npy 文件（实际是 joblib 格式）
data = joblib.load('gene_features_labels_with_attention.npy')

# 提取特征和标签
feature_vectors, all_labels = data

# 2. 定义并初始化HMM模型
# 假设抗性基因和非抗性基因分别对应两个状态
# n_components表示HMM模型的状态数量，这里假设2种状态（抗性和非抗性）
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)

# 3. 训练HMM模型
# 使用CNN-LSTM提取的特征来拟合HMM模型
model.fit(feature_vectors)

# 4. 预测抗性基因状态
# 使用HMM模型对特征序列进行预测，得到每个时间步的状态（抗性或非抗性）
predicted_states = model.predict(feature_vectors)

# 5. 获取预测的概率
# 预测每个序列属于某个状态的概率（抗性或非抗性）
predicted_probs = model.predict_proba(feature_vectors)

# 6. 保存预测结果和预测概率
output_df = pd.DataFrame(predicted_states, columns=['Predicted_State'])

# 如果每个状态的概率是两列（状态0和状态1的概率），我们将其分开并保存
prob_df = pd.DataFrame(predicted_probs, columns=['State_0_Prob', 'State_1_Prob'])

# 合并预测结果和预测概率
final_df = pd.concat([output_df, prob_df], axis=1)

# 保存结果为 CSV 文件
final_df.to_csv('predicted_gene_resistance_probabilities_with_joblib.csv', index=False)

print("HMM模型已训练完成，预测抗性基因概率已保存至 'predicted_gene_resistance_probabilities_with_joblib.csv'")
