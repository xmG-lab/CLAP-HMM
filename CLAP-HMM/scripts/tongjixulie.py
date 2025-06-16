import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt


# 1. 计算每条序列的长度
def calculate_sequence_lengths(fasta_file):
    sequence_lengths = {}

    # 读取FASTA文件
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_id = record.id  # 获取序列ID
        sequence_length = len(record.seq)  # 获取序列的长度
        sequence_lengths[sequence_id] = sequence_length  # 存储序列ID及其长度

    return sequence_lengths


# 2. 统计每个区间的序列数目
def calculate_length_distribution(sequence_lengths, bin_size=1000):
    lengths = list(sequence_lengths.values())

    # 计算最大值和最小值
    min_len = np.min(lengths)
    max_len = np.max(lengths)

    # 计算区间范围：从0开始，每个区间长度为bin_size
    bins = np.arange(0, max_len + bin_size, bin_size)

    # 统计每个区间的序列数目
    bin_counts = {}
    for length in lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                bin_range = f"{bins[i]}-{bins[i + 1]}bp"
                if bin_range not in bin_counts:
                    bin_counts[bin_range] = 0
                bin_counts[bin_range] += 1
                break

    return bin_counts, bins


# 3. 输出各区间的序列数目（按区间大小排序）
def print_length_distribution(bin_counts):
    # 按照区间的起始数字进行排序
    sorted_bins = sorted(bin_counts.items(), key=lambda x: int(x[0].split('-')[0]))

    print("序列长度区间及其对应的序列数目:")
    for bin_range, count in sorted_bins:
        print(f"{bin_range}: {count}条序列")

# 4. 绘制序列长度的直方图
def plot_length_distribution(sequence_lengths, bins):
    lengths = list(sequence_lengths.values())

    # 绘制直方图
    plt.hist(lengths, bins=bins, edgecolor='black')  # 使用计算得到的bins
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Frequency')
    plt.show()


# 使用示例
fasta_file = "D:\desktop\CNNLSTM/rseqs156.fna"  # 替换为你的FASTA文件路径

# 1. 计算序列长度
sequence_lengths = calculate_sequence_lengths(fasta_file)

# 2. 计算区间及每个区间的序列数目
bin_counts, bins = calculate_length_distribution(sequence_lengths, bin_size=1000)

# 3. 输出每个区间的序列数目
print_length_distribution(bin_counts)

# 4. 绘制直方图
plot_length_distribution(sequence_lengths, bins)
