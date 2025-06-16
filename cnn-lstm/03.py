import random
from Bio import SeqIO


def extract_sequences_without_n(fasta_file, output_file):
    # 读取FASTA文件中的所有序列
    sequences_without_n = []

    # 读取每条序列并检查是否包含'N'
    for record in SeqIO.parse(fasta_file, "fasta"):
        if 'N' not in record.seq:
            sequences_without_n.append(record)

    # 将不包含'N'的序列写入新的FASTA文件
    with open(output_file, "w") as output_handle:
        SeqIO.write(sequences_without_n, output_handle, "fasta")

    print(f"已提取不包含'N'的序列，并保存到 {output_file}")
    return sequences_without_n


def select_random_sequences(sequences, num_sequences, output_file):
    # 随机选择指定数量的序列
    selected_sequences = random.sample(sequences, num_sequences)

    # 将选择的序列写入新的FASTA文件
    with open(output_file, "w") as output_handle:
        SeqIO.write(selected_sequences, output_handle, "fasta")

    print(f"已随机选择 {num_sequences} 条序列，并保存到 {output_file}")


# 示例FASTA文件路径
fasta_file = "D:\desktop\CNNLSTM/nrseqs345.fna"  # 替换为你的neseqs文件路径

# 提取不包含'N'的序列
extracted_sequences = extract_sequences_without_n(fasta_file, "D:\desktop\CNNLSTM/nrseqs231.fna")

# 随机选择 39 条序列
select_random_sequences(extracted_sequences, 39, "D:\desktop\CNNLSTM/nrseqs39.fna")
