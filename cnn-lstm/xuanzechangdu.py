from Bio import SeqIO

# 1. 从原始FASTA文件中选择符合条件的序列
def filter_sequences_by_length(input_fasta, output_fasta, min_len=5000, max_len=6000):
    filtered_records = []

    # 读取FASTA文件
    for record in SeqIO.parse(input_fasta, "fasta"):
        sequence_length = len(record.seq)

        # 判断序列长度是否在指定范围内
        if min_len <= sequence_length <= max_len:
            filtered_records.append(record)

    # 将符合条件的序列写入新的FASTA文件
    with open(output_fasta, "w") as output_file:
        SeqIO.write(filtered_records, output_file, "fasta")

    print(f"共有 {len(filtered_records)} 条序列被保存到 {output_fasta} 文件中。")

# 使用示例
input_fasta = "D:\desktop\CNNLSTM/rseqs1251.fna"  # 替换为你的FASTA文件路径
output_fasta = "D:\desktop\CNNLSTM/rseqs156.fna"  # 新文件名，可以根据需要修改

# 调用函数，筛选1000-6000bp之间的序列并保存
filter_sequences_by_length(input_fasta, output_fasta, min_len=5000, max_len=6000)
