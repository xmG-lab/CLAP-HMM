from Bio import SeqIO


def merge_fasta_files(file_list, output_file):
    with open(output_file, "w") as out_handle:
        for fasta_file in file_list:
            for record in SeqIO.parse(fasta_file, "fasta"):
                SeqIO.write(record, out_handle, "fasta")

    print(f"合并后的FASTA文件已保存为: {output_file}")


# 文件列表，包含需要合并的FASTA文件路径
fasta_files = [
    r"D:\desktop\lmsjj\Populus\r\alba\protein.faa",  # 第一个基因文件"D:\desktop\lmsjj\Populus\r\alba\protein.faa"
    r"D:\desktop\lmsjj\Populus\r\euphratica\protein.faa",  # 第二个基因文件
    r"D:\desktop\lmsjj\Populus\r\nigra\protein.faa",  # 第三个基因文件
    r"D:\desktop\lmsjj\Populus\r\trichocarpa\protein.faa"  # 第四个基因文件
]

# 输出文件
output_fasta = r"D:\desktop\CNNLSTM\allrproteins.fna"

# 调用函数合并文件
merge_fasta_files(fasta_files, output_fasta)
