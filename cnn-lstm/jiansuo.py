# 定义文件路径
file_path = "G:\大模型数据\基因组\杨属Populus\小叶杨 Populus simonii.fna"

# 初始化计数器
count = 0

# 读取文件并统计 '>' 的个数
with open(file_path, 'r') as file:
    for line in file:
        # 如果行以 '>' 开头，则计数器加一
        if line.startswith('>'):
            count += 1

# 输出结果
print(f"'>' 符号的个数: {count}")
