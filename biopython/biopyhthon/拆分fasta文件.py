import os
def split_fasta_dict(fasta_file_path):
    sequences_dict = {}
    with open(fasta_file_path, 'r') as file:
        current_id = ""
        current_seq = ""
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences_dict[current_id] = current_seq
                    current_seq = ""
                current_id = line[1:]  # 去掉开头的>作为字典的键
            else:
                current_seq += line
        if current_id and current_seq:
            sequences_dict[current_id] = current_seq
    return sequences_dict
# 使用示例，将这里的'your_fasta_file.fasta'替换成实际的FASTA文件路径
fasta_sequences_dict = split_fasta_dict('H:/研究生/郭玲文章/cov/韩培钰整理文件/17条序列结构基因用于建树的数据/17条阳性全序列/17条序列全正向(-17XY123).fas')
for seq_id, seq in fasta_sequences_dict.items():
    print(f"ID: {seq_id}\nSequence: {seq}\n")
output_folder = 'H:/研究生/郭玲文章/cov/韩培钰整理文件/17条序列结构基因用于建树的数据/17条阳性全序列'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for seq_id, seq in fasta_sequences_dict.items():
    output_file_path = os.path.join(output_folder, f"{seq_id}.txt")
    with open(output_file_path, 'w') as output_file:
        output_file.write(seq)

for seq_id, seq in fasta_sequences_dict.items():
    print(f"ID: {seq_id}\nSequence: {seq}\n")