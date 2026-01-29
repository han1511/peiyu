from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os

# 输入和输出文件路径（请改成实际 fasta 文件名，比如 sequences.fasta）
input_file = r"C:\Users\韩培钰\OneDrive\Desktop\XZYL\S.fas"
output_file = r"C:\Users\韩培钰\OneDrive\Desktop\XZYL\S.fas_no_stop.fasta"

# 终止密码子集合
stop_codons = {"TAA", "TAG", "TGA"}

new_records = []

for record in SeqIO.parse(input_file, "fasta"):
    seq = str(record.seq).upper()
    seq_len = len(seq)

    # 检查并去掉末端终止密码子
    if seq_len % 3 == 0:
        last_codon = seq[-3:]
        if last_codon in stop_codons:
            seq = seq[:-3]

    # 检查序列中间是否存在终止密码子
    for i in range(0, len(seq) - 3, 3):  # 遍历所有完整密码子
        codon = seq[i:i+3]
        if codon in stop_codons:
            print(f"序列 {record.id} 在第 {i//3 + 1} 个密码子位置发现终止密码子 {codon}")

    # 新的序列对象
    new_record = SeqRecord(Seq(seq), id=record.id, description="no_stop_codon")
    new_records.append(new_record)

# 保存新的序列文件
SeqIO.write(new_records, output_file, "fasta")
print(f"处理完成，新序列已保存到 {output_file}")
