from Bio import SeqIO
from Bio.Seq import Seq

# 输入文件路径
fasta_file = r"H:\研究生\郭玲文章\韩培钰整理文件\上传ncbi\部分rdrp\82条序列核苷酸.fasta"
output_file = r"H:\研究生\郭玲文章\韩培钰整理文件\上传ncbi\部分rdrp\82条序列核苷酸_3的倍数-clean.fasta"
output_amino_file = r"H:\研究生\郭玲文章\韩培钰整理文件\上传ncbi\部分rdrp\82条序列氨基酸.fasta"

stop_codons = {"TAA", "TAG", "TGA"}

clean_records = []

for record in SeqIO.parse(fasta_file, "fasta"):
    seq = str(record.seq).upper()
    seq_len = len(seq)

    # 去掉末端终止密码子
    if seq_len >= 3 and seq[-3:] in stop_codons:
        print(f"{record.id}: 去掉末端终止密码子 {seq[-3:]}")
        seq = seq[:-3]

    # 去掉序列内部的终止密码子
    new_seq = ""
    removed_positions = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if codon in stop_codons:
            removed_positions.append(f"{i+1}-{i+3}:{codon}")
            continue
        new_seq += codon

    if removed_positions:
        print(f"{record.id}: 内部去掉 {len(removed_positions)} 个终止密码子 -> {', '.join(removed_positions)}")

    # 再次检查长度是否为 3 的倍数
    if len(new_seq) % 3 != 0:
        print(f"{record.id}: 处理后长度 {len(new_seq)} 不是3的倍数 ❌")
    else:
        print(f"{record.id}: 合格 ✅")

    # 更新序列对象
    record.seq = record.seq.__class__(new_seq)
    clean_records.append(record)

# 翻译核苷酸序列为氨基酸序列
amino_records = []
for record in clean_records:
    # 翻译序列（使用默认的密码子表）
    amino_seq = record.seq.translate(table="Standard", to_stop=False)
    # 创建新的氨基酸序列记录
    amino_record = record.__class__(amino_seq)
    amino_record.id = record.id
    amino_record.name = record.name
    amino_record.description = record.description + " [Translated to amino acid]"
    amino_records.append(amino_record)

# 保存结果
SeqIO.write(clean_records, output_file, "fasta")
SeqIO.write(amino_records, output_amino_file, "fasta")
print(f"\n处理完成！")
print(f"清理后的核苷酸序列已保存到: {output_file}")
print(f"翻译后的氨基酸序列已保存到: {output_amino_file}")
