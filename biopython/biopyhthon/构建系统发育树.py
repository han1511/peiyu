from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import Bio.Phylo as Phylo
import os
import re

# 1. 定义路径（文件名无空格，格式规范）
alignment_path = r"E:\empree\chen\ORF\L-ORF_mafft.fas"  # 输入比对文件
cleaned_aln_path = r"E:\empree\chen\ORF\L-ORF_mafft_cleaned.fas"  # 清洗后的比对文件
output_tree_path = r"E:\empree\chen\ORF\L-ORF_NJ_tree.nwk"  # 输出树文件（无空格）

# 2. 清洗序列（确保无无效字符，避免影响树构建）
def clean_seq(seq_str):
    seq_str = seq_str.upper()  # 小写转大写
    # 核苷酸序列：保留A/T/C/G/-，其他替换为N（根据序列类型调整）
    return re.sub(r'[^ATCG-]', 'N', seq_str)

try:
    # 读取并清洗比对序列
    records = [r for r in SeqIO.parse(alignment_path, "fasta")]
    cleaned_records = [SeqRecord(Seq(clean_seq(str(r.seq))), id=r.id, description="") for r in records]
    SeqIO.write(cleaned_records, cleaned_aln_path, "fasta")
    print(f"✅ 序列清洗完成：{cleaned_aln_path}")
except Exception as e:
    print(f"❌ 序列清洗失败：{str(e)}")
    exit()

# 3. 构建NJ树（确保分支长度完整）
try:
    # 读取清洗后的比对文件
    aln = AlignIO.read(cleaned_aln_path, "fasta")
    # 核苷酸用blastn模型（氨基酸用blosum62），确保距离矩阵计算正常
    calculator = DistanceCalculator("blastn")
    dm = calculator.get_distance(aln)
    # 构建NJ树，强制保留分支长度
    constructor = DistanceTreeConstructor(calculator, method="nj")
    nj_tree = constructor.build_tree(aln)
    # 关键：给根节点添加分支长度（避免格式缺失）
    if nj_tree.root.branch_length is None:
        nj_tree.root.branch_length = 0.1
    print(f"✅ NJ树构建完成，含 {len(nj_tree.get_terminals())} 个物种")
except Exception as e:
    print(f"❌ 树构建失败：{str(e)}")
    exit()

# 4. 保存树文件（用Phylo.write，确保格式规范）
try:
    Phylo.write(nj_tree, output_tree_path, "newick")
    # 额外验证：读取刚保存的文件，确认无格式问题
    test_tree = Phylo.read(output_tree_path, "newick")
    print(f"✅ 树文件保存成功且可正常读取：{output_tree_path}")
except Exception as e:
    print(f"❌ 树文件保存/验证失败：{str(e)}")
    exit()
from Bio import Phylo
tree_path = r"E:\empree\chen\ORF\L-ORF_NJ_tree.nwk"
try:
    tree = Phylo.read(tree_path, "newick")
    print(f"✅ 成功读取树文件，含 {len(tree.get_terminals())} 个物种")
except Exception as e:
    print(f"❌ 读取失败：{str(e)}")
