import Bio
print(Bio.__version__)
from Bio.Seq import Seq
my_seq = Seq("AGTACACTGGT")
print(my_seq)
from Bio import SeqIO
##读取其他文件中的核苷酸序列
fasta_path = "H:/研究生/博士/biopython/ls_orchid.fasta"

for seq_record in SeqIO.parse(fasta_path, "fasta"):
    print(f"ID: {seq_record.id}")
    print(f"Sequence: {seq_record.seq}")
    print(f"Description: {seq_record.description}")
