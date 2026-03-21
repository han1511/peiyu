from Bio.Seq import Seq
my_seq = Seq("GATCG")
for index, letter in enumerate(my_seq):
    print("%i %s" % (index, letter))
print(len(my_seq))
print(my_seq[-1]) #last letter

#计算某个碱基的的频率
from Bio.Seq import Seq
my_seq = Seq("GATCGATGGGCCTATATAGGATCGAAAATCGC")
len(my_seq)
my_seq.count("G")
print(100 * float(my_seq.count("G") + my_seq.count("C")) / len(my_seq))#计算CG含量
##另外一种方式
from Bio.SeqUtils import gc_fraction
my_seq = Seq("GATCGATGGGCCTATATAGGATCGAAAATCGC")
gc_content = gc_fraction(my_seq) * 100 
print(f"GC Content: {gc_content:.2f}%")
#3.2对序列进行切片
my_seq = Seq("GATCGATGGGCCTATATAGGATCGAAAATCGC")
print(my_seq[4:12])
##将序列做成字符串
print(str(my_seq))
fasta_format_string = ">Name\n%s\n" % my_seq
print(fasta_format_string)
##3.4添加序列
from Bio.Seq import Seq
protein_seq = Seq("EVRNAK")
dna_seq = Seq("ACGT")
print(protein_seq + dna_seq)

list_of_seqs = [Seq("ACGT"), Seq("AACC"), Seq("GGTT")]
concatenated = Seq("")
for s in list_of_seqs:
    concatenated += s
print(concatenated)
contigs = [Seq("ATG"), Seq("ATCCCG"), Seq("TTGCA")]
spacer = Seq("N"*10)
print(spacer.join(contigs))
#3.5变更案例
dna_seq = Seq("acgtACGT")
print(dna_seq)
print(dna_seq.upper())
print(dna_seq.lower())
##3.6核苷酸序列的反向和反向互补
from Bio.Seq import Seq
my_seq = Seq("GATCGATGGGCCTATATAGGATCGAAAATCGC")
print(my_seq.complement())
print(my_seq.reverse_complement())
print(my_seq[::-1])#反向序列
#3.7序列的转录
from Bio.Seq import Seq
coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
print(coding_dna)
template_dna = coding_dna.reverse_complement()
print(template_dna)
messenger_rna = coding_dna.transcribe()
print(messenger_rna)

print(template_dna.reverse_complement().transcribe())
messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
print(messenger_rna)
print(messenger_rna.back_transcribe())
#3.8翻译
from Bio.Seq import Seq
coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
coding_dna.translate(table="Vertebrate Mitochondrial")
print(coding_dna.translate(table=2))
print(coding_dna.translate(to_stop=True))
from Bio.Seq import Seq
gene = Seq("GTGAAAAAGATGCAATCTATCGTACTCGCACTTTCCCTGGTTCTGGTCGCTCCCATGGCA"
"GCACAGGCTGCGGAAATTACGTTAGTCCCGTCAGTAAAATTACAGATAGGCGATCGTGAT"
"AATCGTGGCTATTACTGGGATGGAGGTCACTGGCGCGACCACGGCTGGTGGAAACAACAT"
"TATGAATGGCGAGGCAATCGCTGGCACCTACACGGACCGCCGCCACCGCCGCGCCACCAT"
"AAGAAAGCTCCTCATGATCATCACGGCGGTCATGGTCCAGGCAAACATCACCGCTAA")

print(gene.translate(table="Bacterial",to_stop=True))
##3.9密码表
from Bio.Data import CodonTable
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
print(standard_table)
mito_table = CodonTable.unambiguous_dna_by_name["Vertebrate Mitochondrial"]
print(mito_table)

standard_table = CodonTable.unambiguous_dna_by_id[1]
mito_table = CodonTable.unambiguous_dna_by_id[2]
print(standard_table)
print(mito_table)
##对比序列文件
from Bio.Seq import Seq
seq1 = Seq("ACGT")
print(seq1)
print("ACGT" == seq1)
