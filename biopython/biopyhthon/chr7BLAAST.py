from Bio.Blast import NCBIWWW
from Bio import SeqIO
#help (NCBIWWW.qblast)
result_handle = NCBIWWW.qblast("blastn", "nt", "8332116")

fasta_string = open(r"H:/biopython/229E.fasta").read()#使用冠状病毒229E
#recoed = SeqID.read(r"H:/biopython/229E.fasta",format="fasta")#fasta格式可以使用SeqIO.read读取  
print(fasta_string )
result_handle = NCBIWWW.qblast("blastn", "nt", fasta_string )
print(result_handle)
##########################################
#本地运行对比
