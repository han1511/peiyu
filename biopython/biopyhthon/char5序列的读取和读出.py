from Bio import SeqIO

for seq_record in SeqIO.parse("H:/biopython/ls_orchid.gbk", "genbank"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))
identifiers = [seq_record.id for seq_record in SeqIO.parse("H:/biopython/ls_orchid.gbk", "genbank")]
print(identifiers )