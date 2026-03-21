#4.2  创建序列注释
from Bio.Seq import Seq
simple_seq = Seq("GATC")
from Bio.SeqRecord import SeqRecord
simple_seq_r = SeqRecord(simple_seq)
print(simple_seq_r)
simple_seq_r.id = "AC12345"
simple_seq_r.description = "Made up sequence I wish I could write a paper about"
print(simple_seq_r.description)
from Bio.Seq import Seq
simple_seq = Seq("GATC")
from Bio.SeqRecord import SeqRecord
simple_seq_r = SeqRecord(simple_seq, id="AC12345")
print(simple_seq_r)
simple_seq_r.annotations["evidence"] = "None. I just made it up."
print(simple_seq_r.annotations)
print(simple_seq_r.annotations["evidence"])
#从fasta中读取文件进行注释
from Bio import SeqIO
record = SeqIO.read("H:/biopython/NC_005816.fna", "fasta")
print(record)
print(record.id)
print(record.description)
#genbank文件中提取文件
from Bio import SeqIO
record = SeqIO.read("H:/biopython/NC 005816.gb", "genbank")
print(record)
print(record.description)
#Feature, location and position objects
from Bio import SeqFeature
start_pos = SeqFeature.AfterPosition(5)
end_pos = SeqFeature.BetweenPosition(9, left=8, right=9)
my_location = SeqFeature.FeatureLocation(start_pos, end_pos)
print(my_location)
print(my_location.start)
from Bio import SeqIO
my_snp = 4350
record = SeqIO.read("H:/biopython/NC 005816.gb", "genbank")
for feature in record.features:
    if my_snp in feature:
        print("%s %s" % (feature.type, feature.qualifiers.get("db_xref")))
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
seq = Seq("ACCGAGACGGCAAAGGCTAGCATAGGTATGAGACTTCCTTCCTGCCAGTGCTGAGGAACTGGGAGCCTAC")
feature =SeqFeature(FeatureLocation(5, 18), type="gene")
print(feature)
feature_seq = seq[feature.location.start:feature.location.end].reverse_complement()
print(feature_seq)
feature_seq = feature.extract(seq)
print(feature_seq)
#4.4对比
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
record1 = SeqRecord(Seq("ACGT"), id="test")
record2 = SeqRecord(Seq("ACGT"), id="test")
print(record1)
print(record2)
#4.6格式方法
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
record = SeqRecord(
Seq("MMYQQGCFAGGTVLRLAKDLAENNRGARVLVVCSEITAVTFRGPSETHLDSMVGQALFGD"
"GAGAVIVGSDPDLSVERPLYELVWTGATLLPDSEGAIDGHLREVGLTFHLLKDVPGLISK"
"NIEKSLKEAFTPLGISDWNSTFWIAHPGGPAILDQVEAKLGLKEEKMRATREVLSEYGNM"
"SSAC"),id="gi|14150838|gb|AAK54648.1|AF376133_1",description="chalcone synthase [Cucumis sativus]" )
print(record)
print(record.format("fasta"))
#4.7对序列进行切片
from Bio import SeqIO
record = SeqIO.read("H:/biopython/NC 005816.gb", "genbank")
print(record)
print(len(record))
print(record.features[21])  
#4.8Reverse-complementing SeqRecord objects
from Bio import SeqIO
record = SeqIO.read("H:/biopython/NC 005816.gb", "genbank")
print("%s %i %i %i %i" % (record.id, len(record), len(record.features),
len(record.dbxrefs), len(record.annotations)))
rc = record.reverse_complement(id="TESTING")
print("%s %i %i %i %i" % (rc.id, len(rc), len(rc.features), len(rc.dbxrefs),
len(rc.annotations)))



