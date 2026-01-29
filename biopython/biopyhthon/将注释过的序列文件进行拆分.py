from Bio import SeqIO
import os


def split_gb_by_gene_name_and_save_fasta(gb_file_path, output_folder):
    """
    按照基因名称拆分GenBank文件中的序列，并将每个基因保存为单独的FASTA格式文件到指定文件夹。

    参数:
    gb_file_path (str): 输入的GenBank文件路径
    output_folder (str): 输出文件夹路径，用于保存拆分后的基因文件
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for record in SeqIO.parse(gb_file_path, "genbank"):
        for feature in record.features:
            if feature.type == "gene":
                gene_name = ""
                if "gene" in feature.qualifiers:
                    gene_name = feature.qualifiers["gene"][0]
                elif "locus_tag" in feature.qualifiers:
                    gene_name = feature.qualifiers["locus_tag"][0]
                else:
                    continue

                # 构建输出文件路径
                output_file_path = os.path.join(output_folder, gene_name + ".fasta")
                gene_seq = record.seq[feature.location.start:feature.location.end]
                with open(output_file_path, "w") as output_file:
                    output_file.write(">"+gene_name + "\n")
                    output_file.write(str(gene_seq) + "\n")



# 使用示例，将这里的路径替换成实际的路径
gb_file_path = "H:/研究生/郭玲文章/cov/韩培钰整理文件/Mouse coronavirus .gb"
output_folder = "H:/研究生/郭玲文章/cov/韩培钰整理文件"
split_gb_by_gene_name_and_save_fasta(gb_file_path, output_folder)