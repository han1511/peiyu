from Bio import SeqIO

def extract_gene_info(file_path):
    try:
        with open('gene_info.txt', 'w', encoding='utf-8') as f:
            for record in SeqIO.parse(file_path, "genbank"):
                for feature in record.features:
                    if feature.type == "gene":
                        gene_name = feature.qualifiers.get("gene", ["N/A"])[0]
                        start = feature.location.start
                        end = feature.location.end
                        strand = feature.location.strand
                        locus_tag = feature.qualifiers.get("locus_tag", ["N/A"])[0]

                        info = f"基因名称: {gene_name}\n"
                        info += f"起始位置: {start}\n"
                        info += f"结束位置: {end}\n"
                        info += f"链方向: {strand}\n"
                        info += f"locus_tag: {locus_tag}\n"
                        info += "-" * 40 + "\n"

                        f.write(info)

        print("基因信息已成功保存到 gene_info.txt 文件中。")
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"错误: 发生未知错误 {e}。")

if __name__ == "__main__":
    file_path = "H:/研究生/硕士研究生/脉孢菌/串连序列/参考序列/sequence.gb"
    extract_gene_info(file_path)
    