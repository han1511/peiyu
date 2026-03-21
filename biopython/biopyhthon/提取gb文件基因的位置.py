from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation
import os

def extract_sequences_and_genes(gb_file, target_genes=None, output_file=None, sequence_type='nucleotide'):
    """
    从GenBank格式文件中提取序列及特定基因的位置信息
    
    参数:
        gb_file (str): GenBank文件路径
        target_genes (list): 要查找的目标基因名称列表
        output_file (str, optional): 输出文件路径，若为None则不保存
        sequence_type (str): 提取的序列类型，'nucleotide'或'protein'
        
    返回:
        tuple: (sequences, gene_locations)
              sequences为提取的序列字典，gene_locations为基因位置字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(gb_file):
            raise FileNotFoundError(f"文件不存在: {gb_file}")
            
        # 读取GenBank文件
        records = SeqIO.parse(gb_file, "genbank")
        
        sequences = {}
        gene_locations = {}  # 存储基因位置信息
        
        for record in records:
            # 提取核苷酸序列
            if sequence_type.lower() == 'nucleotide':
                seq = str(record.seq)
                sequences[record.id] = seq
                print(f"提取到核苷酸序列: {record.id}, 长度: {len(seq)}")
            # 提取蛋白质序列
            elif sequence_type.lower() == 'protein':
                for feature in record.features:
                    if feature.type == 'CDS':
                        if 'translation' in feature.qualifiers:
                            protein_id = feature.qualifiers.get('protein_id', ['unknown'])[0]
                            protein_seq = feature.qualifiers['translation'][0]
                            sequences[protein_id] = protein_seq
                            print(f"提取到蛋白质序列: {protein_id}, 长度: {len(protein_seq)}")
            
            # 查找目标基因的位置
            if target_genes:
                for feature in record.features:
                    # 查找基因特征
                    if feature.type == 'gene':
                        # 获取基因名称
                        gene_name = feature.qualifiers.get('gene', ['unknown'])[0]
                        # 检查是否为目标基因
                        if gene_name in target_genes:
                            # 解析位置信息
                            location = feature.location
                            # 处理复合位置（如join的情况）
                            if isinstance(location, CompoundLocation):
                                positions = []
                                for part in location.parts:
                                    positions.append(f"{part.start+1}..{part.end}")  # +1因为GenBank是1-based索引
                                location_str = f"join({','.join(positions)})"
                            else:
                                location_str = f"{location.start+1}..{location.end}"
                            
                            # 记录链方向（正向/反向互补）
                            strand = "正向" if location.strand == 1 else "反向互补"
                            
                            # 存储位置信息
                            gene_locations[gene_name] = {
                                "record_id": record.id,
                                "location": location_str,
                                "strand": strand,
                                "start": int(location.start) + 1,  # 转换为1-based
                                "end": int(location.end),
                                "length": int(location.end) - int(location.start)
                            }
                            print(f"找到基因 {gene_name}: {location_str} ({strand})")
        
        # 检查是否有目标基因未找到
        if target_genes:
            for gene in target_genes:
                if gene not in gene_locations:
                    print(f"警告: 未找到基因 {gene} 的位置信息")
        
        # 保存序列（如果指定了输出文件）
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                for seq_id, seq in sequences.items():
                    f.write(f">{seq_id}\n")
                    # 每80个字符换行
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
            print(f"序列已保存到: {output_file}")
            
        return sequences, gene_locations
        
    except Exception as e:
        print(f"处理时发生错误: {str(e)}")
        return None, None

if __name__ == "__main__":
    # 设置文件路径
    input_file = r"H:/研究生/硕士研究生/脉孢菌/串连序列/参考序列/sequence.gb"
    output_file = r"H:/研究生/硕士研究生/脉孢菌/串连序列/参考序列/sequence1.fasta"
    
    # 要查找的目标基因
    target_genes = ['TML', 'QMA', 'TMI', 'DMG']
    
    # 序列类型：'nucleotide'（核苷酸）或'protein'（蛋白质）
    seq_type = 'nucleotide'
    
    # 执行提取
    sequences, gene_locations = extract_sequences_and_genes(
        gb_file=input_file,
        target_genes=target_genes,
        output_file=output_file,
        sequence_type=seq_type
    )
    
    # 打印基因位置结果
    if gene_locations:
        print("\n基因位置汇总:")
        for gene, info in gene_locations.items():
            print(f"{gene}: {info['location']} ({info['strand']}), 长度: {info['length']} bp")
