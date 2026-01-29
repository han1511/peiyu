from Bio import SeqIO
import os

def extract_sequences_from_gb(gb_file, output_file=None, sequence_type='nucleotide'):
        """
        从GenBank格式文件中提取序列
        
        参数:
            gb_file (str): GenBank文件路径
            output_file (str, optional): 输出文件路径，若为None则不保存
            sequence_type (str): 提取的序列类型，'nucleotide'或'protein'
            
        返回:
            dict: 包含提取的序列信息，键为序列ID，值为序列
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(gb_file):
                raise FileNotFoundError(f"文件不存在: {gb_file}")
                
            # 读取GenBank文件
            records = SeqIO.parse(gb_file, "genbank")
            
            sequences = {}
            
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
            
            # 如果指定了输出文件，则保存序列
            if output_file:
                # 创建输出目录（如果不存在）
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                with open(output_file, 'w', encoding='utf-8') as f:
                    for seq_id, seq in sequences.items():
                        f.write(f">{seq_id}\n")
                        # 每80个字符换行，符合FASTA格式习惯
                        for i in range(0, len(seq), 80):
                            f.write(seq[i:i+80] + "\n")
                print(f"序列已保存到: {output_file}")
                
            return sequences
            
        except Exception as e:
            print(f"提取序列时发生错误: {str(e)}")
            return None

if __name__ == "__main__":
        # 设置你的文件路径（请替换为实际文件名）
        # 注意：路径前的r表示原始字符串，确保中文路径正常工作
        input_file = r"H:\研究生\硕士研究生\脉孢菌\串连序列\参考序列\sequence.gb"
        
        # 设置输出文件路径（可根据需要修改）
        output_file = r"H:\研究生\硕士研究生\脉孢菌\串连序列\参考序列\sequence.fasta"

        # 选择序列类型：'nucleotide'（核苷酸）或'protein'（蛋白质）
        seq_type = 'nucleotide'
        
        # 执行提取
        extract_sequences_from_gb(
            gb_file=input_file,
            output_file=output_file,
            sequence_type=seq_type
        )
    