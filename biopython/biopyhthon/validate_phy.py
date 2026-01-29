
import os
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def process_and_convert_fasta(input_path, output_path):
    """
    处理FASTA文件（假设为待比对序列，此处进行简单格式处理）并转换为PHYLIP格式
    
    参数:
        input_path: 输入FASTA文件的绝对路径
        output_path: 输出PHYLIP文件的绝对路径
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 读取FASTA文件（假设为未比对序列，此处模拟简单比对处理）
        # 注意：Biopython主要用于处理已有比对，若需实际比对需使用ClustalW等工具
        records = list(AlignIO.read(input_path, "fasta"))
        print(f"成功读取文件: {input_path}")
        print(f"包含 {len(records)} 条序列")
        
        # 检查序列是否已对齐（长度是否一致）
        seq_lengths = [len(rec) for rec in records]
        if len(set(seq_lengths)) != 1:
            print("警告: 序列长度不一致，将视为未对齐序列进行简单处理（填充gap）")
            # 简单处理：用'-'填充至最长序列长度（实际比对需用专业工具）
            max_len = max(seq_lengths)
            aligned_records = []
            for rec in records:
                padded_seq = str(rec.seq).ljust(max_len, '-')
                aligned_records.append(
                    SeqRecord(Seq(padded_seq), id=rec.id, description="")
                )
            alignment = MultipleSeqAlignment(aligned_records)
        else:
            # 已对齐序列直接使用
            alignment = MultipleSeqAlignment(records)
        
        # 处理序列ID以符合PHYLIP规范（PAML要求）
        processed_records = []
        for rec in alignment:
            # 截断ID至10个字符，移除空白，补充空格
            clean_id = rec.id.split()[0][:10].ljust(10)
            # 确保序列为大写
            upper_seq = str(rec.seq).upper()
            processed_records.append(
                SeqRecord(Seq(upper_seq), id=clean_id, description="")
            )
        processed_alignment = MultipleSeqAlignment(processed_records)
        
        # 保存为PHYLIP格式（interleaved，兼容PAML）
        AlignIO.write(processed_alignment, output_path, "phylip")
        print(f"成功保存为PHYLIP格式: {output_path}")
        print(f"序列数量: {len(processed_alignment)}, 序列长度: {processed_alignment.get_alignment_length()}")
        
        return True
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 你的文件路径（绝对路径）
    input_file = r"I:\Bioinformation\select\paml-4.10.7\bin\51S.fas"
    output_file = r"I:\Bioinformation\select\paml-4.10.7\bin\51S_processed.phy"
    
    # 执行处理和转换
    process_and_convert_fasta(input_file, output_file)
