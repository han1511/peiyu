import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def fix_paml_compatibility(input_fasta, output_phy):
    """
    彻底解决PAML报错：清洗隐形字符、替换非法字符、强制格式分隔
    """
    # 允许的核酸字符（含gap和简并N）
    ALLOWED_CHARS = {'A', 'T', 'C', 'G', '-', 'N'}
    # 匹配隐形/非法控制字符的正则
    INVISIBLE_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

    try:
        # 步骤1：清洗原始文件的隐形字符（避免编码/隐形字符干扰）
        with open(input_fasta, 'r', encoding='utf-8-sig') as f:
            raw_content = f.read()
        cleaned_content = INVISIBLE_PATTERN.sub('', raw_content)
        
        # 临时保存清洗后的FASTA（不修改原始文件）
        temp_fasta = "temp_cleaned.fasta"
        with open(temp_fasta, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        # 步骤2：读取并处理序列
        records = list(SeqIO.parse(temp_fasta, "fasta"))
        if not records:
            print("错误：未读取到有效序列！")
            return False

        processed_records = []
        print("=== 序列非法字符检查 ===")
        for idx, record in enumerate(records, 1):
            seq_id = record.id
            raw_seq = str(record.seq).upper()  # 统一转为大写

            # 检测并报告所有非法字符
            invalid_chars = set(raw_seq) - ALLOWED_CHARS
            if invalid_chars:
                print(f"\n序列 {idx}（ID: {seq_id}）存在非法字符：{invalid_chars}")
                # 定位每个非法字符的位置（方便手动核查原始文件）
                for char in invalid_chars:
                    positions = [i+1 for i, c in enumerate(raw_seq) if c == char]
                    print(f"  字符 '{char}' 出现在位置：{positions}")
                # 替换所有非法字符为N（确保无Z等字符）
                cleaned_seq = ''.join([c if c in ALLOWED_CHARS else 'N' for c in raw_seq])
                print("  已将非法字符替换为N")
            else:
                cleaned_seq = raw_seq
                print(f"\n序列 {idx}（ID: {seq_id}）无非法字符，验证通过")

            # 步骤3：处理序列ID（截断为10字符，去除空格，不足补空格）
            processed_id = re.sub(r'\s+', '', seq_id)[:10].ljust(10)

            processed_records.append(
                SeqRecord(Seq(cleaned_seq), id=processed_id, description="")
            )

        # 步骤4：验证序列长度一致（PAML要求对齐序列）
        seq_lengths = [len(rec) for rec in processed_records]
        if len(set(seq_lengths)) != 1:
            print("\n错误：序列长度不一致！请确保输入是对齐后的FASTA文件。")
            return False
        seq_len = seq_lengths[0]
        seq_count = len(processed_records)
        print(f"\n序列总数：{seq_count}，每条长度：{seq_len}，符合对齐要求")

        # 步骤5：写入严格符合PAML的PHYLIP文件
        with open(output_phy, 'w', newline='\n', encoding='utf-8') as f:
            # 头部：序列数 + 序列长度
            f.write(f"{seq_count} {seq_len}\n")
            line_len = 60  # 每行显示的序列长度
            # 第一部分：带ID的序列行（ID后强制2个空格）
            for rec in processed_records:
                f.write(f"{rec.id}  {str(rec.seq)[:line_len]}\n")
            # 第二部分：后续序列块（无ID，用10个空格占位对齐）
            for i in range(line_len, seq_len, line_len):
                for rec in processed_records:
                    seq_chunk = str(rec.seq)[i:i+line_len]
                    f.write(f"          {seq_chunk}\n")  # 10个空格匹配ID长度

        # 清理临时文件
        os.remove(temp_fasta)
        print(f"\n成功生成PAML兼容文件：{output_phy}")
        return True

    except Exception as e:
        print(f"\n处理过程中出错：{str(e)}")
        return False

# ========== 配置文件路径 ==========
input_fasta = r"I:\Bioinformation\select\paml-4.10.7\bin\51S.fas"
output_phy = r"I:\Bioinformation\select\paml-4.10.7\bin\51S_paml_fixed.phy"

if __name__ == "__main__":
    fix_paml_compatibility(input_fasta, output_phy)