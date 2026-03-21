from Bio import SeqIO
from Bio.Seq import Seq
import argparse

def find_orfs(seq, min_length=150):
    """
    在三条正向阅读框中查找 ORFs
    返回 ORF 的起始、终止和翻译的蛋白序列
    """
    results = []
    seq_len = len(seq)

    for frame in range(3):  # 三个正向阅读框
        trans = seq[frame:].translate(to_stop=False)
        aa_start = 0

        for i, aa in enumerate(trans):
            if aa == "*":  # stop codon
                if i*3 + frame - aa_start*3 >= min_length:
                    orf_seq = seq[frame+aa_start*3:frame+i*3]
                    results.append((frame+aa_start*3, frame+i*3, str(orf_seq.translate())))
                aa_start = i + 1

    return results


def check_fasta(fasta_file, min_orf_length=150, min_seq_length=200, max_n_ratio=0.1):
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = record.seq
        seq_length = len(seq)
        print(f"\n>> Checking sequence: {record.id}, length={seq_length}")
        
        # 基于长度阈值筛选
        if seq_length < min_seq_length:
            print(f"   ⚠️ Low quality: sequence length ({seq_length}) below threshold ({min_seq_length})")
            continue
        
        # 基于N含量比例筛选
        n_count = seq.count('N') + seq.count('n')
        n_ratio = n_count / seq_length if seq_length > 0 else 0
        if n_ratio > max_n_ratio:
            print(f"   ⚠️ Low quality: N content ratio ({n_ratio:.2%}) exceeds threshold ({max_n_ratio:.2%})")
            continue

        orfs = find_orfs(seq, min_length=min_orf_length)
        if not orfs:
            print("   No valid ORF found (sequence may be non-coding or poor quality).")
        else:
            for start, end, protein in orfs:
                has_internal_stop = "*" in protein[:-1]  # 内部是否有 stop codon
                print(f"   ORF {start}-{end}, length={end-start} bp, protein length={len(protein)} aa")
                if has_internal_stop:
                    print("   ⚠️ Contains internal stop codon")
                else:
                    print("   ✅ No internal stop codon")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Check sequences for quality (length, N content) and ORFs")
    
    # 添加输入文件参数
    parser.add_argument("input_file", help="Input FASTA file path")
    
    # 添加可选参数
    parser.add_argument("--min-orf-length", type=int, default=150, help="Minimum ORF length (default: 150)")
    parser.add_argument("--min-seq-length", type=int, default=200, help="Minimum sequence length (default: 200)")
    parser.add_argument("--max-n-ratio", type=float, default=0.1, help="Maximum N content ratio (default: 0.1)")
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用函数进行处理
    check_fasta(
        args.input_file,
        min_orf_length=args.min_orf_length,
        min_seq_length=args.min_seq_length,
        max_n_ratio=args.max_n_ratio
    )
