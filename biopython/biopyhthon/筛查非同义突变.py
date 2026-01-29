from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable
import pandas as pd
from Bio.Seq import Seq
import os  # 用于路径处理
import traceback  # 用于详细错误信息

# ----------------------
# 1. 配置参数
# ----------------------
input_file = "C:\\Users\\韩培钰\\OneDrive\\Desktop\\18cor_extra.fasta"  # 输入文件路径
reference_id = "2017LX127"  # 参考序列ID（需与fasta中一致）
s_cds_start = 1  # S基因CDS起始位置
s_cds_end = 3420  # S基因CDS结束位置
cds_length = s_cds_end - s_cds_start + 1  # CDS长度（需为3的倍数）

# 结果保存路径（使用当前工作目录，避免权限问题）
# 如果你想保存到桌面，可以使用：
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# 然后将下面的os.getcwd()替换为desktop_path

# 直接使用文件名，会保存到当前脚本运行的目录
result_filename = "synonymous_mutations_results.csv"
count_filename = "synonymous_mutations_site_counts.csv"

# ----------------------
# 2. 读取并预处理序列
# ----------------------
def load_sequences(file):
    """读取FASTA文件，返回序列字典 {ID: 序列对象}"""
    try:
        records = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到序列文件：{file}，请检查路径是否正确")
    
    # 检查序列长度是否满足CDS要求
    for id, rec in records.items():
        if len(rec.seq) < cds_length:
            raise ValueError(f"序列{id}长度不足（当前{len(rec.seq)}bp，需至少{cds_length}bp），可能缺失S基因CDS区域")
    return records

# ----------------------
# 3. 序列比对
# ----------------------
def align_sequences(ref, query):
    """全局比对参考序列与待分析序列，返回去缺口的比对结果（Seq对象）"""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    
    try:
        alignments = aligner.align(ref, query)
        best_aln = alignments[0]
        # 去除缺口并转为Seq对象
        aligned_ref = Seq(str(best_aln[0]).replace("-", ""))
        aligned_query = Seq(str(best_aln[1]).replace("-", ""))
        return aligned_ref, aligned_query
    except Exception as e:
        raise RuntimeError(f"序列比对失败：{str(e)}")

# ----------------------
# 4. 检测同义突变
# ----------------------
def find_synonymous_mutations(ref_seq, query_seq, cds_start, cds_length):
    """筛选“碱基变但氨基酸不变”的同义突变"""
    codon_table = CodonTable.unambiguous_rna_by_name["Standard"]
    synonymous_mutations = []
    
    # 截取CDS区域（确保长度匹配）
    ref_cds = ref_seq[:cds_length]
    query_cds = query_seq[:cds_length]
    
    # 逐密码子分析
    for i in range(0, cds_length, 3):
        codon_end = i + 3
        if codon_end > cds_length:
            break  # 跳过不完整密码子
        
        # 提取密码子（Seq对象）
        ref_codon_dna = ref_cds[i:codon_end]
        query_codon_dna = query_cds[i:codon_end]
        
        # 跳过含缺口的密码子
        if "-" in str(ref_codon_dna) or "-" in str(query_codon_dna):
            continue
        
        # DNA转RNA（T→U）
        try:
            ref_codon = ref_codon_dna.transcribe()
            query_codon = query_codon_dna.transcribe()
        except Exception as e:
            print(f"密码子转录失败（位置{i}-{codon_end}）：{str(e)}")
            continue
        
        # 翻译为氨基酸
        try:
            ref_aa = codon_table.forward_table[str(ref_codon)]
            query_aa = codon_table.forward_table[str(query_codon)]
        except KeyError:
            continue  # 跳过终止密码子
        except Exception as e:
            print(f"密码子翻译失败：{str(e)}")
            continue
        
        # 判断同义突变
        if str(ref_codon_dna) != str(query_codon_dna) and ref_aa == query_aa:
            genome_positions = [cds_start + i + j for j in range(3)]
            # 记录每个碱基突变
            for j in range(3):
                if ref_codon_dna[j] != query_codon_dna[j]:
                    synonymous_mutations.append({
                        "genome_position": genome_positions[j],
                        "codon_position": (i // 3) + 1,
                        "ref_base": ref_codon_dna[j],
                        "query_base": query_codon_dna[j],
                        "ref_codon": str(ref_codon_dna),
                        "query_codon": str(query_codon_dna),
                        "amino_acid": ref_aa,
                    })
    return synonymous_mutations

# ----------------------
# 5. 主程序（含异常处理）
# ----------------------
if __name__ == "__main__":
    try:
        # 加载序列
        print(f"正在读取序列文件：{input_file}")
        sequences = load_sequences(input_file)
        ref_seq = sequences[reference_id].seq
        query_ids = [id for id in sequences if id != reference_id]
        print(f"成功加载{len(sequences)}条序列，参考序列：{reference_id}，待分析序列：{len(query_ids)}条")
        
        # 分析突变
        all_results = []
        for query_id in query_ids:
            print(f"正在分析序列: {query_id}")
            aligned_ref, aligned_query = align_sequences(ref_seq, sequences[query_id].seq)
            mutations = find_synonymous_mutations(
                aligned_ref, aligned_query, 
                cds_start=s_cds_start, 
                cds_length=cds_length
            )
            # 添加序列ID
            for mut in mutations:
                mut["query_id"] = query_id
                all_results.append(mut)
        
        # 保存结果
        if all_results:
            df = pd.DataFrame(all_results)
            df = df.sort_values(by=["genome_position", "query_id"])
            
            # 获取保存路径
            result_file = os.path.abspath(result_filename)
            count_file = os.path.abspath(count_filename)
            
            # 保存详细结果，增加异常处理
            try:
                df.to_csv(result_file, index=False)
                print(f"\n同义突变详细结果已保存至：{result_file}")
            except PermissionError:
                raise PermissionError(f"没有权限写入文件：{result_file}，请检查文件是否被占用或选择其他保存位置")
            except Exception as e:
                raise RuntimeError(f"保存结果文件失败：{str(e)}")
            
            # 统计相同位点出现次数
            site_counts = df.groupby("genome_position").size().reset_index(name="出现次数")
            site_counts = site_counts.sort_values(by="出现次数", ascending=False)
            
            try:
                site_counts.to_csv(count_file, index=False)
                print(f"相同突变位点统计结果已保存至：{count_file}")
            except PermissionError:
                raise PermissionError(f"没有权限写入文件：{count_file}，请检查文件是否被占用或选择其他保存位置")
            except Exception as e:
                raise RuntimeError(f"保存统计文件失败：{str(e)}")
            
            # 预览
            print("\n前5条突变结果预览：")
            print(df[["genome_position", "query_id", "ref_base", "query_base", "amino_acid"]].head())
            print("\n出现次数前5的位点预览：")
            print(site_counts.head())
        else:
            print("\n未检测到同义突变位点")
    
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")
        print("详细错误信息：")
        traceback.print_exc()  # 打印完整错误堆栈，方便排查问题
