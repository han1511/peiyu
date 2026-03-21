from Bio import AlignIO
from Bio.SeqUtils import seq1
from itertools import combinations
import pandas as pd

def calculate_similarity(seq_a, seq_b):
    """计算两个等长序列的相似性比例"""
    if len(seq_a) != len(seq_b):
        raise ValueError("序列长度不一致，无法计算相似性")
    match_count = sum(a == b for a, b in zip(seq_a, seq_b))
    return match_count / len(seq_a)

def process_alignment_file(file_path, format='fasta'):
    """读取多序列比对文件并计算所有序列对的相似性"""
    try:
        # 读取比对文件
        alignment = AlignIO.read(file_path, format)
        print(f"成功读取 {len(alignment)} 条序列")
        
        # 存储结果的列表
        similarity_results = []
        
        # 遍历所有序列对并计算相似性
        for i, j in combinations(range(len(alignment)), 2):
            # 获取序列ID
            id1 = alignment[i].id
            id2 = alignment[j].id
            
            # 获取序列并转换为单字母代码
            seq_a = seq1(str(alignment[i].seq))
            seq_b = seq1(str(alignment[j].seq))
            
            # 计算相似性
            similarity = calculate_similarity(seq_a, seq_b)
            
            # 存储结果
            similarity_results.append({
                'Sequence1': id1,
                'Sequence2': id2,
                'Similarity': similarity
            })
        
        # 转换为DataFrame便于后续分析
        results_df = pd.DataFrame(similarity_results)
        return results_df
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None

if __name__ == "__main__":
    # 文件路径
    file_path = r"H:/研究生/郭玲文章/韩培钰整理文件/440bp序列/84条序列Rdrp建树/rdrp-1_split.fasta"
    
    # 处理文件并计算相似性
    results = process_alignment_file(file_path)
    
    if results is not None:
        # 打印结果摘要
        print(f"计算完成，共生成 {len(results)} 对序列相似性数据")
        
        # 保存结果到CSV文件
        output_file = "sequence_similarity.csv"
        results.to_csv(output_file, index=False)
        print(f"结果已保存到 {output_file}")
        
        # 打印相似性最高的10对序列
        print("\n相似性最高的10对序列：")
        print(results.sort_values('Similarity', ascending=False).head(10))