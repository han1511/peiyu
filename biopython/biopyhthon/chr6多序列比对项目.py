from Bio import pairwise2
from Bio import SeqIO
from itertools import combinations
import pandas as pd
import os
import multiprocessing
from functools import partial
from tqdm import tqdm  # 用于显示进度条

def calculate_similarity(seq_a, seq_b, sequence_type='dna'):
    """比对两个序列并计算相似性"""
    # 设置比对参数（简化版，可根据需要调整）
    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -1
    gap_extend_penalty = -0.5
    
    # 执行全局比对
    alignments = pairwise2.align.globalms(
        seq_a, seq_b, 
        match_score, mismatch_penalty, 
        gap_penalty, gap_extend_penalty,
        one_alignment_only=True  # 只保留最佳比对结果
    )
    
    # 若无比对结果，返回0
    if not alignments:
        return 0.0
    
    # 获取最佳比对
    aligned_a, aligned_b, score, begin, end = alignments[0]
    
    # 计算相似性（匹配位置数/比对总长度）
    match_count = sum(a == b for a, b in zip(aligned_a, aligned_b))
    return match_count / len(aligned_a)

def process_pair(args, sequences, sequence_type):
    """处理单个序列对的相似性计算"""
    i, j = args
    seq_a = str(sequences[i].seq)
    seq_b = str(sequences[j].seq)
    id1 = sequences[i].id
    id2 = sequences[j].id
    
    similarity = calculate_similarity(seq_a, seq_b, sequence_type)
    return id1, id2, similarity

def process_sequence_file(file_path, format='fasta', sequence_type='dna', n_jobs=-1):
    """读取序列文件并计算所有序列对的相似性（并行版），返回相似性矩阵 DataFrame"""
    try:
        # 读取序列文件
        sequences = list(SeqIO.parse(file_path, format))
        seq_count = len(sequences)
        print(f"成功读取 {seq_count} 条序列")
        
        # 生成所有序列对的索引
        pairs = list(combinations(range(seq_count), 2))
        total_pairs = len(pairs)
        print(f"共有 {total_pairs} 对序列需要比对")
        
        # 设置并行进程数
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        print(f"使用 {n_jobs} 个进程进行并行计算")
        
        # 创建部分应用函数
        process_func = partial(process_pair, sequences=sequences, sequence_type=sequence_type)
        
        # 并行处理
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_func, pairs),
                total=total_pairs,
                desc="计算相似性"
            ))
        
        # 构建相似性矩阵
        seq_ids = [seq.id for seq in sequences]
        sim_matrix = pd.DataFrame(index=seq_ids, columns=seq_ids)
        # 填充对角线为 1（自身相似性为 100%）
        for seq_id in seq_ids:
            sim_matrix.at[seq_id, seq_id] = 1.0
        # 填充非对角线结果
        for id1, id2, similarity in results:
            sim_matrix.at[id1, id2] = similarity
            sim_matrix.at[id2, id1] = similarity  # 矩阵对称
        
        return sim_matrix
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None

if __name__ == "__main__":
    # 文件路径
    file_path = r"H:/研究生/郭玲文章/韩培钰整理文件/440bp序列/84条序列Rdrp建树/rdrp-1_split.fasta"
    # 序列类型（'dna' 或 'protein'）
    sequence_type = 'dna'
    # 并行进程数（-1表示使用所有CPU核心）
    n_jobs = -1
    
    # 处理文件并计算相似性矩阵
    sim_matrix = process_sequence_file(file_path, sequence_type=sequence_type, n_jobs=n_jobs)
    
    if sim_matrix is not None:
        # 保存结果到CSV文件
        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)
        output_file = f"{file_name}_similarity_matrix.csv"
        sim_matrix.to_csv(output_file, index=True, header=True)
        print(f"结果已保存到 {output_file}")
        
        # 打印相似性最高的10对序列（展示方式调整，取矩阵中除对角线外的最大值）
        # 先构建不含对角线的长表
        long_form = sim_matrix.where(sim_matrix != 1).stack().reset_index()
        long_form.columns = ['Sequence1', 'Sequence2', 'Similarity']
        print("\n相似性最高的10对序列：")
        print(long_form.sort_values('Similarity', ascending=False).head(10))