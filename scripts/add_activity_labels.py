#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为化合物数据添加活性标签
"""

import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_activity_labels(input_file, output_file=None, active_ratio=0.3):
    """
    为化合物数据添加活性标签
    
    参数:
        input_file: 输入 CSV 文件路径
        output_file: 输出 CSV 文件路径（可选）
        active_ratio: 活性化合物比例（0-1 之间）
    """
    # 加载数据
    print(f"正在加载数据：{input_file}")
    df = pd.read_csv(input_file)
    
    # 检查是否包含 SMILES 列
    if 'canonical_smiles' not in df.columns:
        if 'SMILES' in df.columns:
            df = df.rename(columns={'SMILES': 'canonical_smiles'})
        else:
            print("错误：文件中没有 SMILES 列")
            return
    
    # 添加活性标签
    # 这里使用随机分配，实际应用中应该使用实验数据
    import numpy as np
    np.random.seed(42)
    
    total_compounds = len(df)
    active_count = int(total_compounds * active_ratio)
    
    # 创建活性标签
    labels = np.zeros(total_compounds, dtype=int)
    labels[:active_count] = 1
    np.random.shuffle(labels)
    
    df['active'] = labels
    
    # 保存结果
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        output_file = os.path.join(input_dir, f"labeled_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n数据标签添加成功！")
    print(f"总化合物数：{total_compounds}")
    print(f"活性化合物数：{active_count} ({active_ratio*100:.1f}%)")
    print(f"非活性化合物数：{total_compounds - active_count} ({(1-active_ratio)*100:.1f}%)")
    print(f"保存位置：{output_file}")
    
    return output_file

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        active_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        add_activity_labels(input_file, output_file, active_ratio)
    else:
        print("用法：python add_activity_labels.py <输入文件> [输出文件] [活性比例]")
        print("示例：python add_activity_labels.py compounds.csv labeled_compounds.csv 0.3")
