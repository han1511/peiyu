#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从ChEMBL数据库获取抗登革病毒活性数据
"""

import os
import sys
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置
from src.config import DATA_DIR, DATA_CONFIG, ACTIVITY_CONFIG


def fetch_dengue_data(output_dir: str = None, output_file: str = None) -> pd.DataFrame:
    """
    从ChEMBL数据库获取抗登革病毒活性数据
    
    参数:
        output_dir: 输出目录路径 (默认使用配置中的路径)
        output_file: 输出文件名 (默认使用配置中的文件名)
    
    返回:
        pd.DataFrame: 包含抗登革病毒活性数据的DataFrame
    """
    # 使用配置中的默认值
    if output_dir is None:
        output_dir = DATA_DIR['raw']
    if output_file is None:
        output_file = DATA_CONFIG['chembl_data_file']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在从ChEMBL数据库获取登革病毒目标...")
    
    # 搜索登革病毒目标
    target = new_client.target
    # 搜索更广泛的登革热相关目标，包括整个生物体和特定蛋白
    dengue_targets = target.filter(pref_name__icontains="Dengue")
    
    # 获取目标ID
    dengue_target_ids = [target['target_chembl_id'] for target in dengue_targets]
    print(f"找到 {len(dengue_target_ids)} 个登革病毒目标")
    print(f"目标ID: {dengue_target_ids}")
    
    # 搜索活性数据
    activity = new_client.activity
    
    # 获取每个目标的活性数据
    all_activities = []
    
    for target_id in tqdm(dengue_target_ids, desc="获取活性数据"):
        # 获取该目标的活性数据，放宽筛选条件
        activities = activity.filter(
            target_chembl_id=target_id,
            standard_value__isnull=False,
            relation=["<", "<=", "=", ">=", ">"],
            assay_type="B"
        ).only(
            "molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units",
            "standard_type", "activity_comment", "data_validity_comment",
            "assay_chembl_id", "assay_description", "target_chembl_id", "target_pref_name"
        )
        
        all_activities.extend(activities)
    
    print(f"共获取到 {len(all_activities)} 条活性数据")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_activities)
    
    if df.empty:
        print("未获取到任何活性数据")
        return df
    
    # 数据预处理
    print("正在预处理数据...")
    
    # 转换标准值为数值类型
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    
    # 移除标准值为NaN的行
    df = df.dropna(subset=['standard_value', 'canonical_smiles'])
    
    # 计算pIC50值
    df['pIC50'] = -np.log10(df['standard_value'] / 1e9)
    
    # 标记活性化合物 (使用配置中的阈值)
    df['active'] = (df['pIC50'] >= ACTIVITY_CONFIG['pic50_threshold']).astype(int)
    
    # 移除重复的分子
    df = df.sort_values('standard_value').drop_duplicates('molecule_chembl_id', keep='first')
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    print(f"预处理后数据量: {len(df)}")
    print(f"活性化合物数量: {df['active'].sum()}")
    print(f"非活性化合物数量: {len(df) - df['active'].sum()}")
    
    # 保存数据
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"数据已保存到: {output_path}")
    
    return df


if __name__ == "__main__":
    # 获取数据
    df = fetch_dengue_data()
    
    # 如果获取到数据，显示基本信息
    if not df.empty:
        print("\n数据基本信息:")
        print(df.info())
        print("\n数据统计描述:")
        print(df.describe())
