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
from src.config import DATA_DIR, DATA_CONFIG, ACTIVITY_CONFIG, RESULTS_DIR


def activity_label(ic50):
    """
    根据IC50值标记化合物活性
    
    参数:
        ic50: IC50值 (nM)
    
    返回:
        str: 活性标签 ('active', 'intermediate', 'inactive')
    """
    if ic50 < 1000:
        return 'active'
    elif ic50 > 10000:
        return 'inactive'
    else:
        return 'intermediate'


def calculate_pic50(ic50):
    """
    计算pIC50值
    
    参数:
        ic50: IC50值 (nM)
    
    返回:
        float: pIC50值
    """
    # 限制IC50值，避免负的pIC50
    if ic50 > (10**8):
        ic50 = 10**8
    # 转换为M并计算负对数
    m = ic50 * (10**-9)
    return -np.log10(m)


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
    
    # 处理SMILES字符串，获取最长分子组件
    def longest_smiles(smile_string):
        cpd = str(smile_string).split('.')
        return max(cpd, key=len)
    
    df['canonical_smiles'] = df.canonical_smiles.map(longest_smiles)
    
    # 异常值处理
    print("\n异常值分析:")
    print(df['standard_value'].describe())
    
    # 移除异常值（超过5*10^7的IC50值）
    initial_count = len(df)
    df = df[df['standard_value'] < 5 * 10**7]
    print(f"移除异常值后数据量: {len(df)}/{initial_count}")
    
    # 计算pIC50值
    df['pIC50'] = df['standard_value'].map(calculate_pic50)
    
    # 标记活性化合物
    df['activity_class'] = df['standard_value'].map(activity_label)
    df['active'] = (df['pIC50'] >= ACTIVITY_CONFIG['pic50_threshold']).astype(int)
    
    # 移除重复的分子
    df = df.sort_values('standard_value').drop_duplicates('molecule_chembl_id', keep='first')
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    print(f"预处理后数据量: {len(df)}")
    print(f"活性化合物数量: {df['active'].sum()}")
    print(f"非活性化合物数量: {len(df) - df['active'].sum()}")
    print(f"活性化合物比例: {df['active'].sum()/len(df):.2%}")
    
    # 活性分类统计
    class_counts = df['activity_class'].value_counts()
    print("\n活性分类统计:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} ({count/len(df):.2%})")
    
    # 数据分布可视化
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n数据分布分析:")
    # 创建结果目录
    os.makedirs(RESULTS_DIR['figures'], exist_ok=True)
    
    # 1. IC50值分布（对数刻度）
    plt.figure(figsize=(10, 6))
    log_bins = np.logspace(np.log10(df['standard_value'].min()), np.log10(df['standard_value'].max()), 100)
    plt.hist(df['standard_value'], bins=log_bins, log=True, edgecolor='black')
    plt.xscale('log')
    plt.xlabel('IC50 (nM)')
    plt.ylabel('Frequency (log scale)')
    plt.title('IC50 Value Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ic50_dist_path = os.path.join(RESULTS_DIR['figures'], 'ic50_distribution.png')
    plt.savefig(ic50_dist_path, dpi=300)
    plt.close()
    print(f"IC50分布图已保存到: {ic50_dist_path}")
    
    # 2. pIC50值分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pIC50'], bins=30, kde=True, edgecolor='black')
    plt.xlabel('pIC50')
    plt.ylabel('Frequency')
    plt.title('pIC50 Value Distribution')
    plt.axvline(x=ACTIVITY_CONFIG['pic50_threshold'], color='red', linestyle='--', label=f"活性阈值: {ACTIVITY_CONFIG['pic50_threshold']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pic50_dist_path = os.path.join(RESULTS_DIR['figures'], 'pic50_distribution.png')
    plt.savefig(pic50_dist_path, dpi=300)
    plt.close()
    print(f"pIC50分布图已保存到: {pic50_dist_path}")
    
    # 3. 活性/非活性分布
    plt.figure(figsize=(8, 6))
    activity_counts = df['active'].value_counts()
    colors = ['red', 'green']
    plt.pie(activity_counts.values, labels=['非活性', '活性'], autopct='%1.1f%%', colors=colors)
    plt.title('活性/非活性化合物分布')
    plt.tight_layout()
    activity_dist_path = os.path.join(RESULTS_DIR['figures'], 'activity_distribution.png')
    plt.savefig(activity_dist_path, dpi=300)
    plt.close()
    print(f"活性分布饼图已保存到: {activity_dist_path}")
    
    # 4. 活性分类分布
    plt.figure(figsize=(8, 6))
    class_counts = df['activity_class'].value_counts()
    colors = ['green', 'orange', 'red']
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('活性分类分布')
    plt.tight_layout()
    class_dist_path = os.path.join(RESULTS_DIR['figures'], 'activity_class_distribution.png')
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    print(f"活性分类分布图已保存到: {class_dist_path}")
    
    # 5. IC50 vs pIC50散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['standard_value'], df['pIC50'], alpha=0.5)
    plt.xscale('log')
    plt.xlabel('IC50 (nM)')
    plt.ylabel('pIC50')
    plt.title('IC50 vs pIC50 Relationship')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ic50_pic50_path = os.path.join(RESULTS_DIR['figures'], 'ic50_vs_pic50.png')
    plt.savefig(ic50_pic50_path, dpi=300)
    plt.close()
    print(f"IC50 vs pIC50散点图已保存到: {ic50_pic50_path}")
    
    # 6. 活性化合物的pIC50分布
    plt.figure(figsize=(10, 6))
    active_df = df[df['active'] == 1]
    inactive_df = df[df['active'] == 0]
    sns.histplot(active_df['pIC50'], bins=20, kde=True, color='green', label='Active', alpha=0.6)
    sns.histplot(inactive_df['pIC50'], bins=20, kde=True, color='red', label='Inactive', alpha=0.6)
    plt.xlabel('pIC50')
    plt.ylabel('Frequency')
    plt.title('pIC50 Distribution by Activity')
    plt.axvline(x=ACTIVITY_CONFIG['pic50_threshold'], color='blue', linestyle='--', label=f"活性阈值: {ACTIVITY_CONFIG['pic50_threshold']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pic50_activity_path = os.path.join(RESULTS_DIR['figures'], 'pic50_distribution_by_activity.png')
    plt.savefig(pic50_activity_path, dpi=300)
    plt.close()
    print(f"活性化合物pIC50分布图已保存到: {pic50_activity_path}")
    
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
