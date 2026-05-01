#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果可视化：生成汇总图表
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载筛选结果"""
    result_files = []
    results_dir = "E:/Python/dengue_drug_discovery/results"
    for item in os.listdir(results_dir):
        if item.startswith("NS5_") and os.path.isdir(os.path.join(results_dir, item)):
            json_file = os.path.join(results_dir, item, "screening_results.json")
            if os.path.exists(json_file):
                result_files.append(json_file)
    
    if not result_files:
        print("未找到筛选结果文件")
        return None
    
    # 选择最新的结果
    result_files.sort()
    latest_file = result_files[-1]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results, latest_file

def plot_results(results):
    """绘制汇总图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top分数分布
    top_scores = results['top_scores'][:50]
    axes[0, 0].bar(range(len(top_scores)), top_scores, color='#1f77b4')
    axes[0, 0].set_title('Top 50 化合物预测分数分布', fontsize=14)
    axes[0, 0].set_xlabel('化合物排名', fontsize=12)
    axes[0, 0].set_ylabel('预测分数', fontsize=12)
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. 分数直方图
    all_scores = results['top_scores']
    axes[0, 1].hist(all_scores, bins=20, color='#ff7f0e', edgecolor='black')
    axes[0, 1].set_title('Top 100 化合物分数直方图', fontsize=14)
    axes[0, 1].set_xlabel('预测分数', fontsize=12)
    axes[0, 1].set_ylabel('化合物数量', fontsize=12)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. 模型性能对比
    models = ['XGBoost', 'RandomForest', 'SVM', 'LogisticRegression']
    auc_scores = [0.9581, 0.9523, 0.9145, 0.8762]  # 来自日志
    axes[1, 0].bar(models, auc_scores, color=['#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[1, 0].set_title('各模型AUC性能对比', fontsize=14)
    axes[1, 0].set_xlabel('模型名称', fontsize=12)
    axes[1, 0].set_ylabel('AUC值', fontsize=12)
    axes[1, 0].set_ylim(0.7, 1.0)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. 数据分布饼图
    labels = ['活性化合物', '非活性化合物']
    sizes = [405, 405]
    colors = ['#17becf', '#e377c2']
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
    axes[1, 1].set_title('训练数据类别分布', fontsize=14)
    
    plt.tight_layout(pad=3)
    plt.savefig('E:/Python/dengue_drug_discovery/results/screening_summary.png', 
                dpi=300, bbox_inches='tight')
    print("图表已保存: screening_summary.png")
    
    plt.show()

def print_summary(results, file_path):
    """打印文本摘要"""
    print("=" * 70)
    print("登革病毒NS5抑制剂虚拟筛选结果汇总")
    print("=" * 70)
    print(f"结果文件: {file_path}")
    print(f"靶点: {results.get('target', 'N/A')}")
    print(f"PDB ID: {results.get('pdb_id', 'N/A')}")
    print(f"筛选化合物总数: {results.get('compounds_screened', 0):,}")
    print(f"Top化合物数: {len(results.get('top_indices', []))}")
    print(f"\nTop 5 化合物分数:")
    for i, score in enumerate(results.get('top_scores', [])[:5], 1):
        print(f"  {i}. {score:.4f}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    results, file_path = load_results()
    if results:
        print_summary(results, file_path)
        plot_results(results)
