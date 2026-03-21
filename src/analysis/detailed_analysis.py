#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抗登革病毒药物虚拟筛选 - 详细数据分析模块
提供全面的数据探索、模型性能分析和结果解释功能
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入配置
from src.config import DATA_DIR, RESULTS_DIR, DATA_CONFIG, MODEL_CONFIG


def analyze_data_distribution(df):
    """
    分析数据分布
    
    参数:
        df: 包含化合物数据的 DataFrame
    
    返回:
        dict: 分析结果
    """
    print("📊 数据分布分析")
    print("=" * 50)
    
    analysis_results = {}
    
    # 基本统计信息
    print(f"总样本数: {len(df)}")
    print(f"特征列数: {df.select_dtypes(include=[np.number]).shape[1]}")
    
    # 活性分布分析
    if 'active' in df.columns:
        active_count = df['active'].sum()
        inactive_count = len(df) - active_count
        active_ratio = active_count / len(df) if len(df) > 0 else 0
        
        print(f"活性化合物: {active_count} ({active_ratio:.2%})")
        print(f"非活性化合物: {inactive_count} ({1-active_ratio:.2%})")
        
        analysis_results['activity_distribution'] = {
            'active_count': active_count,
            'inactive_count': inactive_count,
            'active_ratio': active_ratio
        }
    
    # 数值特征分布
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\n数值特征统计:")
        print(df[numeric_cols].describe())
        
        analysis_results['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # SMILES 长度分布
    if 'canonical_smiles' in df.columns:
        smiles_lengths = df['canonical_smiles'].str.len()
        print(f"\nSMILES 长度分布:")
        print(f"平均长度: {smiles_lengths.mean():.2f}")
        print(f"最长: {smiles_lengths.max()}")
        print(f"最短: {smiles_lengths.min()}")
        
        analysis_results['smiles_length_stats'] = {
            'mean_length': smiles_lengths.mean(),
            'max_length': smiles_lengths.max(),
            'min_length': smiles_lengths.min()
        }
    
    return analysis_results


def analyze_model_performance(model_results_path):
    """
    分析模型性能
    
    参数:
        model_results_path: 模型结果文件路径
    
    返回:
        dict: 模型性能分析结果
    """
    print("\n📈 模型性能分析")
    print("=" * 50)
    
    # 获取所有模型结果文件
    model_files = [f for f in os.listdir(model_results_path) if f.endswith('_results.json')]
    
    if not model_files:
        print("未找到模型结果文件")
        return {}
    
    model_performance = {}
    
    for file in model_files:
        model_name = file.replace('_results.json', '')
        result_path = os.path.join(model_results_path, file)
        
        try:
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # 提取关键性能指标
            test_metrics = result.get('test_metrics', {})
            cv_results = result.get('cv_results', {})
            
            model_performance[model_name] = {
                'test_metrics': test_metrics,
                'cv_results': cv_results
            }
            
            print(f"\n{model_name} 模型:")
            print(f"  ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
            print(f"  PR-AUC: {test_metrics.get('pr_auc', 0):.4f}")
            print(f"  准确率: {test_metrics.get('accuracy', 0):.4f}")
            print(f"  精确率: {test_metrics.get('precision', 0):.4f}")
            print(f"  召回率: {test_metrics.get('recall', 0):.4f}")
            print(f"  F1 分数: {test_metrics.get('f1_score', 0):.4f}")
            
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
    
    return model_performance


def visualize_data_distributions(df, save_path=None):
    """
    可视化数据分布
    
    参数:
        df: 包含化合物数据的 DataFrame
        save_path: 图表保存路径
    """
    print(f"\n📊 生成数据分布图表")
    
    # 创建图表保存目录
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('数据分布分析', fontsize=16)
    
    # 1. 活性分布饼图
    if 'active' in df.columns:
        ax1 = axes[0, 0]
        active_counts = df['active'].value_counts()
        labels = ['非活性', '活性'] if 0 in active_counts.index else ['活性', '非活性']
        sizes = [active_counts.get(0, 0), active_counts.get(1, 0)]
        colors = ['#ff9999', '#66b3ff']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('化合物活性分布')
    
    # 2. SMILES 长度分布直方图
    if 'canonical_smiles' in df.columns:
        ax2 = axes[0, 1]
        smiles_lengths = df['canonical_smiles'].str.len()
        ax2.hist(smiles_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('SMILES 长度')
        ax2.set_ylabel('频次')
        ax2.set_title('SMILES 长度分布')
    
    # 3. 数值特征分布（前几个）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # 选择前几个数值特征进行展示
        selected_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1]
        
        if len(selected_cols) > 0:
            ax3 = axes[1, 0]
            for col in selected_cols:
                ax3.hist(df[col].dropna(), bins=30, alpha=0.5, label=col)
            ax3.set_xlabel('数值')
            ax3.set_ylabel('频次')
            ax3.set_title('数值特征分布')
            ax3.legend()
        
        # 4. 相关性热力图（选择部分特征）
        if len(numeric_cols) > 1:
            ax4 = axes[1, 1]
            sample_cols = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
            corr_matrix = df[sample_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, ax=ax4)
            ax4.set_title('特征相关性热力图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'data_distributions.png'), dpi=300, bbox_inches='tight')
        print(f"数据分布图表已保存至: {os.path.join(save_path, 'data_distributions.png')}")
    
    plt.show()


def visualize_model_performance(model_results_path, save_path=None):
    """
    可视化模型性能
    
    参数:
        model_results_path: 模型结果文件路径
        save_path: 图表保存路径
    """
    print(f"\n📈 生成模型性能图表")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # 读取模型结果
    model_files = [f for f in os.listdir(model_results_path) if f.endswith('_results.json')]
    
    if not model_files:
        print("未找到模型结果文件")
        return
    
    model_names = []
    metrics_data = {
        'roc_auc': [],
        'pr_auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for file in model_files:
        model_name = file.replace('_results.json', '')
        result_path = os.path.join(model_results_path, file)
        
        try:
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            test_metrics = result.get('test_metrics', {})
            
            model_names.append(model_name)
            for metric in metrics_data.keys():
                metrics_data[metric].append(test_metrics.get(metric, 0))
        
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
    
    # 创建性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能对比', fontsize=16)
    
    # ROC-AUC 和 PR-AUC
    ax1 = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, metrics_data['roc_auc'], width, label='ROC-AUC', alpha=0.8)
    ax1.bar(x + width/2, metrics_data['pr_auc'], width, label='PR-AUC', alpha=0.8)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('AUC 分数')
    ax1.set_title('ROC-AUC vs PR-AUC')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    
    # 准确率、精确率、召回率、F1分数
    ax2 = axes[0, 1]
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    for i, metric in enumerate(metrics_to_plot):
        ax2.plot(model_names, metrics_data[metric], marker='o', label=metric.upper(), linewidth=2)
    ax2.set_xlabel('模型')
    ax2.set_ylabel('分数')
    ax2.set_title('分类性能指标对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 性能雷达图
    ax3 = axes[1, 0]
    from math import pi
    
    # 为了绘制雷达图，我们需要将数据标准化
    if len(model_names) > 0:
        # 计算各指标的平均值
        avg_metrics = {}
        for metric in metrics_data.keys():
            avg_metrics[metric] = np.mean(metrics_data[metric])
        
        # 选择第一个模型作为示例
        angles = [n / float(len(metrics_to_plot)) * 2 * pi for n in range(len(metrics_to_plot))]
        angles += angles[:1]  # 闭合图形
        
        values = [metrics_data[metric][0] for metric in metrics_to_plot]
        values += values[:1]
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        ax3.plot(angles, values, 'o-', linewidth=2, label=model_names[0])
        ax3.fill(angles, values, alpha=0.25)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([m.upper() for m in metrics_to_plot])
        ax3.set_ylim(0, 1)
        ax3.set_title(f'{model_names[0]} 性能雷达图', pad=20)
    
    # 模型性能柱状图
    ax4 = axes[1, 1]
    # 选择最佳模型的关键指标
    if metrics_data['roc_auc']:
        best_model_idx = np.argmax(metrics_data['roc_auc'])
        best_model_name = model_names[best_model_idx]
        
        best_metrics = [metrics_data[metric][best_model_idx] for metric in ['accuracy', 'precision', 'recall', 'f1_score']]
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        bars = ax4.bar(metric_labels, best_metrics, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax4.set_ylabel('分数')
        ax4.set_title(f'最佳模型 ({best_model_name}) 性能')
        
        # 添加数值标签
        for bar, value in zip(bars, best_metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'model_performance.png'), dpi=300, bbox_inches='tight')
        print(f"模型性能图表已保存至: {os.path.join(save_path, 'model_performance.png')}")
    
    plt.show()


def analyze_feature_importance(processed_data_path, models_path, save_path=None):
    """
    分析特征重要性
    
    参数:
        processed_data_path: 处理后数据路径
        models_path: 模型路径
        save_path: 图表保存路径
    """
    print(f"\n🔍 生成特征重要性图表")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    try:
        # 加载处理后的数据
        df = pd.read_csv(processed_data_path)
        
        # 获取特征列
        non_feature_cols = ['compound_id', 'molecule_chembl_id', 'canonical_smiles', 
                           'standard_value', 'pIC50', 'is_active', 'active']
        feature_cols = [col for col in df.columns if col not in non_feature_cols and col.startswith(('Morgan_', 'MACCS_', 'HeavyAtom', 'MolWt'))]
        
        if not feature_cols:
            print("未找到合适的特征列")
            return
        
        # 加载模型并获取特征重要性
        model_files = [f for f in os.listdir(models_path) if f.endswith('_model.pkl')]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle('特征重要性分析', fontsize=16)
        
        for file in model_files:
            model_name = file.replace('_model.pkl', '')
            model_path = os.path.join(models_path, file)
            
            try:
                model = joblib.load(model_path)
                
                # 检查模型是否有特征重要性属性
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # 获取最重要的前20个特征
                    indices = np.argsort(importances)[::-1][:20]
                    
                    # 绘制特征重要性
                    plt.figure(figsize=(15, 8))
                    plt.title(f'{model_name} 特征重要性 (Top 20)')
                    plt.bar(range(len(indices)), importances[indices])
                    plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=45, ha='right')
                    plt.ylabel('重要性')
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(os.path.join(save_path, f'{model_name}_feature_importance.png'), 
                                  dpi=300, bbox_inches='tight')
                        print(f"{model_name} 特征重要性图表已保存至: {os.path.join(save_path, f'{model_name}_feature_importance.png')}")
                    
                    plt.show()
                    
            except Exception as e:
                print(f"加载模型 {model_name} 时出错: {e}")
    
    except Exception as e:
        print(f"分析特征重要性时出错: {e}")


def generate_detailed_report(data_path, model_results_path, models_path, save_dir=None):
    """
    生成详细分析报告
    
    参数:
        data_path: 数据文件路径
        model_results_path: 模型结果路径
        models_path: 模型路径
        save_dir: 报告保存目录
    """
    print(f"\n📋 生成详细分析报告")
    
    if save_dir is None:
        save_dir = os.path.join(RESULTS_DIR['reports'], f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 生成报告内容
    report_content = []
    report_content.append("# 抗登革病毒药物虚拟筛选详细分析报告\n")
    report_content.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 数据概览
    report_content.append("## 1. 数据概览\n")
    report_content.append(f"- 总样本数：{len(df)}\n")
    report_content.append(f"- 特征总数：{df.select_dtypes(include=[np.number]).shape[1]}\n")
    
    if 'active' in df.columns:
        active_count = df['active'].sum()
        inactive_count = len(df) - active_count
        report_content.append(f"- 活性化合物：{active_count} ({active_count/len(df)*100:.2f}%)\n")
        report_content.append(f"- 非活性化合物：{inactive_count} ({inactive_count/len(df)*100:.2f}%)\n")
    
    report_content.append("\n### 1.1. 数据质量评估\n")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        report_content.append("存在缺失值的列：\n")
        for col, count in null_counts[null_counts > 0].items():
            report_content.append(f"- {col}: {count} 个缺失值\n")
    else:
        report_content.append("数据无缺失值\n")
    
    # 2. 模型性能分析
    report_content.append("\n## 2. 模型性能分析\n")
    
    model_files = [f for f in os.listdir(model_results_path) if f.endswith('_results.json')]
    for file in model_files:
        model_name = file.replace('_results.json', '')
        result_path = os.path.join(model_results_path, file)
        
        try:
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            test_metrics = result.get('test_metrics', {})
            
            report_content.append(f"\n### {model_name} 模型性能\n")
            report_content.append(f"- ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}\n")
            report_content.append(f"- PR-AUC: {test_metrics.get('pr_auc', 0):.4f}\n")
            report_content.append(f"- 准确率: {test_metrics.get('accuracy', 0):.4f}\n")
            report_content.append(f"- 精确率: {test_metrics.get('precision', 0):.4f}\n")
            report_content.append(f"- 召回率: {test_metrics.get('recall', 0):.4f}\n")
            report_content.append(f"- F1 分数: {test_metrics.get('f1_score', 0):.4f}\n")
            
            # 交叉验证结果
            cv_results = result.get('cv_results', {})
            if cv_results:
                report_content.append(f"- 交叉验证 ROC-AUC: {cv_results.get('roc_auc_mean', 0):.4f} ± {cv_results.get('roc_auc_std', 0):.4f}\n")
        
        except Exception as e:
            report_content.append(f"读取 {file} 时出错: {e}\n")
    
    # 3. 结果解释
    report_content.append("\n## 3. 结果解释与建议\n")
    report_content.append("### 3.1. 模型表现评估\n")
    
    # 找出最佳模型
    best_model = None
    best_auc = 0
    for file in model_files:
        model_name = file.replace('_results.json', '')
        result_path = os.path.join(model_results_path, file)
        
        try:
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            auc = result.get('test_metrics', {}).get('roc_auc', 0)
            if auc > best_auc:
                best_auc = auc
                best_model = model_name
        except:
            pass
    
    if best_model:
        report_content.append(f"- 最佳模型：{best_model} (ROC-AUC: {best_auc:.4f})\n")
    
    report_content.append("\n### 3.2. 模型适用性分析\n")
    report_content.append("- 随机森林模型：适合处理高维特征，具有良好的泛化能力\n")
    report_content.append("- XGBoost模型：在处理不平衡数据方面表现优异\n")
    report_content.append("- SVM模型：适合小样本数据，对噪声敏感度较低\n")
    
    report_content.append("\n### 3.3. 研究建议\n")
    report_content.append("- 对预测的活性化合物进行分子对接验证\n")
    report_content.append("- 扩大训练数据集以提高模型鲁棒性\n")
    report_content.append("- 尝试深度学习模型以捕获更复杂的分子模式\n")
    report_content.append("- 进行体外实验验证预测结果\n")
    
    # 保存报告
    report_path = os.path.join(save_dir, "detailed_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"详细分析报告已保存至: {report_path}")
    
    # 生成可视化图表
    print("\n📊 生成可视化图表...")
    visualize_data_distributions(df, os.path.join(save_dir, 'figures'))
    visualize_model_performance(model_results_path, os.path.join(save_dir, 'figures'))
    
    print(f"\n✅ 详细分析完成！报告保存在: {save_dir}")


def main():
    """主函数"""
    print("🔍 抗登革病毒药物虚拟筛选 - 详细数据分析")
    print("=" * 60)
    
    # 设置路径
    processed_data_path = os.path.join(DATA_DIR['processed'], DATA_CONFIG['processed_data_file'])
    model_results_path = RESULTS_DIR['models']
    models_path = RESULTS_DIR['models']
    
    # 检查文件是否存在
    if not os.path.exists(processed_data_path):
        print(f"错误：找不到处理后的数据文件: {processed_data_path}")
        return
    
    if not os.path.exists(model_results_path):
        print(f"错误：找不到模型结果目录: {model_results_path}")
        return
    
    # 执行分析
    print("开始数据分析...")
    
    # 1. 数据分布分析
    df = pd.read_csv(processed_data_path)
    data_analysis = analyze_data_distribution(df)
    
    # 2. 模型性能分析
    model_analysis = analyze_model_performance(model_results_path)
    
    # 3. 生成详细报告
    generate_detailed_report(
        processed_data_path, 
        model_results_path, 
        models_path
    )


if __name__ == "__main__":
    main()