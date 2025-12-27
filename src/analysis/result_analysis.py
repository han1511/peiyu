#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果分析和可视化模块
用于分析和可视化模型训练和虚拟筛选的结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置
from src.config import DATA_DIR, RESULTS_DIR, DATA_CONFIG


def load_model_results(model_name: str) -> dict:
    """
    加载模型训练结果
    
    参数:
        model_name: 模型名称
    
    返回:
        dict: 模型结果字典
    """
    results_path = os.path.join(RESULTS_DIR['models'], f'{model_name}_results.json')
    
    if not os.path.exists(results_path):
        print(f"模型结果文件不存在: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def load_model_predictions(model_name: str) -> dict:
    """
    加载模型预测结果
    
    参数:
        model_name: 模型名称
    
    返回:
        dict: 预测结果字典
    """
    predictions_path = os.path.join(RESULTS_DIR['models'], f'{model_name}_test_predictions.pkl')
    
    if not os.path.exists(predictions_path):
        print(f"预测结果文件不存在: {predictions_path}")
        return None
    
    import joblib
    predictions = joblib.load(predictions_path)
    
    return predictions


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_fig: bool = True) -> plt.Figure:
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        save_fig: 是否保存图像
    
    返回:
        plt.Figure: 混淆矩阵图像
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['非活性', '活性'],
               yticklabels=['非活性', '活性'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(RESULTS_DIR['figures'], f'{model_name}_confusion_matrix.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵保存到: {fig_path}")
    
    return plt.gcf()


def plot_roc_curve(y_true, y_pred_prob, model_name: str, save_fig: bool = True) -> plt.Figure:
    """
    绘制ROC曲线
    
    参数:
        y_true: 真实标签
        y_pred_prob: 预测概率
        model_name: 模型名称
        save_fig: 是否保存图像
    
    返回:
        plt.Figure: ROC曲线图像
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(RESULTS_DIR['figures'], f'{model_name}_roc_curve.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线保存到: {fig_path}")
    
    return plt.gcf()


def plot_precision_recall_curve(y_true, y_pred_prob, model_name: str, save_fig: bool = True) -> plt.Figure:
    """
    绘制Precision-Recall曲线
    
    参数:
        y_true: 真实标签
        y_pred_prob: 预测概率
        model_name: 模型名称
        save_fig: 是否保存图像
    
    返回:
        plt.Figure: Precision-Recall曲线图像
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    from sklearn.metrics import average_precision_score
    ap_score = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR曲线 (AP = {ap_score:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'{model_name} Precision-Recall曲线')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(RESULTS_DIR['figures'], f'{model_name}_pr_curve.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall曲线保存到: {fig_path}")
    
    return plt.gcf()


def plot_model_comparison(models_list: list, save_fig: bool = True) -> plt.Figure:
    """
    比较不同模型的性能指标
    
    参数:
        models_list: 模型名称列表
        save_fig: 是否保存图像
    
    返回:
        plt.Figure: 模型比较图像
    """
    # 收集所有模型的性能指标
    metrics_data = []
    
    for model_name in models_list:
        results = load_model_results(model_name)
        if results:
            metrics = results.get('test_metrics', {})
            metrics['model'] = model_name
            metrics_data.append(metrics)
    
    if not metrics_data:
        print("没有模型结果数据可比较")
        return None
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # 选择要比较的指标
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'mcc']
    metrics_to_plot = [m for m in metrics_to_plot if m in df_metrics.columns]
    
    # 创建多子图
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    # 绘制每个指标的条形图
    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='model', y=metric, data=df_metrics, ax=axes[i])
        axes[i].set_title(metric.upper())
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for j, v in enumerate(df_metrics[metric]):
            axes[i].text(j, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(RESULTS_DIR['figures'], 'model_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"模型比较图保存到: {fig_path}")
    
    return plt.gcf()


def analyze_screening_results(screening_results_path: str, model_name: str, save_fig: bool = True) -> dict:
    """
    分析虚拟筛选结果
    
    参数:
        screening_results_path: 虚拟筛选结果文件路径
        model_name: 模型名称
        save_fig: 是否保存图像
    
    返回:
        dict: 筛选结果分析
    """
    if not os.path.exists(screening_results_path):
        print(f"筛选结果文件不存在: {screening_results_path}")
        return None
    
    df = pd.read_csv(screening_results_path)
    
    # 基本统计信息
    total_compounds = len(df)
    predicted_active = df['prediction'].sum()
    predicted_inactive = total_compounds - predicted_active
    
    hit_rate = predicted_active / total_compounds * 100
    
    print(f"总化合物数: {total_compounds}")
    print(f"预测活性化合物数: {predicted_active}")
    print(f"预测非活性化合物数: {predicted_inactive}")
    print(f"命中率: {hit_rate:.2f}%")
    
    # 绘制预测概率分布
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='average_probability', hue='prediction',
                bins=30, kde=True, palette=['blue', 'red'])
    plt.xlabel('预测概率')
    plt.ylabel('化合物数')
    plt.title(f'{model_name} 虚拟筛选预测概率分布')
    plt.legend(['非活性', '活性'])
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(RESULTS_DIR['figures'], f'{model_name}_screening_prob_dist.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"预测概率分布图保存到: {fig_path}")
    
    # 保存筛选结果分析
    analysis_results = {
        'total_compounds': total_compounds,
        'predicted_active': predicted_active,
        'predicted_inactive': predicted_inactive,
        'hit_rate': hit_rate
    }
    
    return analysis_results


def generate_analysis_report(models_list: list, screening_results_path: str = None) -> str:
    """
    生成综合分析报告
    
    参数:
        models_list: 模型名称列表
        screening_results_path: 虚拟筛选结果文件路径
    
    返回:
        str: 报告保存路径
    """
    print("=" * 80)
    print("开始生成分析报告")
    print("=" * 80)
    
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR['figures'], exist_ok=True)
    os.makedirs(RESULTS_DIR['tables'], exist_ok=True)
    
    # 1. 模型性能分析
    print("\n1. 模型性能分析")
    print("-" * 60)
    
    for model_name in models_list:
        print(f"\n模型: {model_name}")
        print("=" * 40)
        
        # 加载结果
        results = load_model_results(model_name)
        predictions = load_model_predictions(model_name)
        
        if results and predictions:
            # 打印性能指标
            print("性能指标:")
            for metric, value in results['test_metrics'].items():
                print(f"{metric}: {value:.4f}")
            
            # 绘制评估图像
            plt.close('all')  # 关闭之前的图像
            
            # 混淆矩阵
            plot_confusion_matrix(predictions['y_true'], predictions['y_pred'], model_name)
            
            # ROC曲线
            plot_roc_curve(predictions['y_true'], predictions['y_pred_prob'], model_name)
            
            # Precision-Recall曲线
            plot_precision_recall_curve(predictions['y_true'], predictions['y_pred_prob'], model_name)
    
    # 2. 模型比较
    print("\n2. 模型比较")
    print("-" * 60)
    
    plt.close('all')
    plot_model_comparison(models_list)
    
    # 3. 虚拟筛选结果分析
    if screening_results_path:
        print("\n3. 虚拟筛选结果分析")
        print("-" * 60)
        
        plt.close('all')
        analyze_screening_results(screening_results_path, 'VirtualScreening')
    
    # 4. 生成HTML报告
    report_path = os.path.join(RESULTS_DIR['tables'], 'analysis_report.html')
    
    # 创建HTML报告
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>抗登革病毒药物筛选分析报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        h3 { color: #2ecc71; }
        .metric { margin: 10px 0; }
        .figure { margin: 20px 0; }
        .figure img { max-width: 100%; height: auto; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>抗登革病毒药物筛选分析报告</h1>
    
    <h2>1. 模型性能分析</h2>
"""
    
    # 添加模型性能指标到报告
    for model_name in models_list:
        results = load_model_results(model_name)
        if results:
            html_content += f"<h3>{model_name}</h3>"
            html_content += "<div class='metrics'>"
            for metric, value in results['test_metrics'].items():
                html_content += f"<div class='metric'><strong>{metric}:</strong> {value:.4f}</div>"
            html_content += "</div>"
            
            # 添加图像
            html_content += f"<div class='figure'><h4>混淆矩阵</h4><img src='../figures/{model_name}_confusion_matrix.png'></div>"
            html_content += f"<div class='figure'><h4>ROC曲线</h4><img src='../figures/{model_name}_roc_curve.png'></div>"
            html_content += f"<div class='figure'><h4>Precision-Recall曲线</h4><img src='../figures/{model_name}_pr_curve.png'></div>"
    
    # 添加模型比较
    html_content += "<h2>2. 模型比较</h2>"
    html_content += f"<div class='figure'><img src='../figures/model_comparison.png'></div>"
    
    # 添加虚拟筛选结果
    if screening_results_path:
        html_content += "<h2>3. 虚拟筛选结果分析</h2>"
        html_content += f"<div class='figure'><img src='../figures/VirtualScreening_screening_prob_dist.png'></div>"
    
    html_content += """
</body>
</html>
"""
    
    # 保存HTML报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n分析报告已生成: {report_path}")
    print("=" * 80)
    print("分析报告生成完成！")
    print("=" * 80)
    
    return report_path


if __name__ == "__main__":
    # 示例用法
    models = ['RandomForest', 'XGBoost', 'SVM', 'Ensemble']
    
    # 生成分析报告
    generate_analysis_report(models)
