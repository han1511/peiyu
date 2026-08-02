#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习模型性能可视化模块

功能：
1. ROC曲线
2. PR曲线  
3. 混淆矩阵热力图
4. 学习曲线
5. 特征重要性图
6. 模型对比柱状图
7. 校准曲线
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve, validation_curve

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案 - 学术论文风格
COLORS = {
    'primary': '#2E5C8A',
    'secondary': '#E67E22', 
    'success': '#27AE60',
    'danger': '#C0392B',
    'warning': '#F39C12',
    'info': '#3498DB',
    'purple': '#8E44AD',
    'teal': '#16A085',
    'gray': '#7F8C8D',
    'light_gray': '#BDC3C7',
    'palette': ['#2E5C8A', '#E67E22', '#27AE60', '#8E44AD', '#C0392B', 
                '#16A085', '#D35400', '#2980B9', '#27AE60', '#F39C12']
}


class ModelVisualizer:
    """
    模型性能可视化类
    
    生成论文级别的机器学习模型性能图表
    """
    
    def __init__(self, output_dir: Optional[Path] = None, style: str = 'seaborn-v0_8-whitegrid'):
        """
        初始化模型可视化器
        
        参数:
            output_dir: 图表输出目录
            style: matplotlib样式
        """
        self.output_dir = output_dir or Path("results/visualization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            
            # 设置全局字体大小
            plt.rcParams.update({
                'font.size': 11,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.2
            })
    
    def plot_roc_curve(self,
                      y_true_dict: Dict[str, np.ndarray],
                      y_proba_dict: Dict[str, np.ndarray],
                      title: str = "ROC Curve",
                      filename: str = "roc_curve.png",
                      figsize: Tuple[int, int] = (8, 7)) -> str:
        """
        绘制ROC曲线
        
        参数:
            y_true_dict: {模型名: 真实标签}
            y_proba_dict: {模型名: 预测概率}
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Random (AUC = 0.500)')
        
        colors = COLORS['palette']
        
        for idx, (model_name, y_true) in enumerate(y_true_dict.items()):
            if model_name not in y_proba_dict:
                continue
                
            y_proba = y_proba_dict[model_name]
            
            # 处理多列概率输出
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]
            elif y_proba.ndim > 1:
                y_proba = y_proba.ravel()
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, color=color, linewidth=2.5, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # 添加AUC评分区域标注
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
        return str(output_path)
    
    def plot_precision_recall_curve(self,
                                   y_true_dict: Dict[str, np.ndarray],
                                   y_proba_dict: Dict[str, np.ndarray],
                                   title: str = "Precision-Recall Curve",
                                   filename: str = "pr_curve.png",
                                   figsize: Tuple[int, int] = (8, 7)) -> str:
        """
        绘制PR曲线
        
        参数:
            y_true_dict: {模型名: 真实标签}
            y_proba_dict: {模型名: 预测概率}
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = COLORS['palette']
        
        for idx, (model_name, y_true) in enumerate(y_true_dict.items()):
            if model_name not in y_proba_dict:
                continue
                
            y_proba = y_proba_dict[model_name]
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]
            elif y_proba.ndim > 1:
                y_proba = y_proba.ravel()
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            
            # 计算baseline
            baseline = np.sum(y_true) / len(y_true)
            
            color = colors[idx % len(colors)]
            ax.plot(recall, precision, color=color, linewidth=2.5,
                   label=f'{model_name} (AP = {pr_auc:.3f})')
        
        # 绘制baseline
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, 
                  alpha=0.6, label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {output_path}")
        return str(output_path)
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str] = None,
                             title: str = "Confusion Matrix",
                             filename: str = "confusion_matrix.png",
                             figsize: Tuple[int, int] = (8, 7),
                             normalize: bool = False) -> str:
        """
        绘制混淆矩阵热力图
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            normalize: 是否归一化
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        if class_names is None:
            class_names = ['Inactive', 'Active']
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            cmap_title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            cmap_title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用自定义颜色映射
        cmap = sns.color_palette("Blues", as_cmap=True)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=class_names, yticklabels=class_names,
                   annot_kws={'size': 14, 'weight': 'bold'},
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   linewidths=2, linecolor='white', ax=ax)
        
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=13)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=13)
        ax.set_title(title, fontweight='bold', pad=15, fontsize=15)
        
        # 添加性能指标文本
        tn, fp, fn, tp = cm.ravel() if not normalize else (0, 0, 0, 0)
        if not normalize:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_text = f'Accuracy: {accuracy:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}'
            ax.text(0.5, -0.12, metrics_text, transform=ax.transAxes,
                   ha='center', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        return str(output_path)
    
    def plot_model_comparison(self,
                             metrics_df: pd.DataFrame,
                             metrics: List[str] = None,
                             title: str = "Model Performance Comparison",
                             filename: str = "model_comparison.png",
                             figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        绘制模型性能对比图
        
        参数:
            metrics_df: 包含模型性能的DataFrame，必须包含'Model'列
            metrics: 要对比的指标列表
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            # 只保留存在的列
            metrics = [m for m in metrics if m in metrics_df.columns]
        
        if 'Model' not in metrics_df.columns:
            logger.error("metrics_df must contain 'Model' column")
            return ""
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            logger.error("No valid metrics to plot")
            return ""
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = COLORS['palette']
        models = metrics_df['Model'].values
        x_pos = np.arange(len(models))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = metrics_df[metric].values
            
            bars = ax.bar(x_pos, values, color=colors[:len(models)], 
                         edgecolor='white', linewidth=1.5, alpha=0.85)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylim([0, 1.15])
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, linewidth=1)
            
            # 高亮最大值
            max_idx = np.argmax(values)
            bars[max_idx].set_edgecolor('gold')
            bars[max_idx].set_linewidth(3)
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison saved to {output_path}")
        return str(output_path)
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importance_values: np.ndarray,
                               title: str = "Feature Importance",
                               filename: str = "feature_importance.png",
                               figsize: Tuple[int, int] = (10, 8),
                               top_k: int = 20) -> str:
        """
        绘制特征重要性图
        
        参数:
            feature_names: 特征名称列表
            importance_values: 特征重要性值
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            top_k: 显示前k个特征
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        # 排序并选择top_k
        indices = np.argsort(importance_values)[::-1][:top_k]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance_values[indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(top_features))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(y_pos, top_importance, color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加累积重要性线
        cumsum = np.cumsum(top_importance)
        ax2 = ax.twiny()
        ax2.plot(cumsum, y_pos, 'ro-', linewidth=2, markersize=4, alpha=0.7)
        ax2.set_xlabel('Cumulative Importance', fontweight='bold', color='red')
        ax2.tick_params(axis='x', colors='red')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance saved to {output_path}")
        return str(output_path)
    
    def plot_learning_curve(self,
                           estimator,
                           X: np.ndarray,
                           y: np.ndarray,
                           cv: int = 5,
                           title: str = "Learning Curve",
                           filename: str = "learning_curve.png",
                           figsize: Tuple[int, int] = (10, 7)) -> str:
        """
        绘制学习曲线
        
        参数:
            estimator: sklearn模型
            X: 特征矩阵
            y: 标签
            cv: 交叉验证折数
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(train_sizes, train_mean, 'o-', color=COLORS['primary'], 
               linewidth=2.5, label='Training Score', markersize=6)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color=COLORS['primary'])
        
        ax.plot(train_sizes, val_mean, 's-', color=COLORS['secondary'],
               linewidth=2.5, label='Validation Score', markersize=6)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color=COLORS['secondary'])
        
        ax.set_xlabel('Training Set Size', fontweight='bold')
        ax.set_ylabel('ROC-AUC Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # 添加gap标注
        final_gap = train_mean[-1] - val_mean[-1]
        ax.annotate(f'Gap: {final_gap:.3f}', 
                   xy=(train_sizes[-1], val_mean[-1]),
                   xytext=(train_sizes[-1]*0.7, val_mean[-1] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                   fontsize=10, color='gray', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curve saved to {output_path}")
        return str(output_path)
    
    def plot_calibration_curve(self,
                              y_true_dict: Dict[str, np.ndarray],
                              y_proba_dict: Dict[str, np.ndarray],
                              title: str = "Calibration Curve",
                              filename: str = "calibration_curve.png",
                              figsize: Tuple[int, int] = (8, 7)) -> str:
        """
        绘制校准曲线
        
        参数:
            y_true_dict: {模型名: 真实标签}
            y_proba_dict: {模型名: 预测概率}
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
            
        from sklearn.calibration import calibration_curve
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly Calibrated')
        
        colors = COLORS['palette']
        
        for idx, (model_name, y_true) in enumerate(y_true_dict.items()):
            if model_name not in y_proba_dict:
                continue
                
            y_proba = y_proba_dict[model_name]
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]
            elif y_proba.ndim > 1:
                y_proba = y_proba.ravel()
            
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            
            color = colors[idx % len(colors)]
            ax.plot(prob_pred, prob_true, 's-', color=color, linewidth=2.5,
                   label=model_name, markersize=6)
        
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curve saved to {output_path}")
        return str(output_path)
    
    def generate_all_model_plots(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_proba: np.ndarray,
                                model_name: str = "Model",
                                feature_names: List[str] = None,
                                feature_importance: np.ndarray = None) -> Dict[str, str]:
        """
        一键生成所有模型性能图表
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            model_name: 模型名称
            feature_names: 特征名称
            feature_importance: 特征重要性
            
        返回:
            dict: {图表名: 文件路径}
        """
        results = {}
        
        # ROC曲线
        results['roc_curve'] = self.plot_roc_curve(
            {model_name: y_true},
            {model_name: y_proba},
            title=f"ROC Curve - {model_name}",
            filename=f"{model_name.lower()}_roc_curve.png"
        )
        
        # PR曲线
        results['pr_curve'] = self.plot_precision_recall_curve(
            {model_name: y_true},
            {model_name: y_proba},
            title=f"Precision-Recall Curve - {model_name}",
            filename=f"{model_name.lower()}_pr_curve.png"
        )
        
        # 混淆矩阵
        results['confusion_matrix'] = self.plot_confusion_matrix(
            y_true, y_pred,
            title=f"Confusion Matrix - {model_name}",
            filename=f"{model_name.lower()}_confusion_matrix.png"
        )
        
        # 校准曲线
        results['calibration_curve'] = self.plot_calibration_curve(
            {model_name: y_true},
            {model_name: y_proba},
            title=f"Calibration Curve - {model_name}",
            filename=f"{model_name.lower()}_calibration_curve.png"
        )
        
        # 特征重要性
        if feature_names is not None and feature_importance is not None:
            results['feature_importance'] = self.plot_feature_importance(
                feature_names, feature_importance,
                title=f"Feature Importance - {model_name}",
                filename=f"{model_name.lower()}_feature_importance.png"
            )
        
        return results
