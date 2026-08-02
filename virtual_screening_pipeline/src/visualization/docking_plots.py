#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接结果可视化模块

功能：
1. 结合能分布直方图/箱线图
2. Top化合物结合能排序图
3. 对接姿势RMSD分布
4. 结合模式对比图
5. 相互作用类型统计
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# 配色方案
COLORS = {
    'strong': '#27AE60',
    'moderate': '#F39C12', 
    'weak': '#E67E22',
    'negligible': '#C0392B',
    'palette': ['#2E5C8A', '#E67E22', '#27AE60', '#8E44AD', '#C0392B', 
                '#16A085', '#D35400', '#2980B9', '#27AE60', '#F39C12']
}


class DockingVisualizer:
    """
    分子对接结果可视化类
    
    生成论文级别的分子对接结果图表
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化对接可视化器
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir or Path("results/visualization/docking")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
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
    
    def plot_binding_affinity_distribution(self,
                                          df: pd.DataFrame,
                                          affinity_col: str = "best_affinity",
                                          title: str = "Binding Affinity Distribution",
                                          filename: str = "affinity_distribution.png",
                                          figsize: Tuple[int, int] = (10, 7),
                                          thresholds: Dict[str, float] = None) -> str:
        """
        绘制结合能分布直方图和核密度估计
        
        参数:
            df: 包含对接结果的DataFrame
            affinity_col: 结合能列名
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            thresholds: 结合能阈值 {'strong': -9.0, 'moderate': -7.0, 'weak': -5.0}
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        if affinity_col not in df.columns:
            logger.error(f"Column {affinity_col} not found in DataFrame")
            return ""
        
        if thresholds is None:
            thresholds = {'strong': -9.0, 'moderate': -7.0, 'weak': -5.0}
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        affinities = df[affinity_col].dropna()
        
        # 主图：直方图+KDE
        ax1 = axes[0]
        
        # 按结合能分类着色
        colors_hist = []
        for val in affinities:
            if val <= thresholds['strong']:
                colors_hist.append(COLORS['strong'])
            elif val <= thresholds['moderate']:
                colors_hist.append(COLORS['moderate'])
            elif val <= thresholds['weak']:
                colors_hist.append(COLORS['weak'])
            else:
                colors_hist.append(COLORS['negligible'])
        
        n, bins, patches = ax1.hist(affinities, bins=50, alpha=0.7, 
                                     color=COLORS['palette'][0], edgecolor='white', linewidth=1)
        
        # 添加KDE曲线
        try:
            from scipy import stats
            kde = stats.gaussian_kde(affinities)
            x_range = np.linspace(affinities.min(), affinities.max(), 200)
            ax1.plot(x_range, kde(x_range) * len(affinities) * (bins[1]-bins[0]), 
                    'r-', linewidth=2.5, label='KDE')
        except Exception:
            pass
        
        # 添加阈值线
        for name, thresh in thresholds.items():
            color = COLORS.get(name, 'gray')
            ax1.axvline(x=thresh, color=color, linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'{name.capitalize()} ({thresh})')
        
        ax1.set_xlabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title(title, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_aff = affinities.mean()
        median_aff = affinities.median()
        stats_text = f'Mean: {mean_aff:.2f} | Median: {median_aff:.2f} | N: {len(affinities)}'
        ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes,
                ha='right', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 子图：箱线图
        ax2 = axes[1]
        bp = ax2.boxplot(affinities, vert=False, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        bp['boxes'][0].set_facecolor(COLORS['palette'][0])
        bp['boxes'][0].set_alpha(0.7)
        
        ax2.set_xlabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax2.set_yticklabels(['All Compounds'])
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Affinity distribution saved to {output_path}")
        return str(output_path)
    
    def plot_top_compounds_ranking(self,
                                  df: pd.DataFrame,
                                  top_n: int = 20,
                                  affinity_col: str = "best_affinity",
                                  name_col: str = "ligand_name",
                                  title: str = "Top Compounds by Binding Affinity",
                                  filename: str = "top_compounds_ranking.png",
                                  figsize: Tuple[int, int] = (10, 10)) -> str:
        """
        绘制Top化合物结合能排序水平柱状图
        
        参数:
            df: 包含对接结果的DataFrame
            top_n: 显示前N个化合物
            affinity_col: 结合能列名
            name_col: 化合物名称列名
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        df_sorted = df.sort_values(affinity_col, ascending=True).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        names = df_sorted[name_col].astype(str).str[:20]  # 截断长名称
        affinities = df_sorted[affinity_col]
        
        # 颜色根据结合能强度
        colors = []
        for val in affinities:
            if val <= -9.0:
                colors.append(COLORS['strong'])
            elif val <= -7.0:
                colors.append(COLORS['moderate'])
            elif val <= -5.0:
                colors.append(COLORS['weak'])
            else:
                colors.append(COLORS['negligible'])
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, affinities, color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar, val in zip(bars, affinities):
            width = bar.get_width()
            ax.text(width - 0.3, bar.get_y() + bar.get_height()/2.,
                   f'{val:.2f}', ha='right', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['strong'], label='Strong (≤ -9.0)'),
            Patch(facecolor=COLORS['moderate'], label='Moderate (-9.0 to -7.0)'),
            Patch(facecolor=COLORS['weak'], label='Weak (-7.0 to -5.0)'),
            Patch(facecolor=COLORS['negligible'], label='Negligible (> -5.0)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top compounds ranking saved to {output_path}")
        return str(output_path)
    
    def plot_affinity_vs_rmsd(self,
                             df: pd.DataFrame,
                             affinity_col: str = "best_affinity",
                             rmsd_col: str = "rmsd_lb",
                             title: str = "Binding Affinity vs RMSD",
                             filename: str = "affinity_vs_rmsd.png",
                             figsize: Tuple[int, int] = (9, 7)) -> str:
        """
        绘制结合能与RMSD的散点图
        
        参数:
            df: 包含对接结果的DataFrame
            affinity_col: 结合能列名
            rmsd_col: RMSD列名
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        if affinity_col not in df.columns or rmsd_col not in df.columns:
            logger.error("Required columns not found")
            return ""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        df_clean = df[[affinity_col, rmsd_col]].dropna()
        
        # 根据结合能着色
        scatter = ax.scatter(df_clean[rmsd_col], df_clean[affinity_col],
                           c=df_clean[affinity_col], cmap='RdYlGn_r',
                           s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Binding Affinity (kcal/mol)', fontweight='bold')
        
        ax.set_xlabel('RMSD (Å)', fontweight='bold')
        ax.set_ylabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        try:
            z = np.polyfit(df_clean[rmsd_col], df_clean[affinity_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_clean[rmsd_col].min(), df_clean[rmsd_col].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.6, label='Trend')
            ax.legend()
        except Exception:
            pass
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Affinity vs RMSD saved to {output_path}")
        return str(output_path)
    
    def plot_docking_summary(self,
                            df: pd.DataFrame,
                            affinity_col: str = "best_affinity",
                            title: str = "Molecular Docking Summary",
                            filename: str = "docking_summary.png",
                            figsize: Tuple[int, int] = (14, 10)) -> str:
        """
        绘制对接结果综合摘要图（多子图）
        
        参数:
            df: 包含对接结果的DataFrame
            affinity_col: 结合能列名
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        affinities = df[affinity_col].dropna()
        
        # 1. 结合能分布直方图
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(affinities, bins=50, color=COLORS['palette'][0], 
                alpha=0.7, edgecolor='white', linewidth=1)
        ax1.axvline(x=affinities.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {affinities.mean():.2f}')
        ax1.axvline(x=affinities.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {affinities.median():.2f}')
        ax1.set_xlabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Binding Affinity Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 统计信息文本
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats_text = f"""
        Docking Statistics
        
        Total Compounds: {len(df)}
        Successful: {len(affinities)}
        
        Binding Affinity:
        Mean: {affinities.mean():.2f}
        Median: {affinities.median():.2f}
        Std: {affinities.std():.2f}
        Min: {affinities.min():.2f}
        Max: {affinities.max():.2f}
        
        Strong Binders (≤ -9.0):
        {len(affinities[affinities <= -9.0])} ({len(affinities[affinities <= -9.0])/len(affinities)*100:.1f}%)
        
        Moderate Binders (-9.0 to -7.0):
        {len(affinities[(affinities > -9.0) & (affinities <= -7.0)])} 
        ({len(affinities[(affinities > -9.0) & (affinities <= -7.0)])/len(affinities)*100:.1f}%)
        """
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 3. 结合能分类饼图
        ax3 = fig.add_subplot(gs[1, 0])
        categories = {
            'Strong': len(affinities[affinities <= -9.0]),
            'Moderate': len(affinities[(affinities > -9.0) & (affinities <= -7.0)]),
            'Weak': len(affinities[(affinities > -7.0) & (affinities <= -5.0)]),
            'Negligible': len(affinities[affinities > -5.0])
        }
        colors_pie = [COLORS['strong'], COLORS['moderate'], COLORS['weak'], COLORS['negligible']]
        ax3.pie(categories.values(), labels=categories.keys(), colors=colors_pie,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
        ax3.set_title('Binding Affinity Categories', fontweight='bold')
        
        # 4. 箱线图
        ax4 = fig.add_subplot(gs[1, 1:])
        bp = ax4.boxplot([affinities], vert=False, patch_artist=True,
                        widths=0.5, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        bp['boxes'][0].set_facecolor(COLORS['palette'][0])
        bp['boxes'][0].set_alpha(0.7)
        ax4.set_xlabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax4.set_title('Box Plot', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Top 10化合物
        ax5 = fig.add_subplot(gs[2, :])
        df_top = df.nsmallest(10, affinity_col)
        names = df_top[name_col if name_col in df.columns else df.columns[0]].astype(str).str[:15]
        vals = df_top[affinity_col]
        
        colors_bar = [COLORS['strong'] if v <= -9.0 else 
                     COLORS['moderate'] if v <= -7.0 else 
                     COLORS['weak'] if v <= -5.0 else COLORS['negligible'] 
                     for v in vals]
        
        y_pos = np.arange(len(names))
        bars = ax5.barh(y_pos, vals, color=colors_bar, edgecolor='white', linewidth=1.5)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(names, fontsize=8)
        ax5.invert_yaxis()
        ax5.set_xlabel('Binding Affinity (kcal/mol)', fontweight='bold')
        ax5.set_title('Top 10 Compounds', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, vals):
            width = bar.get_width()
            ax5.text(width - 0.2, bar.get_y() + bar.get_height()/2.,
                    f'{val:.2f}', ha='right', va='center',
                    fontsize=8, fontweight='bold', color='white')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Docking summary saved to {output_path}")
        return str(output_path)
