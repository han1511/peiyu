#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMET性质可视化模块

功能：
1. ADMET雷达图（综合评估）
2. 各类性质分布图
3. 口服生物利用度对比
4. 毒性风险热力图
5. 类药性评分分布
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
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# 配色
COLORS = {
    'good': '#27AE60',
    'moderate': '#F39C12',
    'poor': '#C0392B',
    'palette': ['#2E5C8A', '#E67E22', '#27AE60', '#8E44AD', '#C0392B']
}


class ADMETVisualizer:
    """
    ADMET性质可视化类
    
    生成论文级别的ADMET评估图表
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化ADMET可视化器
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir or Path("results/visualization/admet")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.rcParams.update({
                'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                'xtick.labelsize': 10, 'ytick.labelsize': 10,
                'figure.dpi': 300, 'savefig.dpi': 300,
                'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2
            })
    
    def _categorical_to_score(self, value: str) -> float:
        """将分类值转换为数值分数用于雷达图"""
        mapping = {
            'High': 1.0, 'High (>50%)': 1.0, 'High BBB Penetration': 1.0,
            'Medium': 0.6, 'Medium (10-50%)': 0.6, 'Moderate BBB Penetration': 0.6,
            'Low': 0.2, 'Low (<10%)': 0.2, 'Low BBB Penetration': 0.2,
            'Low Risk': 1.0, 'Medium Risk': 0.5, 'High Risk': 0.0,
            'Non-Mutagenic': 1.0, 'Potentially Mutagenic': 0.5, 'Mutagenic': 0.0,
            'High Solubility': 1.0, 'Medium Solubility': 0.6, 'Low Solubility': 0.2,
            'Drug-like: Good ADMET profile': 1.0,
            'Drug-like: Minor concerns noted': 0.8,
            'Borderline: May require optimization': 0.5,
            'Non-drug-like: Significant ADMET concerns': 0.0
        }
        return mapping.get(str(value), 0.5)
    
    def plot_admet_radar(self,
                        admet_data: Dict[str, Any],
                        title: str = "ADMET Profile",
                        filename: str = "admet_radar.png",
                        figsize: Tuple[int, int] = (10, 10)) -> str:
        """
        绘制ADMET雷达图
        
        参数:
            admet_data: ADMET结果字典
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        # 提取关键指标
        categories = []
        values = []
        
        absorption = admet_data.get('absorption', {})
        metabolism = admet_data.get('metabolism', {})
        toxicity = admet_data.get('toxicity', {})
        solubility = admet_data.get('solubility', {})
        bbb = admet_data.get('bbb_penetration', {})
        
        # 吸收指标
        if absorption:
            categories.extend(['Intestinal Absorption', 'Oral Bioavailability', 
                             'Caco-2 Permeability'])
            values.extend([
                self._categorical_to_score(absorption.get('human_intestinal_absorption', 'Medium')),
                self._categorical_to_score(absorption.get('oral_bioavailability', 'Medium')),
                min(absorption.get('caco2_permeability', 5) / 8, 1.0)
            ])
        
        # 代谢指标
        if metabolism:
            categories.append('CYP3A4 Safety')
            cyp3a4 = metabolism.get('cyp3a4_inhibition', {})
            values.append(self._categorical_to_score(cyp3a4.get('prediction', 'Medium Risk')))
        
        # 毒性指标
        if toxicity:
            categories.extend(['AMES Safety', 'hERG Safety'])
            ames = toxicity.get('ames_toxicity', {})
            herg = toxicity.get('herg_inhibition', {})
            values.extend([
                self._categorical_to_score(ames.get('prediction', 'Potentially Mutagenic')),
                self._categorical_to_score(herg.get('prediction', 'Medium Risk'))
            ])
        
        # 溶解度
        if solubility:
            categories.append('Solubility')
            values.append(self._categorical_to_score(solubility.get('prediction', 'Medium Solubility')))
        
        # BBB
        if bbb:
            categories.append('BBB Permeability')
            values.append(self._categorical_to_score(bbb.get('prediction', 'Moderate BBB Penetration')))
        
        if not categories:
            logger.error("No ADMET data available for radar chart")
            return ""
        
        # 绘制雷达图
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2.5, color=COLORS['palette'][0])
        ax.fill(angles, values, alpha=0.25, color=COLORS['palette'][0])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['Poor', '', 'Moderate', '', 'Good'], fontsize=8)
        ax.grid(True)
        
        # 添加颜色区域
        ax.axhspan(0, 0.4, alpha=0.1, color='red')
        ax.axhspan(0.4, 0.7, alpha=0.1, color='yellow')
        ax.axhspan(0.7, 1.0, alpha=0.1, color='green')
        
        ax.set_title(title, fontweight='bold', pad=30, fontsize=14)
        
        # 计算综合评分
        overall_score = np.mean(values[:-1])
        score_text = f'Overall Score: {overall_score:.2f}'
        if overall_score >= 0.7:
            score_color = COLORS['good']
            score_label = 'Drug-like'
        elif overall_score >= 0.4:
            score_color = COLORS['moderate']
            score_label = 'Borderline'
        else:
            score_color = COLORS['poor']
            score_label = 'Non-drug-like'
        
        fig.text(0.5, 0.02, f'{score_text} ({score_label})', ha='center',
                fontsize=12, fontweight='bold', color=score_color)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ADMET radar saved to {output_path}")
        return str(output_path)
    
    def plot_property_distribution(self,
                                  df: pd.DataFrame,
                                  properties: List[str] = None,
                                  title: str = "ADMET Property Distribution",
                                  filename: str = "admet_distribution.png",
                                  figsize: Tuple[int, int] = (14, 10)) -> str:
        """
        绘制ADMET性质分布图（多个子图）
        
        参数:
            df: ADMET结果DataFrame
            properties: 要绘制的性质列名列表
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        if properties is None:
            properties = ['hia', 'oral_bioavailability', 'cyp3a4_inhibition',
                         'ames_toxicity', 'herg_inhibition', 'solubility']
            properties = [p for p in properties if p in df.columns]
        
        n_props = len(properties)
        if n_props == 0:
            logger.error("No valid properties to plot")
            return ""
        
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        colors = COLORS['palette']
        
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            
            value_counts = df[prop].value_counts()
            
            # 根据性质类型选择颜色
            bar_colors = []
            for val in value_counts.index:
                val_str = str(val).lower()
                if any(x in val_str for x in ['high', 'good', 'non-mutagenic', 'low risk']):
                    bar_colors.append(COLORS['good'])
                elif any(x in val_str for x in ['medium', 'moderate', 'potential']):
                    bar_colors.append(COLORS['moderate'])
                else:
                    bar_colors.append(COLORS['poor'])
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.85)
            
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(prop.replace('_', ' ').title(), fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, val in zip(bars, value_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 隐藏多余子图
        for idx in range(n_props, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Property distribution saved to {output_path}")
        return str(output_path)
    
    def plot_drug_likeness_heatmap(self,
                                  df: pd.DataFrame,
                                  title: str = "Drug-likeness Assessment",
                                  filename: str = "drug_likeness_heatmap.png",
                                  figsize: Tuple[int, int] = (12, 10)) -> str:
        """
        绘制类药性评估热力图
        
        参数:
            df: ADMET结果DataFrame（行：化合物，列：性质）
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        # 选择关键列并转换为数值
        key_properties = ['hia', 'oral_bioavailability', 'cyp3a4_inhibition',
                         'ames_toxicity', 'herg_inhibition', 'solubility', 'bbb_penetration']
        available_props = [p for p in key_properties if p in df.columns]
        
        if len(available_props) == 0:
            logger.error("No ADMET properties found")
            return ""
        
        # 取前20个化合物
        df_subset = df.head(20)
        
        # 转换为数值矩阵
        score_matrix = []
        compound_names = []
        
        for idx, row in df_subset.iterrows():
            scores = []
            for prop in available_props:
                scores.append(self._categorical_to_score(row.get(prop, 'Medium')))
            score_matrix.append(scores)
            name = str(row.get('compound_name', row.get('SMILES', f'Comp_{idx}') ))[:15]
            compound_names.append(name)
        
        score_matrix = np.array(score_matrix)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        cmap = sns.color_palette("RdYlGn", as_cmap=True)
        sns.heatmap(score_matrix, cmap=cmap, vmin=0, vmax=1,
                   xticklabels=[p.replace('_', ' ').title() for p in available_props],
                   yticklabels=compound_names,
                   annot=True, fmt='.1f', linewidths=0.5,
                   cbar_kws={'label': 'Score (0=Poor, 1=Good)'},
                   ax=ax)
        
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel('ADMET Properties', fontweight='bold')
        ax.set_ylabel('Compounds', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Drug-likeness heatmap saved to {output_path}")
        return str(output_path)
    
    def plot_lipinski_violin(self,
                            df: pd.DataFrame,
                            mw_col: str = "Molecular_Weight",
                            logp_col: str = "LogP",
                            hbd_col: str = "H_Donors",
                            hba_col: str = "H_Acceptors",
                            title: str = "Lipinski Properties Distribution",
                            filename: str = "lipinski_violin.png",
                            figsize: Tuple[int, int] = (14, 8)) -> str:
        """
        绘制Lipinski性质小提琴图
        
        参数:
            df: 化合物性质DataFrame
            mw_col: 分子量列名
            logp_col: LogP列名
            hbd_col: HBD列名
            hba_col: HBA列名
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        cols = [c for c in [mw_col, logp_col, hbd_col, hba_col] if c in df.columns]
        if len(cols) == 0:
            logger.error("No Lipinski properties found")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        thresholds = {
            mw_col: 500,
            logp_col: 5,
            hbd_col: 5,
            hba_col: 10
        }
        
        for idx, col in enumerate(cols):
            ax = axes[idx]
            
            data = df[col].dropna()
            
            # 小提琴图
            parts = ax.violinplot([data], positions=[1], showmeans=True, showmedians=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS['palette'][idx])
                pc.set_alpha(0.7)
            
            # 散点
            y_jittered = data + np.random.normal(0, 0.02, len(data))
            ax.scatter(y_jittered, [1] * len(data), alpha=0.3, s=20, color='black')
            
            # 阈值线
            if col in thresholds:
                ax.axvline(x=thresholds[col], color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Limit: {thresholds[col]}')
                ax.legend()
            
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xlabel(col.replace('_', ' '), fontweight='bold')
            ax.set_title(col.replace('_', ' '), fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 隐藏多余子图
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Lipinski violin plot saved to {output_path}")
        return str(output_path)
