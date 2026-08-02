#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化学空间可视化模块

功能：
1. UMAP降维可视化
2. t-SNE降维可视化
3. PCA降维可视化
4. 化学空间覆盖分析
5. 训练集/测试集/筛选集分布对比
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

COLORS = {
    'train': '#2E5C8A',
    'test': '#E67E22', 
    'validation': '#27AE60',
    'screening': '#8E44AD',
    'active': '#C0392B',
    'inactive': '#BDC3C7',
    'palette': ['#2E5C8A', '#E67E22', '#27AE60', '#8E44AD', '#C0392B', 
                '#16A085', '#D35400', '#2980B9', '#F39C12', '#7F8C8D']
}


class ChemicalSpaceVisualizer:
    """
    化学空间可视化类
    
    使用降维技术展示化合物在化学空间中的分布
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化化学空间可视化器
        
        参数:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir or Path("results/visualization/chemical_space")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.rcParams.update({
                'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                'xtick.labelsize': 10, 'ytick.labelsize': 10,
                'figure.dpi': 300, 'savefig.dpi': 300,
                'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2
            })
    
    def _reduce_dimension(self,
                         features: np.ndarray,
                         method: str = 'umap',
                         n_components: int = 2,
                         random_state: int = 42,
                         **kwargs) -> np.ndarray:
        """
        降维内部函数
        
        参数:
            features: 特征矩阵
            method: 降维方法 ('umap', 'tsne', 'pca')
            n_components: 降维后维度
            random_state: 随机种子
            
        返回:
            np.ndarray: 降维后的坐标
        """
        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(features)
        
        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, features.shape[0] // 10))
            reducer = TSNE(n_components=n_components, perplexity=perplexity,
                          random_state=random_state, n_iter=1000)
            return reducer.fit_transform(features)
        
        elif method.lower() == 'umap':
            try:
                import umap
                n_neighbors = min(15, max(5, features.shape[0] // 20))
                reducer = umap.UMAP(n_components=n_components,
                                   n_neighbors=n_neighbors,
                                   random_state=random_state,
                                   min_dist=0.1)
                return reducer.fit_transform(features)
            except ImportError:
                logger.warning("UMAP not available, falling back to t-SNE")
                return self._reduce_dimension(features, 'tsne', n_components, random_state)
        
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def plot_chemical_space(self,
                           features_dict: Dict[str, np.ndarray],
                           labels_dict: Optional[Dict[str, np.ndarray]] = None,
                           method: str = 'umap',
                           title: str = "Chemical Space Visualization",
                           filename: str = "chemical_space.png",
                           figsize: Tuple[int, int] = (12, 10),
                           alpha: float = 0.6,
                           point_size: int = 30) -> str:
        """
        绘制化学空间分布图
        
        参数:
            features_dict: {数据集名称: 特征矩阵}
            labels_dict: {数据集名称: 标签数组} (用于着色)
            method: 降维方法
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            alpha: 透明度
            point_size: 点大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        # 合并所有特征进行统一降维
        all_features = np.vstack(list(features_dict.values()))
        all_coords = self._reduce_dimension(all_features, method=method)
        
        # 分离各数据集坐标
        coords_dict = {}
        start_idx = 0
        for name, features in features_dict.items():
            end_idx = start_idx + len(features)
            coords_dict[name] = all_coords[start_idx:end_idx]
            start_idx = end_idx
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：按数据集着色
        ax1 = axes[0]
        colors = COLORS['palette']
        
        for idx, (name, coords) in enumerate(coords_dict.items()):
            color = colors[idx % len(colors)]
            ax1.scatter(coords[:, 0], coords[:, 1], c=color,
                       label=f'{name} (n={len(coords)})',
                       alpha=alpha, s=point_size, edgecolors='white', linewidth=0.3)
        
        ax1.set_xlabel(f'{method.upper()} 1', fontweight='bold')
        ax1.set_ylabel(f'{method.upper()} 2', fontweight='bold')
        ax1.set_title(f'{title} - By Dataset', fontweight='bold')
        ax1.legend(loc='best', frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        
        # 右图：按标签着色（如果提供）
        ax2 = axes[1]
        
        if labels_dict:
            # 合并所有标签
            all_labels = np.hstack(list(labels_dict.values()))
            
            active_mask = all_labels == 1
            inactive_mask = all_labels == 0
            
            if np.any(active_mask):
                ax2.scatter(all_coords[active_mask, 0], all_coords[active_mask, 1],
                           c=COLORS['active'], label=f'Active (n={np.sum(active_mask)})',
                           alpha=alpha, s=point_size, edgecolors='white', linewidth=0.3)
            
            if np.any(inactive_mask):
                ax2.scatter(all_coords[inactive_mask, 0], all_coords[inactive_mask, 1],
                           c=COLORS['inactive'], label=f'Inactive (n={np.sum(inactive_mask)})',
                           alpha=alpha, s=point_size, edgecolors='white', linewidth=0.3)
            
            ax2.set_title(f'{title} - By Activity', fontweight='bold')
        else:
            ax2.scatter(all_coords[:, 0], all_coords[:, 1],
                       c=COLORS['palette'][0], alpha=alpha, s=point_size,
                       edgecolors='white', linewidth=0.3)
            ax2.set_title(f'{title} - All Compounds', fontweight='bold')
        
        ax2.set_xlabel(f'{method.upper()} 1', fontweight='bold')
        ax2.set_ylabel(f'{method.upper()} 2', fontweight='bold')
        ax2.legend(loc='best', frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Chemical space plot saved to {output_path}")
        return str(output_path)
    
    def plot_applicability_domain(self,
                                 train_features: np.ndarray,
                                 test_features: np.ndarray,
                                 train_labels: Optional[np.ndarray] = None,
                                 method: str = 'umap',
                                 title: str = "Applicability Domain Analysis",
                                 filename: str = "applicability_domain.png",
                                 figsize: Tuple[int, int] = (12, 10)) -> str:
        """
        绘制适用域分析图
        
        参数:
            train_features: 训练集特征
            test_features: 测试/筛选集特征
            train_labels: 训练集标签
            method: 降维方法
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        # 统一降维
        all_features = np.vstack([train_features, test_features])
        all_coords = self._reduce_dimension(all_features, method=method)
        
        train_coords = all_coords[:len(train_features)]
        test_coords = all_coords[len(train_features):]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：训练集和测试集分布
        ax1 = axes[0]
        
        ax1.scatter(train_coords[:, 0], train_coords[:, 1],
                   c=COLORS['train'], label=f'Training Set (n={len(train_coords)})',
                   alpha=0.6, s=40, edgecolors='white', linewidth=0.3)
        
        ax1.scatter(test_coords[:, 0], test_coords[:, 1],
                   c=COLORS['screening'], label=f'Screening Set (n={len(test_coords)})',
                   alpha=0.6, s=40, edgecolors='white', linewidth=0.3,
                   marker='^')
        
        # 计算并绘制训练集凸包
        try:
            from scipy.spatial import ConvexHull
            if len(train_coords) > 3:
                hull = ConvexHull(train_coords)
                for simplex in hull.simplices:
                    ax1.plot(train_coords[simplex, 0], train_coords[simplex, 1],
                            'b-', alpha=0.3, linewidth=1)
        except Exception:
            pass
        
        ax1.set_xlabel(f'{method.upper()} 1', fontweight='bold')
        ax1.set_ylabel(f'{method.upper()} 2', fontweight='bold')
        ax1.set_title('Training vs Screening Set', fontweight='bold')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # 右图：训练集按活性着色
        ax2 = axes[1]
        
        if train_labels is not None:
            active_mask = train_labels == 1
            inactive_mask = train_labels == 0
            
            ax2.scatter(train_coords[inactive_mask, 0], train_coords[inactive_mask, 1],
                       c=COLORS['inactive'], label=f'Inactive (n={np.sum(inactive_mask)})',
                       alpha=0.6, s=40, edgecolors='white', linewidth=0.3)
            
            ax2.scatter(train_coords[active_mask, 0], train_coords[active_mask, 1],
                       c=COLORS['active'], label=f'Active (n={np.sum(active_mask)})',
                       alpha=0.8, s=50, edgecolors='white', linewidth=0.3)
            
            ax2.set_title('Training Set by Activity', fontweight='bold')
        else:
            ax2.scatter(train_coords[:, 0], train_coords[:, 1],
                       c=COLORS['train'], alpha=0.6, s=40,
                       edgecolors='white', linewidth=0.3)
            ax2.set_title('Training Set', fontweight='bold')
        
        ax2.set_xlabel(f'{method.upper()} 1', fontweight='bold')
        ax2.set_ylabel(f'{method.upper()} 2', fontweight='bold')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Applicability domain plot saved to {output_path}")
        return str(output_path)
    
    def plot_scree_plot(self,
                       features: np.ndarray,
                       title: str = "PCA Scree Plot",
                       filename: str = "scree_plot.png",
                       figsize: Tuple[int, int] = (10, 7)) -> str:
        """
        绘制PCA方差解释率图
        
        参数:
            features: 特征矩阵
            title: 图表标题
            filename: 输出文件名
            figsize: 图表大小
            
        返回:
            str: 输出文件路径
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available")
            return ""
        
        from sklearn.decomposition import PCA
        
        n_components = min(20, features.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(features)
        
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(1, len(explained_variance) + 1)
        
        bars = ax1.bar(x_pos, explained_variance, color=COLORS['palette'][0],
                      alpha=0.7, edgecolor='white', linewidth=1)
        ax1.set_xlabel('Principal Component', fontweight='bold')
        ax1.set_ylabel('Explained Variance (%)', fontweight='bold', color=COLORS['palette'][0])
        ax1.tick_params(axis='y', labelcolor=COLORS['palette'][0])
        
        ax2 = ax1.twinx()
        ax2.plot(x_pos, cumulative_variance, 'ro-', linewidth=2.5, markersize=6)
        ax2.set_ylabel('Cumulative Variance (%)', fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.axhline(y=90, color='blue', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.legend(loc='center right')
        
        ax1.set_title(title, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 标注前3个PC的方差
        for i in range(min(3, len(explained_variance))):
            ax1.text(x_pos[i], explained_variance[i] + 0.5,
                    f'{explained_variance[i]:.1f}%',
                    ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scree plot saved to {output_path}")
        return str(output_path)
