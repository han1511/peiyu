#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量控制与适用域检查模块

功能：
1. 化合物数据质量检查
2. 异常值检测与处理
3. 适用域分析 (AD: Applicability Domain)
4. 数据完整性验证
5. 化学有效性检查
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit未安装，部分化学质量控制功能不可用")


@dataclass
class QCResult:
    """质量控制结果"""
    passed: bool
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, Any]
    filtered_indices: List[int]
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.stats is None:
            self.stats = {}
        if self.filtered_indices is None:
            self.filtered_indices = []


class DataQualityControl:
    """
    数据质量控制类
    
    提供全面的化合物数据质量检查功能
    """
    
    def __init__(self,
                 max_missing_ratio: float = 0.1,
                 outlier_method: str = "iqr",
                 outlier_threshold: float = 1.5,
                 enable_chemistry_check: bool = True):
        """
        初始化数据质量控制
        
        参数:
            max_missing_ratio: 最大允许缺失值比例
            outlier_method: 异常值检测方法 ('iqr', 'zscore', 'mad')
            outlier_threshold: 异常值阈值
            enable_chemistry_check: 是否启用化学有效性检查
        """
        self.max_missing_ratio = max_missing_ratio
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.enable_chemistry_check = enable_chemistry_check and HAS_RDKIT
        
    def validate_compound_library(self, 
                                   df: pd.DataFrame,
                                   smiles_column: str = "SMILES") -> QCResult:
        """
        验证化合物库数据质量
        
        参数:
            df: 化合物数据框
            smiles_column: SMILES列名
            
        返回:
            QCResult: 质量控制结果
        """
        warnings = []
        errors = []
        stats = {}
        filtered_indices = []
        
        # 1. 基本检查
        if df.empty:
            errors.append("化合物库为空")
            return QCResult(False, warnings, errors, stats, filtered_indices)
        
        # 2. 检查必要的列
        if smiles_column not in df.columns:
            errors.append(f"缺少必要的列: {smiles_column}")
            return QCResult(False, warnings, errors, stats, filtered_indices)
        
        # 3. 检查缺失值
        missing_count = df[smiles_column].isna().sum()
        missing_ratio = missing_count / len(df)
        stats['missing_smiles_count'] = missing_count
        stats['missing_smiles_ratio'] = missing_ratio
        
        if missing_ratio > self.max_missing_ratio:
            errors.append(f"SMILES缺失值比例过高: {missing_ratio:.2%} (阈值: {self.max_missing_ratio:.2%})")
        elif missing_count > 0:
            warnings.append(f"发现 {missing_count} 个缺失SMILES")
            filtered_indices.extend(df[df[smiles_column].isna()].index.tolist())
        
        # 4. 检查重复
        duplicates = df[smiles_column].duplicated().sum()
        stats['duplicate_count'] = duplicates
        if duplicates > 0:
            warnings.append(f"发现 {duplicates} 个重复SMILES")
        
        # 5. 化学有效性检查
        if self.enable_chemistry_check:
            valid_smiles = []
            invalid_count = 0
            
            for idx, smiles in df[smiles_column].items():
                if pd.isna(smiles):
                    continue
                    
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    invalid_count += 1
                    filtered_indices.append(idx)
                else:
                    valid_smiles.append(idx)
            
            stats['invalid_smiles_count'] = invalid_count
            stats['valid_smiles_count'] = len(valid_smiles)
            
            if invalid_count > 0:
                invalid_ratio = invalid_count / len(df)
                if invalid_ratio > 0.05:  # 5%阈值
                    errors.append(f"无效SMILES比例过高: {invalid_ratio:.2%}")
                else:
                    warnings.append(f"发现 {invalid_count} 个无效SMILES")
        
        # 6. 化合物大小检查
        if self.enable_chemistry_check:
            heavy_atom_counts = []
            for smiles in df[smiles_column].dropna():
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    heavy_atom_counts.append(mol.GetNumHeavyAtoms())
            
            if heavy_atom_counts:
                stats['avg_heavy_atoms'] = np.mean(heavy_atom_counts)
                stats['max_heavy_atoms'] = np.max(heavy_atom_counts)
                
                too_large = sum(1 for c in heavy_atom_counts if c > 100)
                if too_large > 0:
                    warnings.append(f"发现 {too_large} 个超大分子 (>100重原子)")
        
        # 判断最终结果
        passed = len(errors) == 0
        
        stats['total_compounds'] = len(df)
        stats['compounds_after_filter'] = len(df) - len(set(filtered_indices))
        
        return QCResult(passed, warnings, errors, stats, list(set(filtered_indices)))
    
    def detect_outliers(self, 
                       features: np.ndarray,
                       compound_ids: Optional[List] = None) -> Tuple[np.ndarray, Dict]:
        """
        检测特征异常值
        
        参数:
            features: 特征矩阵
            compound_ids: 化合物ID列表
            
        返回:
            Tuple[np.ndarray, Dict]: (异常值掩码, 统计信息)
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        n_samples = len(features)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        stats = {'method': self.outlier_method, 'threshold': self.outlier_threshold}
        
        if self.outlier_method == "iqr":
            # IQR方法
            for i in range(features.shape[1]):
                col = features[:, i]
                Q1 = np.percentile(col, 25)
                Q3 = np.percentile(col, 75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR
                
                feature_outliers = (col < lower) | (col > upper)
                outlier_mask |= feature_outliers
                
                stats[f'feature_{i}_outliers'] = int(feature_outliers.sum())
        
        elif self.outlier_method == "zscore":
            # Z-score方法
            z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
            outlier_mask = (z_scores > self.outlier_threshold).any(axis=1)
            
        elif self.outlier_method == "mad":
            # MAD方法 (Median Absolute Deviation)
            median = np.median(features, axis=0)
            mad = np.median(np.abs(features - median), axis=0)
            modified_z = 0.6745 * (features - median) / mad
            outlier_mask = (np.abs(modified_z) > self.outlier_threshold).any(axis=1)
        
        stats['total_outliers'] = int(outlier_mask.sum())
        stats['outlier_ratio'] = outlier_mask.sum() / n_samples
        
        if compound_ids is not None:
            outlier_ids = [compound_ids[i] for i in range(n_samples) if outlier_mask[i]]
            stats['outlier_ids'] = outlier_ids[:20]  # 最多保存20个
        
        return outlier_mask, stats
    
    def filter_compounds(self,
                        df: pd.DataFrame,
                        qc_result: QCResult,
                        outlier_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        根据质量控制结果过滤化合物
        
        参数:
            df: 原始数据框
            qc_result: 质量控制结果
            outlier_mask: 异常值掩码
            
        返回:
            pd.DataFrame: 过滤后的数据框
        """
        # 移除无效化合物
        valid_indices = [i for i in range(len(df)) if i not in qc_result.filtered_indices]
        filtered_df = df.iloc[valid_indices].copy()
        
        # 移除异常值
        if outlier_mask is not None and len(outlier_mask) == len(df):
            valid_mask = ~outlier_mask[valid_indices]
            filtered_df = filtered_df[valid_mask].copy()
        
        logger.info(f"数据过滤: {len(df)} -> {len(filtered_df)} 化合物")
        return filtered_df.reset_index(drop=True)


class ApplicabilityDomain:
    """
    适用域分析类
    
    评估测试化合物是否在训练模型的适用域内
    """
    
    def __init__(self,
                 method: str = "leverage",
                 threshold: float = 0.95,
                 k_neighbors: int = 5):
        """
        初始化适用域分析
        
        参数:
            method: 方法 ('leverage', 'knn', 'kernel_density')
            threshold: 阈值
            k_neighbors: k近邻数量
        """
        self.method = method
        self.threshold = threshold
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()
        self.fitted = False
        
        # 训练集参数
        self.train_features = None
        self.train_mean = None
        self.train_cov = None
        self.train_knn = None
        self.h_star = None
    
    def fit(self, train_features: np.ndarray):
        """
        使用训练集拟合适用域
        
        参数:
            train_features: 训练集特征矩阵
        """
        if train_features.ndim == 1:
            train_features = train_features.reshape(-1, 1)
        
        self.train_features = self.scaler.fit_transform(train_features)
        
        if self.method == "leverage":
            # Leverage方法: h = x(X'X)^(-1)x'
            n, p = self.train_features.shape
            self.h_star = 3 * (p + 1) / n  # 阈值
            
        elif self.method == "knn":
            # kNN方法
            self.train_knn = NearestNeighbors(n_neighbors=self.k_neighbors)
            self.train_knn.fit(self.train_features)
            
            # 计算训练集的平均k近邻距离作为阈值
            distances, _ = self.train_knn.kneighbors(self.train_features)
            self.threshold = np.mean(distances[:, -1])
            
        elif self.method == "kernel_density":
            # 核密度估计方法 (简化为基于均值的马氏距离)
            self.train_mean = np.mean(self.train_features, axis=0)
            cov = np.cov(self.train_features.T)
            # 添加正则化避免奇异矩阵
            self.train_cov = cov + np.eye(cov.shape[0]) * 1e-6
        
        self.fitted = True
        logger.info(f"适用域已拟合: method={self.method}, threshold={self.threshold:.4f}")
    
    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测测试样本是否在适用域内
        
        参数:
            test_features: 测试集特征矩阵
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (是否在域内, 距离/统计量)
        """
        if not self.fitted:
            raise ValueError("适用域未拟合，请先调用fit()")
        
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)
        
        test_features = self.scaler.transform(test_features)
        n_test = len(test_features)
        
        if self.method == "leverage":
            # 计算leverage
            XT_X_inv = np.linalg.pinv(self.train_features.T @ self.train_features)
            leverage = np.array([
                x @ XT_X_inv @ x.T for x in test_features
            ])
            in_domain = leverage < self.h_star
            return in_domain, leverage
        
        elif self.method == "knn":
            # 计算k近邻平均距离
            distances, _ = self.train_knn.kneighbors(test_features)
            mean_distances = np.mean(distances, axis=1)
            in_domain = mean_distances < self.threshold
            return in_domain, mean_distances
        
        elif self.method == "kernel_density":
            # 简化的马氏距离
            diff = test_features - self.train_mean
            inv_cov = np.linalg.inv(self.train_cov)
            mahalanobis = np.array([
                np.sqrt(d @ inv_cov @ d) for d in diff
            ])
            # 使用卡方分布的95%分位数作为阈值
            from scipy.stats import chi2
            chi2_threshold = chi2.ppf(self.threshold, df=test_features.shape[1])
            in_domain = mahalanobis < np.sqrt(chi2_threshold)
            return in_domain, mahalanobis
        
        else:
            raise ValueError(f"未知方法: {self.method}")
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """
        获取适用域统计信息
        
        返回:
            Dict: 统计信息
        """
        if not self.fitted:
            return {"error": "适用域未拟合"}
        
        stats = {
            "method": self.method,
            "threshold": float(self.threshold),
            "train_samples": len(self.train_features) if self.train_features is not None else 0,
            "n_features": self.train_features.shape[1] if self.train_features is not None else 0
        }
        
        if self.method == "leverage":
            stats["h_star"] = float(self.h_star)
        
        return stats


class ScreeningValidator:
    """
    筛选验证器
    
    整合数据质量控制和适用域分析
    """
    
    def __init__(self,
                 qc_config: Optional[Dict] = None,
                 ad_config: Optional[Dict] = None):
        """
        初始化筛选验证器
        
        参数:
            qc_config: 质量控制配置
            ad_config: 适用域配置
        """
        qc_config = qc_config or {}
        ad_config = ad_config or {}
        
        self.qc = DataQualityControl(**qc_config)
        self.ad = ApplicabilityDomain(**ad_config)
        
    def validate_inputs(self,
                       compound_df: pd.DataFrame,
                       train_features: Optional[np.ndarray] = None,
                       test_features: Optional[np.ndarray] = None,
                       smiles_column: str = "SMILES") -> Dict[str, Any]:
        """
        全面验证输入数据
        
        参数:
            compound_df: 化合物数据框
            train_features: 训练特征
            test_features: 测试特征
            smiles_column: SMILES列名
            
        返回:
            Dict: 验证结果
        """
        results = {
            'qc_passed': False,
            'ad_passed': False,
            'warnings': [],
            'errors': [],
            'stats': {},
            'recommendations': []
        }
        
        # 1. 数据质量控制
        logger.info("执行数据质量控制检查...")
        qc_result = self.qc.validate_compound_library(compound_df, smiles_column)
        
        results['qc_passed'] = qc_result.passed
        results['warnings'].extend(qc_result.warnings)
        results['errors'].extend(qc_result.errors)
        results['stats']['qc'] = qc_result.stats
        
        if not qc_result.passed:
            results['recommendations'].append("请修复数据质量问题后再进行筛选")
        
        # 2. 适用域检查
        if train_features is not None and test_features is not None:
            logger.info("执行适用域分析...")
            try:
                self.ad.fit(train_features)
                in_domain, distances = self.ad.predict(test_features)
                
                in_domain_ratio = in_domain.mean()
                results['ad_passed'] = in_domain_ratio >= 0.8  # 80%在域内
                results['stats']['ad'] = {
                    'in_domain_ratio': float(in_domain_ratio),
                    'out_of_domain_count': int((~in_domain).sum()),
                    'method': self.ad.method
                }
                
                if in_domain_ratio < 0.8:
                    results['warnings'].append(
                        f"仅 {in_domain_ratio:.1%} 的化合物在模型适用域内，"
                        f"建议扩展训练集或使用更保守的预测"
                    )
                    
            except Exception as e:
                results['errors'].append(f"适用域分析失败: {str(e)}")
        else:
            results['ad_passed'] = True  # 无测试数据，跳过
        
        # 总体评估
        results['overall_passed'] = results['qc_passed'] and results['ad_passed']
        
        return results
    
    def get_validation_summary(self, results: Dict[str, Any]) -> str:
        """
        获取验证结果摘要
        
        参数:
            results: 验证结果
            
        返回:
            str: 摘要文本
        """
        lines = ["=" * 50]
        lines.append("数据验证报告")
        lines.append("=" * 50)
        
        lines.append(f"\n总体状态: {'通过' if results['overall_passed'] else '未通过'}")
        lines.append(f"QC检查: {'通过' if results['qc_passed'] else '未通过'}")
        lines.append(f"AD检查: {'通过' if results['ad_passed'] else '未通过'}")
        
        if results['errors']:
            lines.append(f"\n错误 ({len(results['errors'])}):")
            for error in results['errors']:
                lines.append(f"  - {error}")
        
        if results['warnings']:
            lines.append(f"\n警告 ({len(results['warnings'])}):")
            for warning in results['warnings']:
                lines.append(f"  - {warning}")
        
        if results['recommendations']:
            lines.append(f"\n建议:")
            for rec in results['recommendations']:
                lines.append(f"  - {rec}")
        
        # 统计信息
        stats = results.get('stats', {})
        if 'qc' in stats:
            lines.append(f"\nQC统计:")
            for key, value in stats['qc'].items():
                lines.append(f"  {key}: {value}")
        
        if 'ad' in stats:
            lines.append(f"\nAD统计:")
            for key, value in stats['ad'].items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
