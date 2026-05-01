#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子特征工程模块

功能：
1. 计算分子指纹（Morgan, MACCS, Topological）
2. 计算RDKit分子描述符
3. 特征标准化和归一化
4. 特征选择和降维
5. 特征统计分析

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, Lipinski, MACCSkeys, PandasTools,
        rdMolDescriptors, GetAdjacencyMatrix
    )
    # 从正确的模块导入这些函数
    from rdkit.Chem import Descriptors
    CalcNumHeavyAtoms = Descriptors.HeavyAtomCount
    CalcNumHeteroatoms = Descriptors.NumHeteroatoms
    CalcNumRotatableBonds = Descriptors.NumRotatableBonds
    CalcNumAromaticRings = Descriptors.NumAromaticRings
    CalcNumSaturatedRings = Descriptors.NumSaturatedRings
    CalcNumAliphaticRings = Descriptors.NumAliphaticRings
    CalcTPSA = Descriptors.TPSA
    CalcMolWt = Descriptors.MolWt
    CalcLogP = Descriptors.MolLogP
    CalcNumHBA = Descriptors.NumHAcceptors
    CalcNumHBD = Descriptors.NumHDonors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit not available. Feature calculation will be limited.")

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif

from configs.config import (
    PROJECT_ROOT, PROCESSED_DATA_DIR, LOG_CONFIG,
    FEATURE_CONFIG, DATA_SPLIT_CONFIG
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class MolecularFingerprints:
    """
    分子指纹计算类

    支持多种指纹类型：
    - Morgan指纹（ECFP, FCFP）
    - MACCS Keys
    - Topological指纹
    """

    def __init__(self, fingerprint_types: List[str] = None):
        """
        初始化分子指纹计算器

        参数:
            fingerprint_types: 要计算的指纹类型列表
        """
        self.fingerprint_types = fingerprint_types or ["Morgan", "MACCS"]
        self.fp_config = FEATURE_CONFIG.get("fingerprints", {})

    def calculate_morgan_fingerprint(self,
                                    mol: Chem.Mol,
                                    radius: int = 2,
                                    n_bits: int = 2048,
                                    use_features: bool = False) -> Optional[np.ndarray]:
        """
        计算Morgan指纹（ECFP）

        参数:
            mol: RDKit分子对象
            radius: 指纹半径
            n_bits: 指纹位数
            use_features: 是否使用功能原子特征

        返回:
            np.ndarray: Morgan指纹向量
        """
        if not HAS_RDKIT or mol is None:
            return None

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=n_bits,
                useFeatures=use_features
            )
            return np.array(fp)
        except Exception as e:
            logger.debug(f"Error calculating Morgan fingerprint: {str(e)}")
            return None

    def calculate_maccs_keys(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """
        计算MACCS Keys

        参数:
            mol: RDKit分子对象

        返回:
            np.ndarray: MACCS Keys向量
        """
        if not HAS_RDKIT or mol is None:
            return None

        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except Exception as e:
            logger.debug(f"Error calculating MACCS keys: {str(e)}")
            return None

    def calculate_topological_fingerprint(self,
                                        mol: Chem.Mol,
                                        n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        计算拓扑指纹

        参数:
            mol: RDKit分子对象
            n_bits: 指纹位数

        返回:
            np.ndarray: 拓扑指纹向量
        """
        if not HAS_RDKIT or mol is None:
            return None

        try:
            from rdkit.Chem import rdFingerprintGenerator
            fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
            fp = fpgen.GetFingerprint(mol, nBits=n_bits)
            return np.array(fp)
        except Exception as e:
            logger.debug(f"Error calculating topological fingerprint: {str(e)}")
            return None

    def calculate_all_fingerprints(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """
        计算所有配置的指纹

        参数:
            mol: RDKit分子对象

        返回:
            dict: 包含各类型指纹的字典
        """
        fingerprints = {}

        for fp_type in self.fingerprint_types:
            if fp_type == "Morgan":
                config = self.fp_config.get("Morgan", {})
                fp = self.calculate_morgan_fingerprint(
                    mol,
                    radius=config.get("radius", 2),
                    n_bits=config.get("nBits", 2048),
                    use_features=config.get("use_features", False)
                )
                if fp is not None:
                    fingerprints["Morgan"] = fp

            elif fp_type == "MACCS":
                fp = self.calculate_maccs_keys(mol)
                if fp is not None:
                    fingerprints["MACCS"] = fp

            elif fp_type == "Topological":
                config = self.fp_config.get("Topological", {})
                fp = self.calculate_topological_fingerprint(
                    mol,
                    n_bits=config.get("nBits", 2048)
                )
                if fp is not None:
                    fingerprints["Topological"] = fp

        return fingerprints


class MolecularDescriptors:
    """
    分子描述符计算类

    计算RDKit分子描述符，包括：
    - 基础理化性质
    - 电子拓扑状态
    - 片段描述符
    """

    def __init__(self, descriptor_types: Optional[List[str]] = None):
        """
        初始化分子描述符计算器

        参数:
            descriptor_types: 要计算的描述符类型
        """
        self.descriptor_types = descriptor_types or ["basic", "electronic", "structural"]
        self.desc_config = FEATURE_CONFIG.get("descriptors", {})

    def calculate_basic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        计算基础理化性质描述符

        参数:
            mol: RDKit分子对象

        返回:
            dict: 基础描述符字典
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            descriptors = {
                "Molecular_Weight": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "Num_H_Donors": Lipinski.NumHDonors(mol),
                "Num_H_Acceptors": Lipinski.NumHAcceptors(mol),
                "Num_Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
                "Num_Heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
                "Num_Heavy_Atoms": rdMolDescriptors.CalcNumHeavyAtoms(mol),
                "Num_Aromatic_Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "Num_Saturated_Rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
                "Num_Aliphatic_Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
                "Ring_Count": rdMolDescriptors.CalcNumRings(mol),
                "Fraction_CSP3": Descriptors.FractionCSP3(mol),
                "Num_Valence_Electrons": Descriptors.NumValenceElectrons(mol),
                "Num_ Radical_Electrons": Descriptors.NumRadicalElectrons(mol),
                "Labute_ASA": Descriptors.LabuteASA(mol),
                "BALONEY_J": Descriptors.BalabanJ(mol),
                "Bertz_CT": Descriptors.BertzCT(mol),
                "Hall_Kier_Alpha": Descriptors.HallKierAlpha(mol),
                "Kappa1": Descriptors.Kappa1(mol),
                "Kappa2": Descriptors.Kappa2(mol),
                "Kappa3": Descriptors.Kappa3(mol),
                "Chi0": Descriptors.Chi0(mol),
                "Chi1": Descriptors.Chi1(mol),
                "Chi0n": Descriptors.Chi0n(mol),
                "Chi1n": Descriptors.Chi1n(mol),
                "Chi2n": Descriptors.Chi2n(mol),
                "Chi3n": Descriptors.Chi3n(mol),
                "Chi4n": Descriptors.Chi4n(mol),
                "Chi0v": Descriptors.Chi0v(mol),
                "Chi1v": Descriptors.Chi1v(mol),
                "Chi2v": Descriptors.Chi2v(mol),
                "Chi3v": Descriptors.Chi3v(mol),
                "Chi4v": Descriptors.Chi4v(mol),
                "Exact_Mol_Wt": Descriptors.ExactMolWt(mol),
                "Mol_Radii": Descriptors.MolRadius(mol),
                "Density": Descriptors.MolDensity(mol),
                "Polar_Surface_Area": Descriptors.PolarSurfaceArea(mol),
                "Heavy_Atom_Mol_Wt": Descriptors.HeavyAtomMolWt(mol),
                "Max_Partial_Charge": Descriptors.MaxPartialCharge(mol),
                "Min_Partial_Charge": Descriptors.MinPartialCharge(mol),
                "Max_Abs_Partial_Charge": Descriptors.MaxAbsPartialCharge(mol),
                "Min_Abs_Partial_Charge": Descriptors.MinAbsPartialCharge(mol),
            }

            return descriptors

        except Exception as e:
            logger.debug(f"Error calculating basic descriptors: {str(e)}")
            return {}

    def calculate_electronic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        计算电子描述符

        参数:
            mol: RDKit分子对象

        返回:
            dict: 电子描述符字典
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            descriptors = {
                "Num_Valence_Electrons": Descriptors.NumValenceElectrons(mol),
                "Num_Radical_Electrons": Descriptors.NumRadicalElectrons(mol),
                "Max_Partial_Charge": Descriptors.MaxPartialCharge(mol),
                "Min_Partial_Charge": Descriptors.MinPartialCharge(mol),
                "Max_Abs_Partial_Charge": Descriptors.MaxAbsPartialCharge(mol),
                "Min_Abs_Partial_Charge": Descriptors.MinAbsPartialCharge(mol),
            }

            return descriptors

        except Exception as e:
            logger.debug(f"Error calculating electronic descriptors: {str(e)}")
            return {}

    def calculate_structural_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        计算结构描述符

        参数:
            mol: RDKit分子对象

        返回:
            dict: 结构描述符字典
        """
        if not HAS_RDKIT or mol is None:
            return {}

        try:
            descriptors = {
                "Num_Atoms": mol.GetNumAtoms(),
                "Num_Bonds": mol.GetNumBonds(),
                "Num_Heavy_Atoms": CalcNumHeavyAtoms(mol),
                "Num_Rings": Descriptors.RingCount(mol),
                "Num_Aromatic_Rings": CalcNumAromaticRings(mol),
                "Num_Saturated_Rings": CalcNumSaturatedRings(mol),
                "Num_Aliphatic_Rings": CalcNumAliphaticRings(mol),
                "Num_Heteroatoms": CalcNumHeteroatoms(mol),
                "Num_Chiral_Centers": len(Chem.FindMolChiralCenters(mol)),
            }

            return descriptors

        except Exception as e:
            logger.debug(f"Error calculating structural descriptors: {str(e)}")
            return {}

    def calculate_all_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        计算所有配置的描述符

        参数:
            mol: RDKit分子对象

        返回:
            dict: 所有描述符的字典
        """
        all_descriptors = {}

        for desc_type in self.descriptor_types:
            if desc_type == "basic":
                all_descriptors.update(self.calculate_basic_descriptors(mol))
            elif desc_type == "electronic":
                all_descriptors.update(self.calculate_electronic_descriptors(mol))
            elif desc_type == "structural":
                all_descriptors.update(self.calculate_structural_descriptors(mol))

        return all_descriptors


class FeatureEngineering:
    """
    特征工程类

    用于批量计算分子特征、标准化和特征选择
    """

    def __init__(self,
                fingerprint_types: List[str] = None,
                descriptor_types: List[str] = None):
        """
        初始化特征工程

        参数:
            fingerprint_types: 要计算的指纹类型
            descriptor_types: 要计算的描述符类型
        """
        self.fingerprint_calculator = MolecularFingerprints(fingerprint_types)
        self.descriptor_calculator = MolecularDescriptors(descriptor_types)

        self.fingerprint_scaler = None
        self.descriptor_scaler = None

        self.feature_names = {
            "fingerprint": [],
            "descriptor": []
        }

    def calculate_fingerprints(self, mols: List[Chem.Mol]) -> np.ndarray:
        """
        批量计算分子指纹

        参数:
            mols: RDKit分子对象列表

        返回:
            np.ndarray: 指纹特征矩阵
        """
        fingerprints = []

        for mol in mols:
            if mol is None:
                fingerprints.append(None)
                continue

            fp_dict = self.fingerprint_calculator.calculate_all_fingerprints(mol)

            if not fp_dict:
                fingerprints.append(None)
                continue

            combined_fp = np.concatenate(list(fp_dict.values()))
            fingerprints.append(combined_fp)

        if not fingerprints or all(f is None for f in fingerprints):
            return np.array([])

        valid_fps = [f for f in fingerprints if f is not None]
        max_len = max(len(f) for f in valid_fps) if valid_fps else 0

        padded_fps = []
        for fp in fingerprints:
            if fp is None:
                padded_fps.append(np.zeros(max_len))
            else:
                padded_fps.append(fp if len(fp) == max_len else np.pad(fp, (0, max_len - len(fp))))

        self.feature_names["fingerprint"] = [f"FP_{i}" for i in range(max_len)]

        return np.array(padded_fps)

    def calculate_descriptors(self, mols: List[Chem.Mol]) -> np.ndarray:
        """
        批量计算分子描述符

        参数:
            mols: RDKit分子对象列表

        返回:
            np.ndarray: 描述符特征矩阵
        """
        all_descriptors = []

        for mol in mols:
            if mol is None:
                all_descriptors.append({})
                continue

            desc = self.descriptor_calculator.calculate_all_descriptors(mol)
            all_descriptors.append(desc)

        if not all_descriptors or all(not d for d in all_descriptors):
            return np.array([])

        descriptor_names = list(all_descriptors[0].keys()) if all_descriptors else []

        descriptor_matrix = []
        for desc in all_descriptors:
            row = [desc.get(name, np.nan) for name in descriptor_names]
            descriptor_matrix.append(row)

        self.feature_names["descriptor"] = descriptor_names

        return np.array(descriptor_matrix)

    def calculate_all_features(self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        计算所有特征（指纹 + 描述符）

        参数:
            mols: RDKit分子对象列表

        返回:
            tuple: (特征矩阵, 有效分子索引, 特征名称列表)
        """
        fingerprints = self.calculate_fingerprints(mols)
        descriptors = self.calculate_descriptors(mols)

        if fingerprints.size == 0 and descriptors.size == 0:
            return np.array([]), [], []

        if fingerprints.size == 0:
            features = descriptors
        elif descriptors.size == 0:
            features = fingerprints
        else:
            features = np.hstack([fingerprints, descriptors])

        feature_names = self.feature_names["fingerprint"] + self.feature_names["descriptor"]

        valid_indices = [i for i, mol in enumerate(mols) if mol is not None]

        return features, valid_indices, feature_names

    def standardize_features(self,
                           X: np.ndarray,
                           method: str = "standard") -> np.ndarray:
        """
        标准化特征

        参数:
            X: 特征矩阵
            method: 标准化方法 ('standard', 'minmax', 'robust')

        返回:
            np.ndarray: 标准化后的特征矩阵
        """
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard")
            scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        if method == "standard":
            self.fingerprint_scaler = scaler
        else:
            self.descriptor_scaler = scaler

        return X_scaled

    def remove_low_variance_features(self,
                                   X: np.ndarray,
                                   feature_names: List[str],
                                   threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        """
        移除低方差特征

        参数:
            X: 特征矩阵
            feature_names: 特征名称列表
            threshold: 方差阈值

        返回:
            tuple: (过滤后的特征矩阵, 保留的特征名称)
        """
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)

        selected_mask = selector.get_support()
        selected_features = [name for i, name in enumerate(feature_names) if selected_mask[i]]

        logger.info(f"Removed {len(feature_names) - len(selected_features)} low variance features")

        return X_selected, selected_features

    def select_k_best_features(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              feature_names: List[str],
                              k: int = 100,
                              method: str = "f_classif") -> Tuple[np.ndarray, List[str]]:
        """
        选择K个最佳特征

        参数:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
            k: 要选择的特征数量
            method: 选择方法 ('f_classif', 'mutual_info')

        返回:
            tuple: (选择的特征矩阵, 选择的特征名称)
        """
        if method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        elif method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return X, feature_names

        X_selected = selector.fit_transform(X, y)

        selected_indices = selector.get_support()
        selected_features = [name for i, name in enumerate(feature_names) if selected_indices[i]]

        logger.info(f"Selected {len(selected_features)} best features")

        return X_selected, selected_features


class FeatureDataset:
    """
    特征数据集类

    用于管理分子特征数据集，包括加载、处理和保存
    """

    def __init__(self, name: str = "feature_dataset"):
        """
        初始化特征数据集

        参数:
            name: 数据集名称
        """
        self.name = name
        self.features = None
        self.feature_names = []
        self.molecule_names = []
        self.labels = None
        self.metadata = {}

        logger.info(f"Initialized FeatureDataset: {name}")

    def load_from_smiles(self,
                        smiles_list: List[str],
                        molecule_names: Optional[List[str]] = None,
                        labels: Optional[np.ndarray] = None,
                        fingerprint_types: List[str] = None,
                        descriptor_types: List[str] = None) -> bool:
        """
        从SMILES列表加载数据并计算特征

        参数:
            smiles_list: SMILES字符串列表
            molecule_names: 分子名称列表
            labels: 活性标签数组
            fingerprint_types: 指纹类型列表
            descriptor_types: 描述符类型列表

        返回:
            bool: 加载是否成功
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return False

        try:
            mols = []
            valid_indices = []

            for i, smiles in enumerate(smiles_list):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mols.append(mol)
                        valid_indices.append(i)
                except Exception as e:
                    logger.debug(f"Error parsing SMILES at index {i}: {str(e)}")

            if not mols:
                logger.error("No valid molecules parsed")
                return False

            self.molecule_names = [molecule_names[i] if molecule_names else f"Mol_{i}" for i in valid_indices]

            if labels is not None:
                self.labels = np.array([labels[i] for i in valid_indices])

            feature_engineering = FeatureEngineering(fingerprint_types, descriptor_types)
            self.features, _, self.feature_names = feature_engineering.calculate_all_features(mols)

            logger.info(f"Loaded {len(mols)} molecules with {len(self.feature_names)} features")
            return True

        except Exception as e:
            logger.error(f"Error loading from SMILES: {str(e)}")
            return False

    def load_from_dataframe(self,
                           df: pd.DataFrame,
                           smiles_column: str = "SMILES",
                           name_column: Optional[str] = None,
                           label_column: Optional[str] = None,
                           fingerprint_types: List[str] = None,
                           descriptor_types: List[str] = None) -> bool:
        """
        从DataFrame加载数据并计算特征

        参数:
            df: 包含SMILES的DataFrame
            smiles_column: SMILES列名
            name_column: 分子名称列名
            label_column: 活性标签列名
            fingerprint_types: 指纹类型列表
            descriptor_types: 描述符类型列表

        返回:
            bool: 加载是否成功
        """
        smiles_list = df[smiles_column].tolist()

        molecule_names = None
        if name_column and name_column in df.columns:
            molecule_names = df[name_column].tolist()

        labels = None
        if label_column and label_column in df.columns:
            labels = df[label_column].values

        return self.load_from_smiles(
            smiles_list,
            molecule_names,
            labels,
            fingerprint_types,
            descriptor_types
        )

    def split_data(self,
                  train_ratio: float = 0.7,
                  test_ratio: float = 0.3,
                  stratify: bool = True,
                  random_state: int = 42) -> Dict[str, Any]:
        """
        划分训练集和测试集

        参数:
            train_ratio: 训练集比例
            test_ratio: 测试集比例
            stratify: 是否分层抽样
            random_state: 随机种子

        返回:
            dict: 包含划分结果的字典
        """
        if self.features is None or len(self.features) == 0:
            logger.error("No features to split")
            return {}

        try:
            from sklearn.model_selection import train_test_split

            kwargs = {
                "test_size": test_ratio,
                "random_state": random_state,
                "shuffle": True
            }

            if stratify and self.labels is not None:
                kwargs["stratify"] = self.labels

            indices = np.arange(len(self.features))
            train_idx, test_idx = train_test_split(indices, **kwargs)

            split_results = {
                "train_features": self.features[train_idx],
                "test_features": self.features[test_idx],
                "train_indices": train_idx,
                "test_indices": test_idx,
                "train_names": [self.molecule_names[i] for i in train_idx],
                "test_names": [self.molecule_names[i] for i in test_idx]
            }

            if self.labels is not None:
                split_results["train_labels"] = self.labels[train_idx]
                split_results["test_labels"] = self.labels[test_idx]

            logger.info(f"Split data: {len(train_idx)} train, {len(test_idx)} test")

            return split_results

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return {}

    def save(self, output_path: Path) -> bool:
        """
        保存特征数据集

        参数:
            output_path: 输出文件路径

        返回:
            bool: 保存是否成功
        """
        try:
            data = {
                "name": self.name,
                "features": self.features,
                "feature_names": self.feature_names,
                "molecule_names": self.molecule_names,
                "labels": self.labels,
                "metadata": self.metadata
            }

            np.savez(output_path, **data)
            logger.info(f"Saved feature dataset to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            return False

    def load(self, input_path: Path) -> bool:
        """
        加载特征数据集

        参数:
            input_path: 输入文件路径

        返回:
            bool: 加载是否成功
        """
        try:
            data = np.load(input_path, allow_pickle=True)

            self.name = data.get("name", "loaded_dataset")
            self.features = data["features"]
            self.feature_names = data["feature_names"].tolist()
            self.molecule_names = data["molecule_names"].tolist()
            self.labels = data.get("labels", None)
            self.metadata = data.get("metadata", {}).item() if "metadata" in data else {}

            logger.info(f"Loaded feature dataset from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        返回:
            dict: 统计信息字典
        """
        stats = {
            "name": self.name,
            "num_molecules": len(self.features) if self.features is not None else 0,
            "num_features": len(self.feature_names),
            "feature_types": {
                "fingerprint": len([n for n in self.feature_names if n.startswith("FP_")]),
                "descriptor": len([n for n in self.feature_names if not n.startswith("FP_")])
            }
        }

        if self.labels is not None:
            stats["label_distribution"] = {
                "positive": int(np.sum(self.labels == 1)),
                "negative": int(np.sum(self.labels == 0))
            }

        if self.features is not None:
            stats["feature_statistics"] = {
                "mean": float(np.mean(self.features)),
                "std": float(np.std(self.features)),
                "min": float(np.min(self.features)),
                "max": float(np.max(self.features))
            }

        return stats


def generate_features(smiles_list: List[str],
                    molecule_names: Optional[List[str]] = None,
                    labels: Optional[np.ndarray] = None,
                    fingerprint_types: List[str] = None,
                    descriptor_types: List[str] = None) -> Optional[FeatureDataset]:
    """
    便捷函数：生成特征数据集

    参数:
        smiles_list: SMILES字符串列表
        molecule_names: 分子名称列表
        labels: 活性标签数组
        fingerprint_types: 指纹类型列表
        descriptor_types: 描述符类型列表

    返回:
        FeatureDataset: 特征数据集对象
    """
    dataset = FeatureDataset("generated_dataset")

    success = dataset.load_from_smiles(
        smiles_list,
        molecule_names,
        labels,
        fingerprint_types,
        descriptor_types
    )

    if success:
        return dataset
    else:
        return None


if __name__ == "__main__":
    logger.info("Testing MolecularFeatures module")

    test_smiles = ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]

    dataset = generate_features(test_smiles)

    if dataset:
        logger.info(f"Generated dataset with {len(dataset.features)} molecules and {len(dataset.feature_names)} features")
        logger.info(f"Dataset statistics: {dataset.get_statistics()}")
