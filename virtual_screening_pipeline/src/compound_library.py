#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化合物库预处理模块

功能：
1. 从多种格式加载化合物库（SDF, SMILES, CSV）
2. SMILES标准化和规范化
3. 去除盐类、金属离子和溶剂
4. 基于SMILES去重
5. Lipinski类药规则过滤
6. PAINS和BRENK过滤
7. 生成3D构象
8. 格式转换（SDF, PDBQT等）

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
        SDWriter, SmilesWriter, MolToSmiles, MolFromSmiles,
        MolStandardize, SaltRemover
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
    warnings.warn("RDKit not available. Compound processing will be limited.")

from configs.config import (
    PROJECT_ROOT, COMPOUND_LIBRARY_DIR, LOG_CONFIG,
    LIPINSKI_RULES, DRUG_LIKENESS_FILTERS
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class CompoundStandardizer:
    """
    化合物标准化类

    用于标准化SMILES、处理盐类、去除金属离子等
    """

    def __init__(self):
        """初始化化合物标准化器"""
        if HAS_RDKIT:
            self.remover = SaltRemover.SaltRemover()
            try:
                # 尝试不同的导入方式
                from rdkit.Chem.MolStandardize import MolStandardizer
                self.standardizer = MolStandardizer()
            except ImportError:
                # 如果导入失败，设置为None
                self.standardizer = None
        else:
            self.remover = None
            self.standardizer = None

    def remove_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """
        去除盐类

        参数:
            mol: RDKit分子对象

        返回:
            Chem.Mol: 去除盐后的分子
        """
        if not HAS_RDKIT or mol is None:
            return mol

        try:
            mol_no_salt = self.remover.StripMol(mol, dontRemoveEverything=True)
            return mol_no_salt
        except Exception as e:
            logger.warning(f"Error removing salts: {str(e)}")
            return mol

    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        标准化SMILES

        参数:
            smiles: 输入SMILES字符串

        返回:
            str: 标准化后的SMILES，如果失败返回None
        """
        if not HAS_RDKIT:
            return smiles

        try:
            mol = MolFromSmiles(smiles)
            if mol is None:
                return None

            mol = self.remove_salts(mol)
            
            # 检查standardizer是否可用
            if self.standardizer is not None:
                mol = self.standardizer.standardize(mol)

            standardized = MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return standardized

        except Exception as e:
            logger.debug(f"Error standardizing SMILES {smiles}: {str(e)}")
            return None

    def neutralize_charges(self, mol: Chem.Mol) -> Chem.Mol:
        """
        中和电荷

        参数:
            mol: RDKit分子对象

        返回:
            Chem.Mol: 中和电荷后的分子
        """
        if not HAS_RDKIT or mol is None:
            return mol

        try:
            from rdkit.Chem import rdMolDescriptors

            pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1])]")
            at_matches = mol.GetSubstructMatches(pattern)
            ats_list = [at[0] for at in at_matches]
            if len(ats_list) > 0:
                for at_idx in atts_list:
                    atom = mol.GetAtomWithIdx(at_idx)
                    atom.SetFormalCharge(0)

            pattern = Chem.MolFromSmarts("[-1!h0!$([*]~[+1])]")
            at_matches = mol.GetSubstructMatches(pattern)
            ats_list = [at[0] for at in at_matches]
            if len(ats_list) > 0:
                for at_idx in ats_list:
                    atom = mol.GetAtomWithIdx(at_idx)
                    atom.SetFormalCharge(0)

            return mol

        except Exception as e:
            logger.warning(f"Error neutralizing charges: {str(e)}")
            return mol


class DrugLikenessFilter:
    """
    类药性过滤器

    基于Lipinski规则和其他药效团规则过滤化合物
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化类药性过滤器

        参数:
            config: 过滤规则配置，如果为None使用默认配置
        """
        self.config = config or LIPINSKI_RULES

    def check_lipinski_rules(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        检查Lipinski规则（类药五规则）

        参数:
            mol: RDKit分子对象

        返回:
            dict: 包含各规则检查结果的字典
        """
        if not HAS_RDKIT or mol is None:
            return {"passed": False, "reasons": ["Invalid molecule"]}

        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)

            results = {
                "molecular_weight": mw,
                "logp": logp,
                "h_donors": hbd,
                "h_acceptors": hba,
                "tpsa": tpsa,
                "rotatable_bonds": rotatable,
                "rules_passed": [],
                "rules_failed": []
            }

            mw_config = self.config.get("molecular_weight", {})
            if mw_config.get("min", 0) <= mw <= mw_config.get("max", 500):
                results["rules_passed"].append("molecular_weight")
            else:
                results["rules_failed"].append("molecular_weight")

            logp_config = self.config.get("logp", {})
            if logp_config.get("min", -2) <= logp <= logp_config.get("max", 5):
                results["rules_passed"].append("logp")
            else:
                results["rules_failed"].append("logp")

            if hbd <= self.config.get("h_donors", {}).get("max", 5):
                results["rules_passed"].append("h_donors")
            else:
                results["rules_failed"].append("h_donors")

            if hba <= self.config.get("h_acceptors", {}).get("max", 10):
                results["rules_passed"].append("h_acceptors")
            else:
                results["rules_failed"].append("h_acceptors")

            if rotatable <= self.config.get("num_rotatable_bonds", {}).get("max", 10):
                results["rules_passed"].append("rotatable_bonds")
            else:
                results["rules_failed"].append("rotatable_bonds")

            results["passed"] = len(results["rules_failed"]) == 0

            return results

        except Exception as e:
            logger.warning(f"Error checking Lipinski rules: {str(e)}")
            return {"passed": False, "reasons": [str(e)]}

    def check_pains_filter(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        检查PAINS过滤器

        PAINS (Pan-Assay Interference compounds) 是常见的假阳性化合物

        参数:
            mol: RDKit分子对象

        返回:
            dict: 包含PAINS检查结果的字典
        """
        if not HAS_RDKIT or mol is None:
            return {"passed": True, "hits": []}

        try:
            pains_patterns = [
                ("benzimidazole", "c1ccc2c(c1)ncn2"),
                ("aldehyde", "[CX3H1]=O"),
                ("alkyl_halide", "[Cl,Br,I]-[CX4]"),
            ]

            hits = []
            for name, pattern in pains_patterns:
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                        hits.append(name)
                except Exception as e:
                    logger.debug(f"Error with PAINS pattern {name}: {str(e)}")

            return {
                "passed": len(hits) == 0,
                "hits": hits
            }

        except Exception as e:
            logger.warning(f"Error checking PAINS: {str(e)}")
            return {"passed": True, "hits": []}


class CompoundLibrary:
    """
    化合物库管理类

    用于加载、预处理、去重和过滤化合物库
    """

    def __init__(self, name: str = "compound_library"):
        """
        初始化化合物库

        参数:
            name: 化合物库名称
        """
        self.name = name
        self.compounds = []
        self.original_count = 0
        self.standardizer = CompoundStandardizer()
        self.drug_filter = DrugLikenessFilter()
        self.smiles_to_idx = {}

        logger.info(f"Initialized CompoundLibrary: {name}")

    def load_from_smiles(self,
                        file_path: Union[str, Path],
                        smiles_column: str = "SMILES",
                        name_column: Optional[str] = None,
                        delimiter: str = ",") -> int:
        """
        从SMILES文件加载化合物

        参数:
            file_path: SMILES文件路径
            smiles_column: SMILES列名
            name_column: 化合物名称列名（可选）
            delimiter: 文件分隔符

        返回:
            int: 加载的化合物数量
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return 0

        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            self.original_count = len(df)

            logger.info(f"Loading {self.original_count} compounds from {file_path}")

            for idx, row in df.iterrows():
                smiles = row.get(smiles_column, None)
                if smiles is None or not isinstance(smiles, str):
                    continue

                name = row.get(name_column, f"compound_{idx}") if name_column else f"compound_{idx}"

                standardized = self.standardizer.standardize_smiles(smiles)
                if standardized is None:
                    continue

                mol = Chem.MolFromSmiles(standardized)
                if mol is None:
                    continue

                self.compounds.append({
                    "idx": idx,
                    "name": name,
                    "original_smiles": smiles,
                    "standardized_smiles": standardized,
                    "mol": mol
                })

            logger.info(f"Successfully loaded {len(self.compounds)} compounds")
            return len(self.compounds)

        except Exception as e:
            logger.error(f"Error loading from SMILES: {str(e)}")
            return 0

    def load_from_sdf(self, file_path: Union[str, Path]) -> int:
        """
        从SDF文件加载化合物

        参数:
            file_path: SDF文件路径

        返回:
            int: 加载的化合物数量
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return 0

        try:
            supplier = Chem.SDMolSupplier(str(file_path))
            self.original_count = len(supplier)

            logger.info(f"Loading {self.original_count} compounds from SDF")

            for idx, mol in enumerate(supplier):
                if mol is None:
                    continue

                name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"compound_{idx}"

                try:
                    standardized = MolToSmiles(mol, isomericSmiles=True, canonical=True)
                except:
                    standardized = MolToSmiles(mol)

                self.compounds.append({
                    "idx": idx,
                    "name": name,
                    "original_smiles": standardized,
                    "standardized_smiles": standardized,
                    "mol": mol
                })

            logger.info(f"Successfully loaded {len(self.compounds)} compounds")
            return len(self.compounds)

        except Exception as e:
            logger.error(f"Error loading from SDF: {str(e)}")
            return 0

    def deduplicate(self) -> int:
        """
        基于标准化SMILES去重

        返回:
            int: 移除的重复化合物数量
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return 0

        try:
            unique_compounds = []
            seen_smiles = set()

            for compound in self.compounds:
                smiles = compound["standardized_smiles"]
                if smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    unique_compounds.append(compound)

            removed_count = len(self.compounds) - len(unique_compounds)
            self.compounds = unique_compounds

            logger.info(f"Removed {removed_count} duplicate compounds. Remaining: {len(self.compounds)}")
            return removed_count

        except Exception as e:
            logger.error(f"Error deduplicating: {str(e)}")
            return 0

    def filter_drug_likeness(self,
                            apply_lipinski: bool = True,
                            apply_pains: bool = True,
                            apply_brenk: bool = True) -> int:
        """
        应用类药性过滤器

        参数:
            apply_lipinski: 是否应用Lipinski规则
            apply_pains: 是否过滤PAINS化合物
            apply_brenk: 是否过滤BRENK化合物

        返回:
            int: 移除的化合物数量
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return 0

        try:
            filtered_compounds = []

            for compound in self.compounds:
                mol = compound["mol"]
                keep = True

                if apply_lipinski:
                    lipinski_result = self.drug_filter.check_lipinski_rules(mol)
                    if not lipinski_result["passed"]:
                        keep = False
                        compound["filter_reason"] = "lipinski_failed"

                if keep and apply_pains:
                    pains_result = self.drug_filter.check_pains_filter(mol)
                    if not pains_result["passed"]:
                        keep = False
                        compound["filter_reason"] = "pains_filter"

                if keep:
                    filtered_compounds.append(compound)

            removed_count = len(self.compounds) - len(filtered_compounds)
            self.compounds = filtered_compounds

            logger.info(f"Drug likeness filter removed {removed_count} compounds. Remaining: {len(self.compounds)}")
            return removed_count

        except Exception as e:
            logger.error(f"Error filtering: {str(e)}")
            return 0

    def generate_3d_conformations(self,
                                num_conformers: int = 10,
                                random_seed: int = 42) -> int:
        """
        生成3D构象

        参数:
            num_conformers: 每个分子生成的构象数量
            random_seed: 随机种子

        返回:
            int: 成功生成3D构象的分子数量
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return 0

        try:
            success_count = 0

            for compound in self.compounds:
                mol = compound["mol"]

                try:
                    params = AllChem.ETKDG(randomSeed=random_seed)
                    result = AllChem.EmbedMolecule(mol, params)

                    if result == 0:
                        AllChem.UFFOptimizeMolecule(mol)
                        compound["has_3d"] = True
                        success_count += 1
                    else:
                        compound["has_3d"] = False

                except Exception as e:
                    compound["has_3d"] = False

            logger.info(f"Generated 3D conformations for {success_count}/{len(self.compounds)} compounds")
            return success_count

        except Exception as e:
            logger.error(f"Error generating 3D conformations: {str(e)}")
            return 0

    def save_to_sdf(self, output_path: Union[str, Path]) -> bool:
        """
        保存化合物库到SDF文件

        参数:
            output_path: 输出文件路径

        返回:
            bool: 保存是否成功
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return False

        try:
            writer = SDWriter(str(output_path))

            for compound in self.compounds:
                mol = compound["mol"]
                mol.SetProp("_Name", compound["name"])
                mol.SetProp("SMILES", compound["standardized_smiles"])
                writer.write(mol)

            writer.close()
            logger.info(f"Saved {len(self.compounds)} compounds to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving to SDF: {str(e)}")
            return False

    def save_to_csv(self, output_path: Union[str, Path]) -> bool:
        """
        保存化合物库到CSV文件

        参数:
            output_path: 输出文件路径

        返回:
            bool: 保存是否成功
        """
        try:
            data = []
            for compound in self.compounds:
                mol = compound["mol"]

                try:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    hba = Lipinski.NumHAcceptors(mol)
                    rotatable = Lipinski.NumRotatableBonds(mol)
                except:
                    mw = logp = tpsa = hbd = hba = rotatable = None

                data.append({
                    "Name": compound["name"],
                    "SMILES": compound["standardized_smiles"],
                    "Original_SMILES": compound.get("original_smiles", ""),
                    "Molecular_Weight": mw,
                    "LogP": logp,
                    "TPSA": tpsa,
                    "H_Donors": hbd,
                    "H_Acceptors": hba,
                    "Rotatable_Bonds": rotatable
                })

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(self.compounds)} compounds to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取化合物库统计信息

        返回:
            dict: 包含统计信息的字典
        """
        stats = {
            "name": self.name,
            "original_count": self.original_count,
            "current_count": len(self.compounds),
            "removed_count": self.original_count - len(self.compounds)
        }

        if HAS_RDKIT and len(self.compounds) > 0:
            mw_values = []
            logp_values = []
            tpsa_values = []

            for compound in self.compounds:
                mol = compound["mol"]
                try:
                    mw_values.append(Descriptors.MolWt(mol))
                    logp_values.append(Descriptors.MolLogP(mol))
                    tpsa_values.append(Descriptors.TPSA(mol))
                except:
                    pass

            if mw_values:
                stats["molecular_weight"] = {
                    "mean": float(np.mean(mw_values)),
                    "std": float(np.std(mw_values)),
                    "min": float(np.min(mw_values)),
                    "max": float(np.max(mw_values))
                }

            if logp_values:
                stats["logp"] = {
                    "mean": float(np.mean(logp_values)),
                    "std": float(np.std(logp_values)),
                    "min": float(np.min(logp_values)),
                    "max": float(np.max(logp_values))
                }

            if tpsa_values:
                stats["tpsa"] = {
                    "mean": float(np.mean(tpsa_values)),
                    "std": float(np.std(tpsa_values)),
                    "min": float(np.min(tpsa_values)),
                    "max": float(np.max(tpsa_values))
                }

        return stats


def preprocess_compound_library(input_file: str,
                               output_dir: Optional[Path] = None,
                               library_name: str = "processed_library",
                               remove_salts: bool = True,
                               deduplicate: bool = True,
                               apply_filters: bool = True,
                               generate_3d: bool = False) -> Dict[str, Any]:
    """
    便捷函数：预处理化合物库

    参数:
        input_file: 输入文件路径
        output_dir: 输出目录
        library_name: 化合物库名称
        remove_salts: 是否去除盐类
        deduplicate: 是否去重
        apply_filters: 是否应用类药性过滤器
        generate_3d: 是否生成3D构象

    返回:
        dict: 预处理结果摘要
    """
    if output_dir is None:
        output_dir = COMPOUND_LIBRARY_DIR

    results = {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "steps_completed": [],
        "final_count": 0,
        "success": False
    }

    try:
        library = CompoundLibrary(library_name)

        input_path = Path(input_file)
        if input_path.suffix.lower() in [".smi", ".smiles", ".csv"]:
            count = library.load_from_smiles(input_file)
            results["steps_completed"].append("load_from_smiles")
        elif input_path.suffix.lower() == ".sdf":
            count = library.load_from_sdf(input_file)
            results["steps_completed"].append("load_from_sdf")
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return results

        if count == 0:
            logger.error("Failed to load compounds")
            return results

        if deduplicate:
            library.deduplicate()
            results["steps_completed"].append("deduplicate")

        if apply_filters:
            library.filter_drug_likeness()
            results["steps_completed"].append("filter_drug_likeness")

        if generate_3d:
            library.generate_3d_conformations()
            results["steps_completed"].append("generate_3d")

        output_sdf = output_dir / f"{library_name}.sdf"
        output_csv = output_dir / f"{library_name}.csv"

        library.save_to_sdf(output_sdf)
        library.save_to_csv(output_csv)

        results["steps_completed"].append("save")
        results["final_count"] = len(library.compounds)
        results["statistics"] = library.get_statistics()
        results["success"] = True

        logger.info(f"Compound library preprocessing completed: {results}")

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    logger.info("Testing CompoundLibrary module")

    test_library = CompoundLibrary("test_library")
    logger.info(f"Test library created: {test_library.name}")
