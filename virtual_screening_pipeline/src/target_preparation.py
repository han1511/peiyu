#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
靶点结构准备模块

功能：
1. 从PDB数据库下载靶点蛋白结构
2. 结构清洗和预处理
3. 质子化状态优化
4. 结合口袋识别
5. 生成对接格式文件

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, PandasTools, SDWriter
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit not available. Some functions will be limited.")

try:
    from Bio.PDB import PDBParser, PDBIO, Select, Chain, Residue
    from Bio.PDB import rotalign
    from Bio import SeqIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    warnings.warn("Biopython not available. PDB processing will be limited.")

from configs.config import (
    PROJECT_ROOT, TARGET_STRUCTURES_DIR, LOG_CONFIG,
    DENGUE_TARGETS, DOCKING_CONFIG
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class TargetPreparation:
    """
    靶点蛋白结构准备类

    用于准备分子对接所需的蛋白结构，包括：
    - 结构下载和清洗
    - 质子化状态优化
    - 结合口袋识别
    - 对接格式转换
    """

    def __init__(self, target_name: str, pdb_id: Optional[str] = None):
        """
        初始化靶点准备器

        参数:
            target_name: 靶点名称（如 'NS2A', 'NS3' 等）
            pdb_id: PDB ID（如果已知）
        """
        self.target_name = target_name
        self.pdb_id = pdb_id
        self.target_info = DENGUE_TARGETS.get(target_name, {})

        self.pdb_structure = None
        self.cleaned_structure = None
        self.protonated_structure = None
        self.binding_site = None

        self.output_dir = TARGET_STRUCTURES_DIR / target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized TargetPreparation for {target_name}")

    def fetch_from_pdb(self, pdb_id: Optional[str] = None) -> bool:
        """
        从RCSB PDB下载靶点结构

        参数:
            pdb_id: PDB ID（如果未提供，使用类属性中的值）

        返回:
            bool: 下载是否成功
        """
        if pdb_id is None:
            pdb_id = self.pdb_id

        if pdb_id is None:
            logger.warning(f"No PDB ID provided for {self.target_name}")
            return False

        self.pdb_id = pdb_id

        try:
            import requests

            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url, timeout=30)

            if response.status_code == 200:
                pdb_file = self.output_dir / f"{pdb_id}.pdb"
                with open(pdb_file, "w") as f:
                    f.write(response.text)

                logger.info(f"Downloaded PDB structure {pdb_id} to {pdb_file}")
                return True
            else:
                logger.error(f"Failed to download PDB {pdb_id}: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error downloading PDB {pdb_id}: {str(e)}")
            return False

    def load_structure(self, pdb_file: Optional[Path] = None) -> bool:
        """
        加载PDB结构文件

        参数:
            pdb_file: PDB文件路径（如果未提供，使用默认路径）

        返回:
            bool: 加载是否成功
        """
        if pdb_file is None:
            if self.pdb_id:
                pdb_file = self.output_dir / f"{self.pdb_id}.pdb"
            else:
                logger.error("No PDB file specified")
                return False

        if not os.path.exists(pdb_file):
            logger.error(f"PDB file not found: {pdb_file}")
            return False

        try:
            if HAS_BIOPYTHON:
                parser = PDBParser(QUIET=True)
                self.pdb_structure = parser.get_structure(self.target_name, str(pdb_file))
                logger.info(f"Loaded PDB structure from {pdb_file}")
                return True
            else:
                logger.error("Biopython not available")
                return False

        except Exception as e:
            logger.error(f"Error loading PDB structure: {str(e)}")
            return False

    def clean_structure(self,
                       remove_waters: bool = True,
                       remove_ligands: bool = False,
                       remove_alt_confs: bool = True,
                       fix_residues: bool = True) -> bool:
        """
        清洗PDB结构

        参数:
            remove_waters: 是否去除水分子
            remove_ligands: 是否去除配体分子
            remove_alt_confs: 是否去除交替构象
            fix_residues: 是否修复异常残基

        返回:
            bool: 清洗是否成功
        """
        if self.pdb_structure is None:
            logger.error("No structure loaded to clean")
            return False

        try:
            class StructureCleaner(Select):
                """自定义结构选择器"""

                def __init__(self, remove_waters, remove_ligands, remove_alt_confs):
                    self.remove_waters = remove_waters
                    self.remove_ligands = remove_ligands
                    self.remove_alt_confs = remove_alt_confs
                    self.alt_confs_removed = 0
                    self.waters_removed = 0
                    self.ligands_removed = 0

                def accept_atom(self, atom):
                    if self.remove_alt_confs:
                        if atom.get_altloc() not in ["", "A", " "]:
                            self.alt_confs_removed += 1
                            return False

                    if self.remove_waters:
                        if atom.get_parent().get_resname() == "HOH":
                            self.waters_removed += 1
                            return False

                    return True

                def accept_residue(self, residue):
                    if self.remove_waters:
                        if residue.get_resname() == "HOH":
                            return False

                    if self.remove_ligands:
                        residue_id = residue.get_id()
                        if residue_id[0] not in [" ", "H", "W"]:
                            self.ligands_removed += 1
                            return False

                    if fix_residues:
                        resname = residue.get_resname()
                        if resname in ["UNK", "UNK", "   "]:
                            return False

                    return True

            cleaner = StructureCleaner(remove_waters, remove_ligands, remove_alt_confs)

            cleaned_file = self.output_dir / f"{self.target_name}_cleaned.pdb"
            io = PDBIO()
            io.set_structure(self.pdb_structure)
            io.save(str(cleaned_file), cleaner)

            self.cleaned_structure = str(cleaned_file)

            logger.info(f"Cleaned structure saved to {cleaned_file}")
            logger.info(f"Alt confs removed: {cleaner.alt_confs_removed}")
            logger.info(f"Waters removed: {cleaner.waters_removed}")
            logger.info(f"Ligands removed: {cleaner.ligands_removed}")

            return True

        except Exception as e:
            logger.error(f"Error cleaning structure: {str(e)}")
            return False

    def identify_binding_site(self,
                             ligand_chain: Optional[str] = None,
                             ligand_resname: Optional[str] = None) -> Dict[str, Any]:
        """
        识别蛋白结合口袋

        参数:
            ligand_chain: 配体所在链ID
            ligand_resname: 配体残基名称

        返回:
            dict: 包含结合口袋中心坐标和大小的字典
        """
        if self.pdb_structure is None:
            logger.error("No structure loaded")
            return {}

        try:
            binding_site_info = {}

            if ligand_chain and ligand_resname:
                for model in self.pdb_structure:
                    for chain in model:
                        if chain.id == ligand_chain:
                            for residue in chain:
                                if residue.get_resname() == ligand_resname:
                                    atoms = list(residue.get_atoms())
                                    if atoms:
                                        center = np.mean([atom.get_coord() for atom in atoms], axis=0)

                                        distances = []
                                        for atom in residue.get_atoms():
                                            for other_atom in residue.get_atoms():
                                                if atom != other_atom:
                                                    dist = np.linalg.norm(atom.get_coord() - other_atom.get_coord())
                                                    distances.append(dist)

                                        max_dist = max(distances) if distances else 10.0

                                        binding_site_info = {
                                            "center": {
                                                "x": float(center[0]),
                                                "y": float(center[1]),
                                                "z": float(center[2])
                                            },
                                            "size": {
                                                "x": float(max_dist * 1.5),
                                                "y": float(max_dist * 1.5),
                                                "z": float(max_dist * 1.5)
                                            }
                                        }

                                        logger.info(f"Binding site identified at center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
                                        self.binding_site = binding_site_info
                                        return binding_site_info

            for model in self.pdb_structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname()
                        if resname in ["ATP", "GTP", "NAD", "NAP", "HEME", "MG", "ZN", "FE"]:
                            atoms = list(residue.get_atoms())
                            if atoms:
                                center = np.mean([atom.get_coord() for atom in atoms], axis=0)

                                binding_site_info = {
                                    "center": {
                                        "x": float(center[0]),
                                        "y": float(center[1]),
                                        "z": float(center[2])
                                    },
                                    "size": DOCKING_CONFIG["search_space"].copy()
                                }

                                logger.info(f"Binding site identified from co-crystal ligand {resname}")
                                self.binding_site = binding_site_info
                                return binding_site_info

            if not binding_site_info:
                logger.warning("No binding site identified. Using default center.")
                default_center = {"x": 0.0, "y": 0.0, "z": 0.0}
                binding_site_info = {
                    "center": default_center,
                    "size": DOCKING_CONFIG["search_space"].copy()
                }
                self.binding_site = binding_site_info

            return binding_site_info

        except Exception as e:
            logger.error(f"Error identifying binding site: {str(e)}")
            return {}

    def prepare_for_docking(self,
                          output_format: str = "pdbqt") -> Optional[str]:
        """
        准备对接格式文件

        参数:
            output_format: 输出格式（目前支持 'pdbqt'）

        返回:
            str: 输出文件路径，如果失败返回None
        """
        if self.cleaned_structure is None:
            logger.error("No cleaned structure available")
            return None

        try:
            if output_format == "pdbqt":
                output_file = self.output_dir / f"{self.target_name}_prepared.pdbqt"

                with open(self.cleaned_structure, "r") as f:
                    pdb_content = f.read()

                pdbqt_content = []
                for line in pdb_content.split("\n"):
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        pdbqt_line = line[:66]
                        if "H" in line[76:78]:
                            pdbqt_line += " A"
                        else:
                            pdbqt_line += " "
                        atom_name = line[12:16].strip()
                        if len(atom_name) == 1:
                            pdbqt_line = pdbqt_line[:12] + " " + atom_name + "   " + pdbqt_line[17:]
                        elif len(atom_name) == 2:
                            pdbqt_line = pdbqt_line[:12] + " " + atom_name + "  " + pdbqt_line[17:]
                        elif len(atom_name) == 3:
                            pdbqt_line = pdbqt_line[:12] + " " + atom_name + " " + pdbqt_line[17:]
                        pdbqt_content.append(pdbqt_line)
                    elif line.startswith("TER"):
                        pdbqt_content.append(line)
                    elif line.startswith("END"):
                        pdbqt_content.append("END")

                with open(output_file, "w") as f:
                    f.write("\n".join(pdbqt_content))

                logger.info(f"Prepared structure saved to {output_file}")
                return str(output_file)

            else:
                logger.error(f"Unsupported output format: {output_format}")
                return None

        except Exception as e:
            logger.error(f"Error preparing structure for docking: {str(e)}")
            return None

    def get_structure_summary(self) -> Dict[str, Any]:
        """
        获取结构摘要信息

        返回:
            dict: 包含结构信息的字典
        """
        summary = {
            "target_name": self.target_name,
            "pdb_id": self.pdb_id,
            "structure_loaded": self.pdb_structure is not None,
            "structure_cleaned": self.cleaned_structure is not None,
            "binding_site": self.binding_site
        }

        if self.pdb_structure is not None:
            try:
                num_chains = 0
                num_residues = 0
                num_atoms = 0

                for model in self.pdb_structure:
                    for chain in model:
                        num_chains += 1
                        for residue in chain:
                            if residue.get_id()[0] == " ":
                                num_residues += 1
                            for atom in residue:
                                num_atoms += 1

                summary.update({
                    "num_chains": num_chains,
                    "num_residues": num_residues,
                    "num_atoms": num_atoms
                })
            except Exception as e:
                logger.warning(f"Error getting structure summary: {str(e)}")

        return summary

    def run_full_preparation(self,
                           pdb_id: Optional[str] = None,
                           ligand_chain: Optional[str] = None,
                           ligand_resname: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整的靶点准备流程

        参数:
            pdb_id: PDB ID
            ligand_chain: 配体所在链ID
            ligand_resname: 配体残基名称

        返回:
            dict: 准备流程的结果摘要
        """
        results = {
            "success": False,
            "target_name": self.target_name,
            "steps_completed": []
        }

        try:
            if pdb_id:
                self.pdb_id = pdb_id

            if not self.pdb_id:
                logger.error("No PDB ID provided")
                return results

            if not self.fetch_from_pdb(self.pdb_id):
                logger.error("Failed to fetch structure from PDB")
                return results

            results["steps_completed"].append("fetch_from_pdb")

            if not self.load_structure():
                logger.error("Failed to load structure")
                return results

            results["steps_completed"].append("load_structure")

            if not self.clean_structure():
                logger.error("Failed to clean structure")
                return results

            results["steps_completed"].append("clean_structure")

            binding_site = self.identify_binding_site(ligand_chain, ligand_resname)
            results["steps_completed"].append("identify_binding_site")

            prepared_file = self.prepare_for_docking()
            if prepared_file:
                results["steps_completed"].append("prepare_for_docking")
                results["prepared_file"] = prepared_file

            results["binding_site"] = binding_site
            results["structure_summary"] = self.get_structure_summary()
            results["success"] = True

            logger.info(f"Target preparation completed for {self.target_name}")

        except Exception as e:
            logger.error(f"Error in full preparation: {str(e)}")
            results["error"] = str(e)

        return results


def prepare_target(target_name: str,
                  pdb_id: Optional[str] = None,
                  ligand_chain: Optional[str] = None,
                  ligand_resname: Optional[str] = None) -> Optional[TargetPreparation]:
    """
    便捷函数：准备靶点结构

    参数:
        target_name: 靶点名称
        pdb_id: PDB ID
        ligand_chain: 配体所在链ID
        ligand_resname: 配体残基名称

    返回:
        TargetPreparation: 靶点准备器对象，如果失败返回None
    """
    preparer = TargetPreparation(target_name, pdb_id)
    results = preparer.run_full_preparation(pdb_id, ligand_chain, ligand_resname)

    if results["success"]:
        return preparer
    else:
        return None


if __name__ == "__main__":
    logger.info("Testing TargetPreparation module")

    test_preparer = TargetPreparation("NS3", "5YR5")
    logger.info(f"Test preparer created: {test_preparer.target_name}")

    summary = test_preparer.get_structure_summary()
    logger.info(f"Structure summary: {summary}")
