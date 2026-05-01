#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接模块

功能：
1. AutoDock Vina批量对接
2. 对接参数配置
3. 对接结果解析
4. 结合口袋分析
5. 相互作用分析

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, PandasTools
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit not available. Some functions will be limited.")

from configs.config import (
    PROJECT_ROOT, RESULTS_DIR, TARGET_STRUCTURES_DIR, LOG_CONFIG,
    DOCKING_CONFIG, BINDING_AFFINITY_THRESHOLD
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class DockingConfig:
    """
    分子对接配置类

    管理对接参数和设置
    """

    def __init__(self, config_dict: Optional[Dict] = None):
        """
        初始化对接配置

        参数:
            config_dict: 配置字典，如果为None使用默认配置
        """
        self.config = config_dict or DOCKING_CONFIG.copy()

    def set_binding_site(self,
                        center_x: float,
                        center_y: float,
                        center_z: float,
                        size_x: float = 22.5,
                        size_y: float = 22.5,
                        size_z: float = 22.5) -> None:
        """
        设置对接结合位点

        参数:
            center_x, center_y, center_z: 结合位点中心坐标
            size_x, size_y, size_z: 对接盒子大小
        """
        self.config["search_space"] = {
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z
        }

    def set_exhaustiveness(self, exhaustiveness: int) -> None:
        """
        设置穷尽性参数

        参数:
            exhaustiveness: 穷尽性参数（计算量因子）
        """
        self.config["exhaustiveness"] = exhaustiveness

    def set_num_poses(self, num_poses: int) -> None:
        """
        设置生成对接构象数量

        参数:
            num_poses: 每个分子生成的构象数量
        """
        self.config["num_poses"] = num_poses

    def to_dict(self) -> Dict:
        """返回配置字典"""
        return self.config.copy()


class AutoDockVina:
    """
    AutoDock Vina对接类

    用于与AutoDock Vina对接软件交互
    """

    def __init__(self,
                vina_executable: Optional[str] = None,
                receptor_file: Optional[str] = None,
                config: Optional[DockingConfig] = None):
        """
        初始化AutoDock Vina对接器

        参数:
            vina_executable: Vina可执行文件路径
            receptor_file: 受体蛋白文件路径
            config: 对接配置对象
        """
        self.vina_executable = vina_executable or self._find_vina_executable()
        self.receptor_file = receptor_file
        self.config = config or DockingConfig()

        if self.vina_executable is None:
            logger.warning("AutoDock Vina not found. Docking functionality will be limited.")

    def _find_vina_executable(self) -> Optional[str]:
        """查找Vina可执行文件"""
        possible_paths = [
            "vina",
            "vina.exe",
            "C:/Program Files/AutoDock-Vina/vina.exe",
            "C:/Program Files (x86)/AutoDock-Vina/vina.exe",
            os.path.expanduser("~/vina/vina.exe"),
            os.path.expanduser("~/autodock_vina_1_1_2/bin/vina.exe")
        ]

        for path in possible_paths:
            try:
                result = subprocess.run([path, "--version"],
                                      capture_output=True,
                                      text=True,
                                      timeout=5)
                if result.returncode == 0:
                    logger.info(f"Found AutoDock Vina at: {path}")
                    return path
            except:
                continue

        return None

    def prepare_ligand(self,
                      smiles: str,
                      output_file: str,
                      random_seed: int = 42) -> bool:
        """
        准备配体文件（SMILES转PDBQT）

        参数:
            smiles: 配体SMILES
            output_file: 输出PDBQT文件路径
            random_seed: 随机种子

        返回:
            bool: 准备是否成功
        """
        if not HAS_RDKIT:
            logger.error("RDKit not available")
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return False

            mol = Chem.AddHs(mol)

            try:
                params = AllChem.ETKDG(randomSeed=random_seed)
                AllChem.EmbedMolecule(mol, params)
                AllChem.UFFOptimizeMolecule(mol)
            except:
                logger.warning(f"Could not generate 3D conformation for {smiles}")

            writer = Chem.PDBWriter(output_file)
            writer.write(mol)
            writer.close()

            logger.debug(f"Prepared ligand: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error preparing ligand: {str(e)}")
            return False

    def prepare_receptor(self,
                        pdb_file: str,
                        output_file: str) -> bool:
        """
        准备受体文件（PDB转PDBQT）

        参数:
            pdb_file: PDB文件路径
            output_file: 输出PDBQT文件路径

        返回:
            bool: 准备是否成功
        """
        if not os.path.exists(pdb_file):
            logger.error(f"PDB file not found: {pdb_file}")
            return False

        try:
            with open(pdb_file, "r") as f:
                pdb_content = f.read()

            pdbqt_lines = []
            for line in pdb_content.split("\n"):
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    pdbqt_line = line[:66]
                    resname = line[17:20] if len(line) > 20 else ""

                    if resname in ["HOH", "WAT"]:
                        continue

                    atom_name = line[12:16].strip()
                    if len(atom_name) == 1:
                        pdbqt_line = pdbqt_line[:12] + " " + atom_name + "   " + pdbqt_line[17:]
                    elif len(atom_name) == 2:
                        pdbqt_line = pdbqt_line[:12] + " " + atom_name + "  " + pdbqt_line[17:]
                    elif len(atom_name) == 3:
                        pdbqt_line = pdbqt_line[:12] + " " + atom_name + " " + pdbqt_line[17:]

                    pdbqt_line += " 0.00  0.00\n"
                    pdbqt_lines.append(pdbqt_line)
                elif line.startswith("TER"):
                    pdbqt_lines.append("END\n")

            with open(output_file, "w") as f:
                f.write("\n".join(pdbqt_lines))

            logger.info(f"Prepared receptor: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error preparing receptor: {str(e)}")
            return False

    def dock(self,
            ligand_file: str,
            output_file: str,
            log_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        执行分子对接

        参数:
            ligand_file: 配体PDBQT文件
            output_file: 输出文件路径
            log_file: 日志文件路径

        返回:
            dict: 对接结果字典
        """
        if self.vina_executable is None:
            logger.error("AutoDock Vina not available")
            return None

        if self.receptor_file is None:
            logger.error("No receptor file specified")
            return None

        if not os.path.exists(ligand_file):
            logger.error(f"Ligand file not found: {ligand_file}")
            return None

        if not os.path.exists(self.receptor_file):
            logger.error(f"Receptor file not found: {self.receptor_file}")
            return None

        try:
            cmd = [
                self.vina_executable,
                "--receptor", self.receptor_file,
                "--ligand", ligand_file,
                "--out", output_file,
                "--exhaustiveness", str(self.config.config.get("exhaustiveness", 32)),
                "--num_modes", str(self.config.config.get("num_poses", 20)),
                "--energy_range", str(self.config.config.get("energy_range", 3.0))
            ]

            search_space = self.config.config.get("search_space", {})
            if search_space.get("center_x") is not None:
                cmd.extend([
                    "--center_x", str(search_space.get("center_x")),
                    "--center_y", str(search_space.get("center_y")),
                    "--center_z", str(search_space.get("center_z")),
                    "--size_x", str(search_space.get("size_x", 22.5)),
                    "--size_y", str(search_space.get("size_y", 22.5)),
                    "--size_z", str(search_space.get("size_z", 22.5))
                ])

            if log_file:
                cmd.extend(["--log", log_file])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.debug(f"Docking completed: {ligand_file}")
                return self._parse_vina_output(result.stdout)
            else:
                logger.error(f"Docking failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Docking timeout")
            return None
        except Exception as e:
            logger.error(f"Error during docking: {str(e)}")
            return None

    def _parse_vina_output(self, output: str) -> Dict[str, Any]:
        """解析Vina输出"""
        results = {
            "poses": [],
            "best_mode": None,
            "best_affinity": None
        }

        lines = output.split("\n")
        mode_started = False
        current_mode = {}

        for line in lines:
            line = line.strip()

            if line.startswith("-----+"):
                mode_started = True
                continue

            if mode_started and line and not line.startswith("Writing"):
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        mode_num = int(parts[0])
                        affinity = float(parts[1])
                        rmsd_lb = float(parts[2])
                        rmsd_ub = float(parts[3])

                        current_mode = {
                            "mode": mode_num,
                            "affinity": affinity,
                            "rmsd_lb": rmsd_lb,
                            "rmsd_ub": rmsd_ub
                        }

                        results["poses"].append(current_mode)

                        if results["best_affinity"] is None or affinity < results["best_affinity"]:
                            results["best_affinity"] = affinity
                            results["best_mode"] = current_mode

                    except ValueError:
                        continue

        return results

    def batch_dock(self,
                  ligand_files: List[str],
                  output_dir: str,
                  log_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量对接

        参数:
            ligand_files: 配体文件列表
            output_dir: 输出目录
            log_dir: 日志目录

        返回:
            list: 对接结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        docking_results = []

        for i, ligand_file in enumerate(ligand_files):
            ligand_name = Path(ligand_file).stem
            output_file = os.path.join(output_dir, f"{ligand_name}_out.pdbqt")
            log_file = os.path.join(log_dir, f"{ligand_name}.log") if log_dir else None

            logger.info(f"Docking {i+1}/{len(ligand_files)}: {ligand_name}")

            result = self.dock(ligand_file, output_file, log_file)

            if result:
                result["ligand_file"] = ligand_file
                result["output_file"] = output_file
                result["ligand_name"] = ligand_name

            docking_results.append(result)

        return docking_results


class HADDOCKInterface:
    """
    HADDOCK对接接口类

    用于与HADDOCK对接软件交互（需要本地安装HADDOCK）
    """

    def __init__(self,
                haddock_dir: Optional[str] = None,
                receptor_file: Optional[str] = None,
                ligand_file: Optional[str] = None):
        """
        初始化HADDOCK接口

        参数:
            haddock_dir: HADDOCK安装目录
            receptor_file: 受体文件路径
            ligand_file: 配体文件路径
        """
        self.haddock_dir = haddock_dir
        self.receptor_file = receptor_file
        self.ligand_file = ligand_file
        self.project_name = "haddock_project"

    def setup_docking(self,
                     project_dir: str,
                     receptor_file: Optional[str] = None,
                     ligand_file: Optional[str] = None) -> bool:
        """
        设置对接项目

        参数:
            project_dir: 项目目录
            receptor_file: 受体文件
            ligand_file: 配体文件

        返回:
            bool: 设置是否成功
        """
        if receptor_file:
            self.receptor_file = receptor_file
        if ligand_file:
            self.ligand_file = ligand_file

        if not self.receptor_file or not self.ligand_file:
            logger.error("Receptor and ligand files must be specified")
            return False

        try:
            os.makedirs(project_dir, exist_ok=True)

            run_param_file = os.path.join(project_dir, "run.param")
            with open(run_param_file, "w") as f:
                f.write(f"project_dir = {project_dir}\n")
                f.write(f"receptor_file = {self.receptor_file}\n")
                f.write(f"ligand_file = {self.ligand_file}\n")
                f.write(f"project_name = {self.project_name}\n")

            logger.info(f"HADDOCK project setup in {project_dir}")
            return True

        except Exception as e:
            logger.error(f"Error setting up HADDOCK project: {str(e)}")
            return False

    def run_docking(self,
                   project_dir: str,
                   cpu_cores: int = -1) -> bool:
        """
        运行HADDOCK对接

        参数:
            project_dir: 项目目录
            cpu_cores: CPU核心数，-1表示全部

        返回:
            bool: 运行是否成功
        """
        if self.haddock_dir is None:
            logger.error("HADDOCK directory not specified")
            return False

        try:
            haddock_run_script = os.path.join(self.haddock_dir, "run.csh")

            if not os.path.exists(haddock_run_script):
                logger.error(f"HADDOCK run script not found: {haddock_run_script}")
                return False

            cmd = [haddock_run_script, project_dir]

            if cpu_cores > 0:
                cmd.extend(["-cpu", str(cpu_cores)])

            subprocess.run(cmd, cwd=project_dir, check=True)

            logger.info(f"HADDOCK docking completed in {project_dir}")
            return True

        except Exception as e:
            logger.error(f"Error running HADDOCK: {str(e)}")
            return False


class DockingResults:
    """
    对接结果分析类

    用于解析、分析和过滤对接结果
    """

    def __init__(self, results: Optional[List[Dict]] = None):
        """
        初始化对接结果分析器

        参数:
            results: 对接结果列表
        """
        self.results = results or []
        self.df = None

    def add_results(self, results: List[Dict]) -> None:
        """
        添加对接结果

        参数:
            results: 对接结果列表
        """
        self.results.extend(results)
        self.df = None

    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为DataFrame

        返回:
            pd.DataFrame: 对接结果DataFrame
        """
        if not self.results:
            return pd.DataFrame()

        data = []
        for result in self.results:
            if result is None:
                continue

            row = {
                "ligand_name": result.get("ligand_name", ""),
                "ligand_file": result.get("ligand_file", ""),
                "best_affinity": result.get("best_affinity"),
                "num_poses": len(result.get("poses", []))
            }

            poses = result.get("poses", [])
            if poses:
                row["mode_1_affinity"] = poses[0].get("affinity")
                row["mode_2_affinity"] = poses[1].get("affinity") if len(poses) > 1 else None
                row["mode_3_affinity"] = poses[2].get("affinity") if len(poses) > 2 else None

            data.append(row)

        self.df = pd.DataFrame(data)
        return self.df

    def filter_by_affinity(self, threshold: float = -7.0) -> pd.DataFrame:
        """
        按结合能过滤

        参数:
            threshold: 结合能阈值（kcal/mol）

        返回:
            pd.DataFrame: 过滤后的结果
        """
        if self.df is None:
            self.to_dataframe()

        return self.df[self.df["best_affinity"] <= threshold].copy()

    def get_top_compounds(self, top_n: int = 100, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        获取排名最高的化合物

        参数:
            top_n: 返回前n个
            threshold: 可选的结合能阈值

        返回:
            pd.DataFrame: 排名最高的化合物
        """
        if self.df is None:
            self.to_dataframe()

        if threshold is not None:
            df_filtered = self.df[self.df["best_affinity"] <= threshold].copy()
        else:
            df_filtered = self.df.copy()

        df_sorted = df_filtered.sort_values("best_affinity", ascending=True)

        return df_sorted.head(top_n)

    def summarize(self) -> Dict[str, Any]:
        """
        生成结果摘要

        返回:
            dict: 结果摘要
        """
        if self.df is None:
            self.to_dataframe()

        if self.df.empty:
            return {"total_compounds": 0}

        summary = {
            "total_compounds": len(self.df),
            "compounds_with_poses": int(self.df["num_poses"].sum()),
            "binding_affinity": {
                "mean": float(self.df["best_affinity"].mean()),
                "std": float(self.df["best_affinity"].std()),
                "min": float(self.df["best_affinity"].min()),
                "max": float(self.df["best_affinity"].max())
            }
        }

        strong_threshold = BINDING_AFFINITY_THRESHOLD["strong"]
        moderate_threshold = BINDING_AFFINITY_THRESHOLD["moderate"]
        weak_threshold = BINDING_AFFINITY_THRESHOLD["weak"]

        summary["binding_categories"] = {
            "strong": int((self.df["best_affinity"] <= strong_threshold).sum()),
            "moderate": int(((self.df["best_affinity"] > strong_threshold) &
                           (self.df["best_affinity"] <= moderate_threshold)).sum()),
            "weak": int(((self.df["best_affinity"] > moderate_threshold) &
                       (self.df["best_affinity"] <= weak_threshold)).sum()),
            "negligible": int((self.df["best_affinity"] > weak_threshold).sum())
        }

        return summary


class MolecularDocking:
    """
    分子对接主类

    整合Vina和HADDOCK对接功能
    """

    def __init__(self,
                target_name: str,
                receptor_file: Optional[str] = None,
                vina_executable: Optional[str] = None):
        """
        初始化分子对接器

        参数:
            target_name: 靶点名称
            receptor_file: 受体蛋白文件
            vina_executable: Vina可执行文件路径
        """
        self.target_name = target_name
        self.receptor_file = receptor_file

        self.vina = AutoDockVina(vina_executable, receptor_file)
        self.results = DockingResults()

        self.output_dir = RESULTS_DIR / "docking" / target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized MolecularDocking for {target_name}")

    def set_binding_site(self,
                        center_x: float,
                        center_y: float,
                        center_z: float,
                        size_x: float = 22.5,
                        size_y: float = 22.5,
                        size_z: float = 22.5) -> None:
        """
        设置结合位点

        参数:
            center_x, center_y, center_z: 结合位点中心坐标
            size_x, size_y, size_z: 对接盒子大小
        """
        config = DockingConfig()
        config.set_binding_site(center_x, center_y, center_z, size_x, size_y, size_z)
        self.vina.config = config

        logger.info(f"Binding site set to center=({center_x}, {center_y}, {center_z})")

    def dock_compounds(self,
                      smiles_list: List[str],
                      compound_names: Optional[List[str]] = None,
                      batch_size: int = 100) -> DockingResults:
        """
        对化合物列表进行对接

        参数:
            smiles_list: SMILES列表
            compound_names: 化合物名称列表
            batch_size: 批处理大小

        返回:
            DockingResults: 对接结果对象
        """
        if compound_names is None:
            compound_names = [f"compound_{i}" for i in range(len(smiles_list))]

        temp_ligand_dir = self.output_dir / "ligands"
        temp_output_dir = self.output_dir / "outputs"
        temp_ligand_dir.mkdir(parents=True, exist_ok=True)
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        docking_results = []

        for i, (smiles, name) in enumerate(zip(smiles_list, compound_names)):
            logger.info(f"Docking {i+1}/{len(smiles_list)}: {name}")

            ligand_file = temp_ligand_dir / f"{name}.pdbqt"
            output_file = temp_output_dir / f"{name}_out.pdbqt"

            if not self.vina.prepare_ligand(smiles, str(ligand_file)):
                logger.warning(f"Failed to prepare ligand: {name}")
                continue

            result = self.vina.dock(str(ligand_file), str(output_file))

            if result:
                result["ligand_name"] = name
                result["smiles"] = smiles
                docking_results.append(result)

        self.results.add_results(docking_results)

        logger.info(f"Docking completed: {len(docking_results)}/{len(smiles_list)} successful")

        return self.results

    def save_results(self, output_file: Optional[str] = None) -> bool:
        """
        保存对接结果

        参数:
            output_file: 输出文件路径

        返回:
            bool: 保存是否成功
        """
        if output_file is None:
            output_file = self.output_dir / f"{self.target_name}_docking_results.csv"

        try:
            df = self.results.to_dataframe()
            df.to_csv(output_file, index=False)

            summary = self.results.summarize()
            summary_file = str(output_file).replace(".csv", "_summary.json")

            import json
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Results saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False


def run_docking_pipeline(smiles_list: List[str],
                        compound_names: List[str],
                        target_name: str,
                        receptor_file: str,
                        binding_site: Dict[str, float],
                        output_dir: Optional[Path] = None,
                        threshold: float = -7.0) -> Dict[str, Any]:
    """
    运行完整的分子对接流程

    参数:
        smiles_list: SMILES列表
        compound_names: 化合物名称列表
        target_name: 靶点名称
        receptor_file: 受体蛋白文件
        binding_site: 结合位点信息
        output_dir: 输出目录
        threshold: 结合能阈值

    返回:
        dict: 对接结果
    """
    results = {
        "target_name": target_name,
        "total_compounds": len(smiles_list),
        "successful_docking": 0,
        "top_compounds": [],
        "success": False
    }

    try:
        docking = MolecularDocking(target_name, receptor_file)

        if binding_site.get("center_x") is not None:
            docking.set_binding_site(
                binding_site["center_x"],
                binding_site["center_y"],
                binding_site["center_z"],
                binding_site.get("size_x", 22.5),
                binding_site.get("size_y", 22.5),
                binding_site.get("size_z", 22.5)
            )

        docking_results = docking.dock_compounds(smiles_list, compound_names)

        top_compounds = docking_results.get_top_compounds(top_n=100, threshold=threshold)

        results["successful_docking"] = len(docking_results.results)
        results["top_compounds"] = top_compounds.to_dict("records")
        results["summary"] = docking_results.summarize()

        if output_dir:
            docking.save_results(output_dir / f"{target_name}_docking_results.csv")

        results["success"] = True

        logger.info(f"Docking pipeline completed for {target_name}")

    except Exception as e:
        logger.error(f"Error in docking pipeline: {str(e)}")
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    logger.info("Testing MolecularDocking module")

    docking = MolecularDocking("test_target")
    logger.info(f"Created MolecularDocking instance for {docking.target_name}")
