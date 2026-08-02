#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果分析报告模块

功能：
1. 虚拟筛选结果综合分析
2. 机器学习模型性能评估
3. 分子对接结果分析
4. ADMET性质统计
5. 生成符合研究出版要求的Markdown报告
6. 数据可视化

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict

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
    warnings.warn("Matplotlib not available. Visualization will be limited.")

from configs.config import (
    PROJECT_ROOT, RESULTS_DIR, LOG_CONFIG,
    REPORT_CONFIG, MODEL_PERFORMANCE_THRESHOLDS,
    BINDING_AFFINITY_THRESHOLD
)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit not available. Molecular visualization will be limited.")

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class ModelPerformanceAnalyzer:
    """
    模型性能分析类

    用于分析机器学习模型的性能指标
    """

    def __init__(self):
        """初始化模型性能分析器"""
        self.results = {}

    def add_model_results(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        添加模型结果

        参数:
            model_name: 模型名称
            metrics: 性能指标字典
        """
        self.results[model_name] = metrics

    def generate_performance_table(self) -> pd.DataFrame:
        """
        生成性能对比表

        返回:
            pd.DataFrame: 性能对比表格
        """
        data = []
        for model_name, metrics in self.results.items():
            row = {"Model": model_name}
            row.update(metrics)
            data.append(row)

        df = pd.DataFrame(data)

        if "auc" in df.columns:
            df = df.sort_values("auc", ascending=False)

        return df

    def check_performance_thresholds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查模型是否达到性能阈值

        参数:
            df: 性能对比表

        返回:
            dict: 阈值检查结果
        """
        thresholds = MODEL_PERFORMANCE_THRESHOLDS

        passed_models = []
        failed_models = []

        for _, row in df.iterrows():
            model_name = row["Model"]
            passed = True
            failed_criteria = []

            if "auc" in row and row["auc"] < thresholds["auc"]:
                passed = False
                failed_criteria.append(f"AUC ({row['auc']:.3f} < {thresholds['auc']})")

            if "sensitivity" in row and row["sensitivity"] < thresholds["sensitivity"]:
                passed = False
                failed_criteria.append(f"Sensitivity ({row['sensitivity']:.3f} < {thresholds['sensitivity']})")

            if "specificity" in row and row["specificity"] < thresholds["specificity"]:
                passed = False
                failed_criteria.append(f"Specificity ({row['specificity']:.3f} < {thresholds['specificity']})")

            if "f1_score" in row and row["f1_score"] < thresholds["f1_score"]:
                passed = False
                failed_criteria.append(f"F1 ({row['f1_score']:.3f} < {thresholds['f1_score']})")

            if passed:
                passed_models.append(model_name)
            else:
                failed_models.append({"model": model_name, "failed": failed_criteria})

        return {
            "passed_models": passed_models,
            "failed_models": failed_models,
            "all_passed": len(failed_models) == 0
        }


class DockingResultsAnalyzer:
    """
    对接结果分析类

    用于分析分子对接结果
    """

    def __init__(self):
        """初始化对接结果分析器"""
        self.results_df = None

    def load_results(self, results_file: str) -> pd.DataFrame:
        """
        加载对接结果

        参数:
            results_file: 结果文件路径

        返回:
            pd.DataFrame: 结果DataFrame
        """
        if results_file.endswith(".csv"):
            self.results_df = pd.read_csv(results_file)
        else:
            logger.error(f"Unsupported file format: {results_file}")
            return pd.DataFrame()

        return self.results_df

    def analyze_binding_affinity(self) -> Dict[str, Any]:
        """
        分析结合亲和力

        返回:
            dict: 结合亲和力分析结果
        """
        if self.results_df is None or self.results_df.empty:
            return {}

        affinity_col = "best_affinity"
        if affinity_col not in self.results_df.columns:
            return {}

        stats = {
            "mean": float(self.results_df[affinity_col].mean()),
            "std": float(self.results_df[affinity_col].std()),
            "min": float(self.results_df[affinity_col].min()),
            "max": float(self.results_df[affinity_col].max()),
            "median": float(self.results_df[affinity_col].median())
        }

        thresholds = BINDING_AFFINITY_THRESHOLD

        categories = {}
        for name, threshold in thresholds.items():
            count = int((self.results_df[affinity_col] <= threshold).sum())
            percentage = count / len(self.results_df) * 100
            categories[name] = {"count": count, "percentage": percentage}

        stats["categories"] = categories

        return stats

    def get_top_compounds(self, top_n: int = 50, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        获取排名最高的化合物

        参数:
            top_n: 返回前n个
            threshold: 可选的结合能阈值

        返回:
            pd.DataFrame: Top化合物DataFrame
        """
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()

        df_sorted = self.results_df.sort_values("best_affinity", ascending=True)

        if threshold is not None:
            df_sorted = df_sorted[df_sorted["best_affinity"] <= threshold]

        return df_sorted.head(top_n)


class ADMETResultsAnalyzer:
    """
    ADMET结果分析类

    用于分析ADMET评估结果
    """

    def __init__(self):
        """初始化ADMET结果分析器"""
        self.results_df = None

    def load_results(self, results_file: str) -> pd.DataFrame:
        """
        加载ADMET结果

        参数:
            results_file: 结果文件路径

        返回:
            pd.DataFrame: 结果DataFrame
        """
        self.results_df = pd.read_csv(results_file)
        return self.results_df

    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        生成ADMET统计摘要

        返回:
            dict: 统计摘要
        """
        if self.results_df is None or self.results_df.empty:
            return {}

        summary = {
            "total_compounds": len(self.results_df),
            "valid_structures": int(self.results_df["valid_structure"].sum()) if "valid_structure" in self.results_df.columns else len(self.results_df)
        }

        categorical_columns = [
            "hia", "oral_bioavailability", "cyp3a4_inhibition",
            "ames_toxicity", "herg_inhibition", "solubility", "bbb_penetration"
        ]

        for col in categorical_columns:
            if col in self.results_df.columns:
                value_counts = self.results_df[col].value_counts().to_dict()
                summary[col] = value_counts

        return summary

    def calculate_drug_likeness_score(self) -> pd.DataFrame:
        """
        计算类药性评分

        返回:
            pd.DataFrame: 包含类药性评分的DataFrame
        """
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()

        df = self.results_df.copy()

        df["drug_likeness_score"] = 0

        if "hia" in df.columns:
            df.loc[df["hia"] == "High", "drug_likeness_score"] += 2
            df.loc[df["hia"] == "Medium", "drug_likeness_score"] += 1

        if "cyp3a4_inhibition" in df.columns:
            df.loc[df["cyp3a4_inhibition"] == "Low Risk", "drug_likeness_score"] += 2
            df.loc[df["cyp3a4_inhibition"] == "Medium Risk", "drug_likeness_score"] += 1

        if "ames_toxicity" in df.columns:
            df.loc[df["ames_toxicity"] == "Non-Mutagenic", "drug_likeness_score"] += 2
            df.loc[df["ames_toxicity"] == "Potentially Mutagenic", "drug_likeness_score"] += 1

        if "herg_inhibition" in df.columns:
            df.loc[df["herg_inhibition"] == "Low Risk", "drug_likeness_score"] += 2
            df.loc[df["herg_inhibition"] == "Medium Risk", "drug_likeness_score"] += 1

        if "solubility" in df.columns:
            df.loc[df["solubility"] == "High Solubility", "drug_likeness_score"] += 2
            df.loc[df["solubility"] == "Medium Solubility", "drug_likeness_score"] += 1

        return df

    def filter_drug_like(self, min_score: int = 6) -> pd.DataFrame:
        """
        过滤类药性化合物

        参数:
            min_score: 最小类药性评分

        return: pd.DataFrame: 过滤后的DataFrame
        """
        df_scored = self.calculate_drug_likeness_score()

        if "drug_likeness_score" not in df_scored.columns:
            return df_scored

        return df_scored[df_scored["drug_likeness_score"] >= min_score]


class ReportGenerator:
    """
    报告生成类

    用于生成符合研究出版要求的结果报告
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化报告生成器

        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir or RESULTS_DIR / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_content = []
        self.figures = []

    def add_title(self, title: str, level: int = 1) -> None:
        """
        添加标题

        参数:
            title: 标题文本
            level: 标题级别
        """
        prefix = "#" * level
        self.report_content.append(f"\n{prefix} {title}\n")

    def add_text(self, text: str) -> None:
        """
        添加文本

        参数:
            text: 文本内容
        """
        self.report_content.append(f"{text}\n")

    def add_table(self, df: pd.DataFrame, caption: Optional[str] = None) -> None:
        """
        添加表格

        参数:
            df: 表格数据
            caption: 表格标题
        """
        if caption:
            self.report_content.append(f"\n**{caption}**\n")

        table_str = df.to_markdown(index=False)
        self.report_content.append(table_str)
        self.report_content.append("\n")

    def add_code_block(self, code: str, language: str = "python") -> None:
        """
        添加代码块

        参数:
            code: 代码内容
            language: 编程语言
        """
        self.report_content.append(f"\n```{language}\n{code}\n```\n")

    def add_list(self, items: List[str], ordered: bool = False) -> None:
        """
        添加列表

        参数:
            items: 列表项
            ordered: 是否为有序列表
        """
        for i, item in enumerate(items):
            if ordered:
                self.report_content.append(f"{i+1}. {item}")
            else:
                self.report_content.append(f"- {item}")
        self.report_content.append("\n")

    def add_figure(self, figure_path: str, caption: Optional[str] = None) -> None:
        """
        添加图片

        参数:
            figure_path: 图片路径
            caption: 图片标题
        """
        if caption:
            self.report_content.append(f"\n![{caption}]({figure_path})\n")
        else:
            self.report_content.append(f"\n![]({figure_path})\n")

        self.figures.append(figure_path)

    def save_report(self, filename: str, format: str = "markdown") -> str:
        """
        保存报告

        参数:
            filename: 文件名
            format: 报告格式

        返回:
            str: 报告文件路径
        """
        if format == "markdown":
            output_path = self.output_dir / filename
            if not output_path.suffix:
                output_path = output_path.with_suffix(".md")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.report_content))

            logger.info(f"Report saved to {output_path}")
            return str(output_path)

        else:
            logger.error(f"Unsupported format: {format}")
            return ""


class VirtualScreeningReporter:
    """
    虚拟筛选结果报告生成类

    整合所有分析结果，生成完整的研究报告
    """

    def __init__(self, target_name: str, output_dir: Optional[Path] = None):
        """
        初始化虚拟筛选报告生成器

        参数:
            target_name: 靶点名称
            output_dir: 输出目录
        """
        self.target_name = target_name
        self.output_dir = output_dir or RESULTS_DIR / "reports" / target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_generator = ReportGenerator(self.output_dir)

        self.model_analyzer = ModelPerformanceAnalyzer()
        self.docking_analyzer = DockingResultsAnalyzer()
        self.admet_analyzer = ADMETResultsAnalyzer()

        self.pipeline_results = {}
        self.compound_info = {}  # 存储化合物SMILES信息

    def add_compound_info(self, compound_info: Dict[str, str]) -> None:
        """
        添加化合物信息（SMILES等）

        参数:
            compound_info: 化合物信息字典，key为化合物ID，value为SMILES
        """
        self.compound_info.update(compound_info)

    def draw_molecule_structure(self, smiles: str, filename: str, size: Tuple[int, int] = (300, 300)) -> Optional[str]:
        """
        绘制分子结构图片

        参数:
            smiles: SMILES字符串
            filename: 输出文件名
            size: 图片大小

        返回:
            str: 图片路径，如果失败返回None
        """
        if not HAS_RDKIT:
            logger.warning("RDKit not available, cannot draw molecule structure")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None

            img_path = self.output_dir / filename
            img = Draw.MolToImage(mol, size=size)
            img.save(img_path)

            logger.info(f"Molecule structure saved to {img_path}")
            return str(img_path)
        except Exception as e:
            logger.error(f"Error drawing molecule: {str(e)}")
            return None

    def draw_top_compounds_grid(self, top_compounds: pd.DataFrame, smiles_col: str = "SMILES", 
                                title_col: str = "CID", top_n: int = 9) -> Optional[str]:
        """
        绘制Top化合物的网格图

        参数:
            top_compounds: Top化合物DataFrame
            smiles_col: SMILES列名
            title_col: 标题列名
            top_n: 显示前N个化合物（应为完全平方数）

        返回:
            str: 图片路径，如果失败返回None
        """
        if not HAS_RDKIT:
            logger.warning("RDKit not available, cannot draw molecule grid")
            return None

        try:
            mols = []
            legends = []
            
            for _, row in top_compounds.head(top_n).iterrows():
                smiles = row.get(smiles_col, "")
                title = str(row.get(title_col, f"Compound {len(mols)+1}"))
                
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
                    legends.append(title)

            if not mols:
                logger.warning("No valid molecules to draw")
                return None

            # 计算网格大小
            grid_size = int(len(mols) ** 0.5)
            if grid_size * grid_size < len(mols):
                grid_size += 1

            img_path = self.output_dir / "top_compounds_grid.png"
            img = Draw.MolsToGridImage(mols, molsPerRow=grid_size, subImgSize=(300, 300), legends=legends)
            
            # 使用正确的保存方式
            img.save(str(img_path))

            logger.info(f"Compound grid saved to {img_path}")
            return str(img_path)
        except Exception as e:
            logger.error(f"Error drawing compound grid: {str(e)}")
            return None

    def generate_summary_report(self,
                             pipeline_results: Dict[str, Any],
                             model_performance: Optional[Dict[str, Dict]] = None,
                             admet_results: Optional[pd.DataFrame] = None) -> str:
        """
        生成虚拟筛选摘要报告

        参数:
            pipeline_results: 流程结果字典
            model_performance: 模型性能字典
            admet_results: ADMET结果DataFrame

        返回:
            str: 报告文件路径
        """
        reporter = self.report_generator

        reporter.add_title(f"Virtual Screening Report: {self.target_name}", level=1)

        reporter.add_text(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        reporter.add_text(f"**Target:** {self.target_name}")

        reporter.add_title("Executive Summary", level=2)

        if model_performance:
            best_model = max(model_performance.items(), key=lambda x: x[1].get("auc", 0))
            reporter.add_text(f"- **Best Model:** {best_model[0]} (AUC: {best_model[1].get('auc', 0):.4f})")

        if admet_results is not None:
            reporter.add_text(f"- **Compounds Evaluated:** {len(admet_results)}")
            if "overall_assessment" in admet_results.columns:
                drug_like = admet_results[admet_results["overall_assessment"].str.contains("Good", na=False)]
                reporter.add_text(f"- **Drug-like Compounds:** {len(drug_like)}")

        reporter.add_title("Methods", level=2)

        reporter.add_title("1. Target Preparation", level=3)
        reporter.add_text("Target protein structure was obtained from PDB database and prepared using standard protocols including:")
        reporter.add_list([
            "Removal of water molecules and heteroatoms",
            "Addition of hydrogen atoms",
            "Protonation state assignment",
            "Energy minimization"
        ])

        reporter.add_title("2. Compound Library", level=3)
        reporter.add_text("Compound library was preprocessed to ensure drug-likeness:")
        reporter.add_list([
            "SMILES standardization",
            "Salt removal",
            "Deduplication based on canonical SMILES",
            "Lipinski Rule of 5 filtering"
        ])

        reporter.add_title("3. Machine Learning Screening", level=3)
        reporter.add_text("Multiple machine learning models were trained for virtual screening:")
        reporter.add_list([
            "XGBoost (Gradient Boosting)",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "Logistic Regression"
        ])
        reporter.add_text("Models were evaluated using 5-fold stratified cross-validation.")

        reporter.add_title("4. Molecular Docking", level=3)
        reporter.add_text("AutoDock Vina was used for molecular docking simulations to predict binding affinities.")

        reporter.add_title("5. ADMET Prediction", level=3)
        reporter.add_text("ADMET properties were predicted using in-silico methods:")
        reporter.add_list([
            "Absorption (HIA, Caco-2 permeability)",
            "Metabolism (CYP450 inhibition)",
            "Toxicity (AMES, hERG)",
            "Solubility",
            "BBB penetration"
        ])

        reporter.add_title("Results", level=2)

        if model_performance:
            reporter.add_title("Model Performance", level=3)

            df_perf = pd.DataFrame([
                {"Model": name, **metrics} for name, metrics in model_performance.items()
            ])

            if "auc" in df_perf.columns:
                df_perf = df_perf.sort_values("auc", ascending=False)

            reporter.add_table(df_perf, caption="Table 1: Machine Learning Model Performance")

        if admet_results is not None:
            reporter.add_title("ADMET Properties", level=3)

            if "hia" in admet_results.columns:
                hia_dist = admet_results["hia"].value_counts()
                reporter.add_text(f"Human Intestinal Absorption: {hia_dist.to_dict()}")

            if "overall_assessment" in admet_results.columns:
                drug_like_count = admet_results["overall_assessment"].str.contains("Good", na=False).sum()
                reporter.add_text(f"Drug-like Compounds: {drug_like_count}/{len(admet_results)}")

            # 添加分子结构展示
            reporter.add_title("Top Compound Structures", level=3)
            reporter.add_text("The following figure shows the chemical structures of the top 9 compounds identified in this screening:")
            
            # 绘制Top化合物网格图
            if "SMILES" in admet_results.columns:
                grid_path = self.draw_top_compounds_grid(admet_results, smiles_col="SMILES", top_n=9)
                if grid_path:
                    reporter.add_figure(grid_path, caption="Figure 1: Chemical structures of top 9 compounds")
            
            # 添加Top化合物详细表格（包含SMILES）
            if "SMILES" in admet_results.columns:
                # 动态选择存在的列
                available_cols = []
                for col in ["SMILES", "predicted_activity", "hia", "solubility", "overall_assessment", "CID", "Score"]:
                    if col in admet_results.columns:
                        available_cols.append(col)
                if available_cols:
                    top_df = admet_results.head(10)[available_cols]
                    reporter.add_table(top_df, caption="Table 2: Top 10 Compounds with Properties")

        reporter.add_title("Conclusion", level=2)
        reporter.add_text("This virtual screening study identified potential inhibitors for dengue virus {target} protein. "
                        "The multi-stage screening approach combining machine learning, molecular docking, and ADMET prediction "
                        "provides a robust pipeline for antiviral drug discovery.".format(target=self.target_name))

        reporter.add_title("References", level=2)
        reporter.add_list([
            "Lipinski CA, et al. (2001) Adv Drug Deliv Rev.",
            "Morris GM, et al. (2009) J Comput Chem.",
            "Trott O, et al. (2010) J Comput Chem."
        ])

        report_path = reporter.save_report(f"{self.target_name}_virtual_screening_report.md")

        return report_path

    def generate_full_report(self,
                           results_dir: Path,
                           include_figures: bool = True) -> str:
        """
        生成完整报告

        参数:
            results_dir: 结果目录
            include_figures: 是否包含图表

        返回:
            str: 报告文件路径
        """
        reporter = self.report_generator

        reporter.add_title(f"Comprehensive Virtual Screening Report: {self.target_name}", level=1)

        reporter.add_text(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        reporter.add_text(f"**Target:** Dengue Virus {self.target_name}")
        reporter.add_text(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self._load_all_results(results_dir)

        self._generate_methods_section()
        self._generate_model_performance_section()
        self._generate_screening_results_section()
        self._generate_admet_section()
        self._generate_conclusions()

        report_path = reporter.save_report(f"{self.target_name}_full_report.md")

        return report_path

    def _load_all_results(self, results_dir: Path) -> None:
        """加载所有结果文件"""
        model_results_file = results_dir / "models" / "performance_summary.json"
        if model_results_file.exists():
            with open(model_results_file) as f:
                self.model_results = json.load(f)

        admet_file = results_dir / "admet_results.csv"
        if admet_file.exists():
            self.admet_analyzer.load_results(str(admet_file))

        docking_file = results_dir / "docking_results.csv"
        if docking_file.exists():
            self.docking_analyzer.load_results(str(docking_file))

    def _generate_methods_section(self) -> None:
        """生成方法章节"""
        reporter = self.report_generator

        reporter.add_title("Methods", level=2)

        reporter.add_title("1.1 Target Preparation", level=3)
        reporter.add_text("The 3D structure of the dengue virus {target} protein was prepared using the following protocol:".format(target=self.target_name))
        reporter.add_list([
            "Structure acquisition from RCSB PDB",
            "Removal of crystallographic water molecules",
            "Addition of polar hydrogen atoms",
            "Protonation state optimization at pH 7.4",
            "Energy minimization using UFF force field"
        ])

        reporter.add_title("1.2 Compound Library Preparation", level=3)
        reporter.add_text("The compound library was processed to ensure chemical integrity and drug-likeness:")
        reporter.add_list([
            "SMILES canonicalization and standardization",
            "Salt and solvent removal",
            "Inorganic molecule filtering",
            "Duplication removal based on InChI keys",
            "Lipinski Rule of 5 compliance check"
        ])

        reporter.add_title("1.3 Molecular Feature Calculation", level=3)
        reporter.add_text("Molecular fingerprints and descriptors were calculated using RDKit:")
        reporter.add_list([
            "Morgan fingerprints (ECFP4, radius=2, 2048 bits)",
            "MACCS structural keys (167 bits)",
            "RDKit molecular descriptors (200+ descriptors)"
        ])

        reporter.add_title("1.4 Machine Learning Models", level=3)
        reporter.add_text("Four classification algorithms were trained to predict compound activity:")
        reporter.add_list([
            "XGBoost: Gradient boosting with 1000 estimators, max_depth=10",
            "Random Forest: 1000 trees, max_depth=15",
            "SVM: RBF kernel with probability calibration",
            "Logistic Regression: L2 regularization"
        ])

    def _generate_model_performance_section(self) -> None:
        """生成模型性能章节"""
        reporter = self.report_generator

        reporter.add_title("2. Model Performance", level=2)

        if hasattr(self, "model_results"):
            df = pd.DataFrame([
                {"Model": name, **metrics} for name, metrics in self.model_results.items()
            ])
            reporter.add_table(df, caption="Model Performance Metrics")

    def _generate_screening_results_section(self) -> None:
        """生成筛选结果章节"""
        reporter = self.report_generator

        reporter.add_title("3. Virtual Screening Results", level=2)

        if hasattr(self, "docking_analyzer") and self.docking_analyzer.results_df is not None:
            stats = self.docking_analyzer.analyze_binding_affinity()
            reporter.add_text(f"**Total Compounds Docked:** {len(self.docking_analyzer.results_df)}")
            reporter.add_text(f"**Mean Binding Affinity:** {stats.get('mean', 0):.2f} kcal/mol")
            reporter.add_text(f"**Best Binding Affinity:** {stats.get('min', 0):.2f} kcal/mol")

    def _generate_admet_section(self) -> None:
        """生成ADMET章节"""
        reporter = self.report_generator

        reporter.add_title("4. ADMET Properties", level=2)

        if hasattr(self, "admet_analyzer") and self.admet_analyzer.results_df is not None:
            summary = self.admet_analyzer.generate_summary_statistics()
            reporter.add_text(f"**Total Compounds Evaluated:** {summary.get('total_compounds', 0)}")

    def _generate_conclusions(self) -> None:
        """生成结论章节"""
        reporter = self.report_generator

        reporter.add_title("5. Conclusions", level=2)
        reporter.add_text("This study employed a comprehensive virtual screening approach to identify potential dengue virus {target} inhibitors. "
                         "The multi-tiered screening strategy combining machine learning classification, molecular docking, "
                         "and ADMET prediction effectively prioritized compounds with favorable pharmacological properties.".format(target=self.target_name))


def generate_publication_report(target_name: str,
                               results_dir: str,
                               output_file: Optional[str] = None) -> str:
    """
    便捷函数：生成符合出版要求的报告

    参数:
        target_name: 靶点名称
        results_dir: 结果目录
        output_file: 输出文件名

    返回:
        str: 报告文件路径
    """
    reporter = VirtualScreeningReporter(target_name, Path(results_dir).parent / "reports")
    report_path = reporter.generate_full_report(Path(results_dir))

    return report_path


if __name__ == "__main__":
    logger.info("Testing Result Analysis module")

    analyzer = VirtualScreeningReporter("NS2A", RESULTS_DIR / "test")
    logger.info(f"Created VirtualScreeningReporter for {analyzer.target_name}")
