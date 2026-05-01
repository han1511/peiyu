#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟筛选主流程脚本

完整的虚拟筛选流程，整合所有模块：
1. 靶点结构准备
2. 化合物库预处理
3. 分子特征工程
4. 机器学习模型训练
5. 分子对接
6. ADMET评估
7. 结果分析与报告

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    PROJECT_ROOT, RESULTS_DIR, LOG_CONFIG,
    DENGUE_TARGETS, COMPOUND_LIBRARIES,
    DATA_SPLIT_CONFIG, ML_MODELS,
    BINDING_AFFINITY_THRESHOLD
)

from src.target_preparation import TargetPreparation, prepare_target
from src.compound_library import CompoundLibrary, preprocess_compound_library
from src.molecular_features import FeatureDataset, FeatureEngineering
from src.ml_screening import VirtualScreening, ModelTrainer, EnsembleClassifier
from src.molecular_docking import MolecularDocking, DockingConfig
from src.admet_evaluation import ADMETCalculator, ADMETBatchEvaluator

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class VirtualScreeningPipeline:
    """
    虚拟筛选完整流程类

    整合所有筛选步骤，提供端到端的虚拟筛选服务
    """

    def __init__(self,
                target_name: str,
                compound_library_path: Optional[str] = None,
                output_dir: Optional[Path] = None):
        """
        初始化虚拟筛选流程

        参数:
            target_name: 靶点名称（如 'NS2A', 'NS3' 等）
            compound_library_path: 化合物库文件路径
            output_dir: 输出目录
        """
        self.target_name = target_name
        self.compound_library_path = compound_library_path

        self.output_dir = output_dir or RESULTS_DIR / "virtual_screening" / target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_preparer = None
        self.compound_library = None
        self.feature_dataset = None
        self.screening_models = None
        self.docking_engine = None

        self.pipeline_results = {
            "target_name": target_name,
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "success": False
        }

        logger.info(f"Initialized VirtualScreeningPipeline for {target_name}")

    def step1_prepare_target(self,
                            pdb_id: Optional[str] = None,
                            ligand_chain: Optional[str] = None,
                            ligand_resname: Optional[str] = None) -> Dict[str, Any]:
        """
        步骤1：准备靶点结构

        参数:
            pdb_id: PDB ID
            ligand_chain: 配体所在链ID
            ligand_resname: 配体残基名称

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 1: Preparing Target Structure")
        logger.info("=" * 60)

        step_result = {
            "step": "target_preparation",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            target_info = DENGUE_TARGETS.get(self.target_name, {})
            pdb_id = pdb_id or target_info.get("pdb_id")

            if pdb_id is None:
                logger.warning(f"No PDB ID available for {self.target_name}")
                step_result["note"] = "No crystal structure available. Using homology model recommended."
                step_result["success"] = True
                return step_result

            self.target_preparer = TargetPreparation(self.target_name, pdb_id)

            prep_results = self.target_preparer.run_full_preparation(
                pdb_id=pdb_id,
                ligand_chain=ligand_chain,
                ligand_resname=ligand_resname
            )

            step_result.update(prep_results)
            step_result["success"] = prep_results.get("success", False)

            logger.info(f"Target preparation completed: {prep_results.get('success', False)}")

        except Exception as e:
            logger.error(f"Error in target preparation: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["target_preparation"] = step_result

        return step_result

    def step2_load_compound_library(self,
                                   library_path: Optional[str] = None,
                                   smiles_column: str = "SMILES",
                                   name_column: Optional[str] = None) -> Dict[str, Any]:
        """
        步骤2：加载化合物库

        参数:
            library_path: 化合物库路径
            smiles_column: SMILES列名
            name_column: 化合物名称列名

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 2: Loading Compound Library")
        logger.info("=" * 60)

        step_result = {
            "step": "compound_library_loading",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            library_path = library_path or self.compound_library_path

            if library_path is None:
                logger.error("No compound library path provided")
                step_result["error"] = "No library path"
                return step_result

            library_path = Path(library_path)

            if not library_path.exists():
                logger.error(f"Library file not found: {library_path}")
                step_result["error"] = f"File not found: {library_path}"
                return step_result

            self.compound_library = CompoundLibrary(f"{self.target_name}_library")

            if library_path.suffix.lower() in [".smi", ".smiles", ".csv"]:
                count = self.compound_library.load_from_smiles(
                    str(library_path),
                    smiles_column=smiles_column,
                    name_column=name_column
                )
            elif library_path.suffix.lower() == ".sdf":
                count = self.compound_library.load_from_sdf(str(library_path))
            else:
                logger.error(f"Unsupported file format: {library_path.suffix}")
                step_result["error"] = f"Unsupported format: {library_path.suffix}"
                return step_result

            if count > 0:
                step_result["original_count"] = count
                step_result["current_count"] = len(self.compound_library.compounds)
                step_result["success"] = True
                logger.info(f"Loaded {count} compounds from library")
            else:
                step_result["error"] = "No compounds loaded"
                logger.error("No compounds loaded from library")

        except Exception as e:
            logger.error(f"Error loading compound library: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["compound_library_loading"] = step_result

        return step_result

    def step3_preprocess_compounds(self,
                                  deduplicate: bool = True,
                                  apply_filters: bool = True,
                                  generate_3d: bool = False) -> Dict[str, Any]:
        """
        步骤3：预处理化合物库

        参数:
            deduplicate: 是否去重
            apply_filters: 是否应用类药性过滤
            generate_3d: 是否生成3D构象

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 3: Preprocessing Compounds")
        logger.info("=" * 60)

        step_result = {
            "step": "compound_preprocessing",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            if self.compound_library is None:
                logger.error("No compound library loaded")
                step_result["error"] = "No library loaded"
                return step_result

            original_count = len(self.compound_library.compounds)
            step_result["original_count"] = original_count

            if deduplicate:
                removed = self.compound_library.deduplicate()
                step_result["deduplicated"] = removed

            if apply_filters:
                removed = self.compound_library.filter_drug_likeness()
                step_result["filtered"] = removed

            if generate_3d:
                self.compound_library.generate_3d_conformations()

            final_count = len(self.compound_library.compounds)
            step_result["final_count"] = final_count
            step_result["retention_rate"] = final_count / original_count if original_count > 0 else 0

            output_sdf = self.output_dir / f"{self.target_name}_preprocessed.sdf"
            output_csv = self.output_dir / f"{self.target_name}_preprocessed.csv"

            self.compound_library.save_to_sdf(output_sdf)
            self.compound_library.save_to_csv(output_csv)

            step_result["output_files"] = {
                "sdf": str(output_sdf),
                "csv": str(output_csv)
            }

            stats = self.compound_library.get_statistics()
            step_result["statistics"] = stats

            step_result["success"] = True

            logger.info(f"Preprocessing completed: {final_count}/{original_count} compounds retained")

        except Exception as e:
            logger.error(f"Error preprocessing compounds: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["compound_preprocessing"] = step_result

        return step_result

    def step4_generate_features(self,
                               fingerprint_types: Optional[List[str]] = None,
                               descriptor_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        步骤4：生成分子特征

        参数:
            fingerprint_types: 指纹类型列表
            descriptor_types: 描述符类型列表

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 4: Generating Molecular Features")
        logger.info("=" * 60)

        step_result = {
            "step": "feature_generation",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            if self.compound_library is None:
                logger.error("No compound library available")
                step_result["error"] = "No library"
                return step_result

            smiles_list = [c["standardized_smiles"] for c in self.compound_library.compounds]
            names_list = [c["name"] for c in self.compound_library.compounds]

            self.feature_dataset = FeatureDataset(f"{self.target_name}_features")

            success = self.feature_dataset.load_from_smiles(
                smiles_list=smiles_list,
                molecule_names=names_list,
                fingerprint_types=fingerprint_types or ["Morgan", "MACCS"],
                descriptor_types=descriptor_types or ["basic", "electronic", "structural"]
            )

            if not success:
                step_result["error"] = "Feature generation failed"
                return step_result

            stats = self.feature_dataset.get_statistics()
            step_result["statistics"] = stats
            step_result["feature_count"] = len(self.feature_dataset.feature_names)

            feature_file = self.output_dir / f"{self.target_name}_features.npz"
            self.feature_dataset.save(feature_file)
            step_result["feature_file"] = str(feature_file)

            step_result["success"] = True

            logger.info(f"Generated {stats['num_features']} features for {stats['num_molecules']} molecules")

        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["feature_generation"] = step_result

        return step_result

    def step5_train_models(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None,
                         model_names: Optional[List[str]] = None,
                         use_ensemble: bool = True) -> Dict[str, Any]:
        """
        步骤5：训练机器学习模型

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            model_names: 模型名称列表
            use_ensemble: 是否训练集成模型

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 5: Training Machine Learning Models")
        logger.info("=" * 60)

        step_result = {
            "step": "model_training",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            self.screening_models = VirtualScreening()

            train_results = self.screening_models.train_models(
                X_train, y_train, X_val, y_val, model_names
            )

            step_result["model_training"] = train_results

            if use_ensemble:
                ensemble_results = self.screening_models.train_ensemble(
                    X_train, y_train, model_names
                )
                step_result["ensemble_training"] = ensemble_results

            step_result["success"] = True

            logger.info(f"Model training completed. Best model: {train_results.get('best_model')}")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["model_training"] = step_result

        return step_result

    def step6_screen_compounds(self,
                             use_ensemble: bool = True,
                             probability_threshold: float = 0.5,
                             top_n: int = 1000) -> Dict[str, Any]:
        """
        步骤6：筛选化合物

        参数:
            use_ensemble: 是否使用集成模型
            probability_threshold: 概率阈值
            top_n: 返回前n个化合物

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 6: Screening Compounds")
        logger.info("=" * 60)

        step_result = {
            "step": "compound_screening",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            if self.screening_models is None:
                logger.error("Models not trained yet")
                step_result["error"] = "Models not trained"
                return step_result

            if self.feature_dataset is None or self.feature_dataset.features is None:
                logger.error("No features available")
                step_result["error"] = "No features"
                return step_result

            X_screening = self.feature_dataset.features

            screen_results = self.screening_models.screen_compounds(
                X_screening,
                use_ensemble=use_ensemble,
                probability_threshold=probability_threshold
            )

            step_result["screening_results"] = screen_results

            if screen_results.get("probabilities") is not None:
                probs = np.array(screen_results["probabilities"])
                top_indices = np.argsort(probs)[::-1][:top_n]

                top_compounds = []
                for idx in top_indices:
                    compound = {
                        "name": self.feature_dataset.molecule_names[idx],
                        "smiles": self.feature_dataset.molecule_names[idx],
                        "probability": float(probs[idx]),
                        "prediction": "Active" if screen_results["predictions"][idx] == 1 else "Inactive"
                    }
                    top_compounds.append(compound)

                step_result["top_compounds"] = top_compounds

            step_result["success"] = True

            logger.info(f"Screening completed: {screen_results.get('predicted_active', 0)} predicted as active")

        except Exception as e:
            logger.error(f"Error screening compounds: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["compound_screening"] = step_result

        return step_result

    def step7_evaluate_admet(self,
                           smiles_list: Optional[List[str]] = None,
                           compound_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        步骤7：ADMET评估

        参数:
            smiles_list: SMILES列表（如果为None，使用筛选后的化合物）
            compound_names: 化合物名称列表

        返回:
            dict: 步骤结果
        """
        logger.info("=" * 60)
        logger.info("Step 7: ADMET Evaluation")
        logger.info("=" * 60)

        step_result = {
            "step": "admet_evaluation",
            "success": False,
            "start_time": datetime.now().isoformat()
        }

        try:
            if smiles_list is None:
                if self.compound_library:
                    smiles_list = [c["standardized_smiles"] for c in self.compound_library.compounds]
                    compound_names = [c["name"] for c in self.compound_library.compounds]
                else:
                    step_result["error"] = "No compounds available"
                    return step_result

            evaluator = ADMETBatchEvaluator()
            admet_df = evaluator.evaluate_smiles_list(smiles_list, compound_names)

            admet_file = self.output_dir / f"{self.target_name}_admet_results.csv"
            evaluator.save_results(admet_df, str(admet_file))

            stats = evaluator.get_statistics(admet_df)
            step_result["statistics"] = stats
            step_result["output_file"] = str(admet_file)

            filtered_df = evaluator.filter_compounds(admet_df)
            step_result["filtered_count"] = len(filtered_df)

            filtered_file = self.output_dir / f"{self.target_name}_admet_filtered.csv"
            evaluator.save_results(filtered_df, str(filtered_file))
            step_result["filtered_file"] = str(filtered_file)

            step_result["success"] = True

            logger.info(f"ADMET evaluation completed: {len(admet_df)} compounds evaluated")

        except Exception as e:
            logger.error(f"Error in ADMET evaluation: {str(e)}")
            step_result["error"] = str(e)

        step_result["end_time"] = datetime.now().isoformat()
        self.pipeline_results["steps"]["admet_evaluation"] = step_result

        return step_result

    def run_full_pipeline(self,
                         compound_library_path: str,
                         pdb_id: Optional[str] = None,
                         training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                         fingerprint_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行完整虚拟筛选流程

        参数:
            compound_library_path: 化合物库路径
            pdb_id: PDB ID
            training_data: (X_train, y_train) 训练数据
            fingerprint_types: 指纹类型

        返回:
            dict: 完整流程结果
        """
        logger.info("=" * 60)
        logger.info(f"Starting Full Virtual Screening Pipeline for {self.target_name}")
        logger.info("=" * 60)

        try:
            self.step1_prepare_target(pdb_id=pdb_id)

            self.step2_load_compound_library(library_path=compound_library_path)

            self.step3_preprocess_compounds()

            self.step4_generate_features(fingerprint_types=fingerprint_types)

            if training_data is not None:
                X_train, y_train = training_data

                split_result = self.feature_dataset.split_data(
                    train_ratio=DATA_SPLIT_CONFIG["train_ratio"],
                    test_ratio=DATA_SPLIT_CONFIG.get("test_ratio", 1 - DATA_SPLIT_CONFIG["train_ratio"]),
                    stratify=DATA_SPLIT_CONFIG["stratify"],
                    random_state=DATA_SPLIT_CONFIG["random_state"]
                )

                X_train_split = split_result["train_features"]
                y_train_split = split_result["train_labels"]
                X_val_split = split_result["test_features"]
                y_val_split = split_result["test_labels"]

                self.step5_train_models(X_train_split, y_train_split, X_val_split, y_val_split)

                self.step6_screen_compounds()

            self.step7_evaluate_admet()

            self.pipeline_results["success"] = True

            logger.info("=" * 60)
            logger.info("Virtual Screening Pipeline Completed Successfully")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            self.pipeline_results["error"] = str(e)

        self.pipeline_results["end_time"] = datetime.now().isoformat()

        results_file = self.output_dir / f"{self.target_name}_pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump(self.pipeline_results, f, indent=2, default=str)

        logger.info(f"Pipeline results saved to {results_file}")

        return self.pipeline_results


def main():
    """
    主函数

    命令行用法:
    python main_pipeline.py --target NS2A --library path/to/compound_library.csv
    """
    parser = argparse.ArgumentParser(description="Virtual Screening Pipeline for Dengue Virus Inhibitors")

    parser.add_argument("--target", type=str, required=True, help="Target name (e.g., NS2A, NS3)")
    parser.add_argument("--library", type=str, required=True, help="Path to compound library file")
    parser.add_argument("--pdb-id", type=str, help="PDB ID for target structure")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--fingerprint", type=str, nargs="+", default=["Morgan", "MACCS"],
                       help="Fingerprint types to generate")
    parser.add_argument("--ensemble", action="store_true", default=True, help="Use ensemble models")
    parser.add_argument("--top-n", type=int, default=1000, help="Number of top compounds to select")

    args = parser.parse_args()

    pipeline = VirtualScreeningPipeline(
        target_name=args.target,
        compound_library_path=args.library,
        output_dir=Path(args.output) if args.output else None
    )

    results = pipeline.run_full_pipeline(
        compound_library_path=args.library,
        pdb_id=args.pdb_id,
        fingerprint_types=args.fingerprint
    )

    if results.get("success"):
        print("\n" + "=" * 60)
        print("Virtual Screening Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"Results saved to: {pipeline.output_dir}")
    else:
        print("\n" + "=" * 60)
        print("Virtual Screening Pipeline Failed!")
        print("=" * 60)
        if "error" in results:
            print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()
