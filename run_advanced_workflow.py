#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
登革病毒抑制剂虚拟筛选和分子对接工作流 - 进阶版

核心升级功能：
1. 多维度特征融合（Morgan指纹 + 2D物理化学描述符）
2. SMOTE处理数据不平衡
3. 适用域(Applicability Domain)检查
4. 全面评估指标（MCC、PR-AUC）
5. 构象搜索与并行分子对接
"""

import os
import sys
import argparse
import json
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

print("=" * 70)
print("登革病毒抑制剂虚拟筛选和分子对接工作流 - 进阶版")
print("=" * 70)
print("Python version:", sys.version.split()[0])

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize as MolStandardize
    HAS_RDKIT = True
    print("RDKit: OK")
except ImportError as e:
    HAS_RDKIT = False
    print(f"RDKit Error: {e}")
    sys.exit(1)

# 导入自定义模块
from src.compound_library import CompoundLibrary
from src.molecular_features import FeatureEngineering
from src.ml_screening import VirtualScreening
from src.admet_evaluation import ADMETCalculator
from src.result_analysis import VirtualScreeningReporter
from src.molecular_docking import AutoDockVina

# ============================================
# 全局变量用于并行对接
# ============================================
GLOBAL_DOCKING_CONFIG = {
    'vina_executable': None,
    'receptor_file': None,
    'exhaustiveness': 8,
    'docking_timeout': 600,
    'search_space': {
        'center_x': 25.0,
        'center_y': 162.0,
        'center_z': 25.0,
        'size_x': 25.0,
        'size_y': 25.0,
        'size_z': 25.0
    }
}

# ============================================
# 进阶功能：SMOTE数据平衡
# ============================================
def apply_smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
        k = min(k_neighbors, max(1, np.sum(y == 1) - 1))
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"  SMOTE完成: 原始 {len(y)} -> 平衡后 {len(y_resampled)}")
        print(f"  活性比例: {sum(y_resampled == 1)/len(y_resampled):.2%}")
        return X_resampled, y_resampled
    except ImportError:
        print("  警告: imblearn未安装，跳过SMOTE")
        return X, y
    except Exception as e:
        print(f"  SMOTE失败: {e}")
        return X, y

# ============================================
# 进阶功能：适用域检查 (Applicability Domain) - 向量化加速版
# ============================================
def calculate_applicability_domain(train_features: np.ndarray,
                                   test_features: np.ndarray,
                                   threshold: float = 0.6) -> np.ndarray:
    from sklearn.metrics.pairwise import pairwise_distances

    if train_features.ndim == 2 and test_features.ndim == 2:
        train_binary = (train_features > 0).astype(np.float32)
        test_binary = (test_features > 0).astype(np.float32)

        train_norm = np.sum(train_binary, axis=1)
        test_norm = np.sum(test_binary, axis=1)

        in_domain = np.zeros(test_features.shape[0], dtype=bool)
        batch_size = 1000

        for i in range(0, test_features.shape[0], batch_size):
            batch_end = min(i + batch_size, test_features.shape[0])
            batch = test_binary[i:batch_end]

            dot = np.dot(batch, train_binary.T)
            sim_matrix = dot / (test_norm[i:batch_end, None] + train_norm[None, :] - dot + 1e-10)
            max_sims = np.max(sim_matrix, axis=1)
            in_domain[i:batch_end] = max_sims >= threshold

        return in_domain

    distances = pairwise_distances(test_features, train_features, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    threshold_distance = np.percentile(min_distances, 95)
    return min_distances <= threshold_distance

# ============================================
# 进阶功能：多构象生成（ETKDG算法）
# ============================================
def generate_conformations(mol: Chem.Mol, num_confs: int = 10) -> Chem.Mol:
    if mol is None:
        return None

    try:
        mol_h = Chem.AddHs(mol)

        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()

        AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)

        try:
            AllChem.MMFFOptimizeMoleculesConfs(mol_h)
        except AttributeError:
            for conf_id in range(mol_h.GetNumConformers()):
                AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)

        min_energy = float('inf')
        best_conf_id = 0

        for conf_id in range(mol_h.GetNumConformers()):
            try:
                energy = AllChem.MMFFGetMoleculeForceField(
                    mol_h, AllChem.MMFFGetMoleculeProperties(mol_h),
                    confId=conf_id).CalcEnergy()
                if energy < min_energy:
                    min_energy = energy
                    best_conf_id = conf_id
            except:
                continue

        best_mol = Chem.Mol(mol_h, confId=best_conf_id)
        return best_mol

    except Exception as e:
        print(f"  构象生成失败: {e}")
        return mol

# ============================================
# 进阶功能：单化合物对接（全局函数，用于并行）
# ============================================
def dock_single_compound(args):
    idx, smiles, output_dir = args

    try:
        compound_dir = os.path.join(output_dir, f"compound_{idx}")
        os.makedirs(compound_dir, exist_ok=True)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  化合物 {idx} SMILES无效")
            return None

        # 使用AutoDockVina类的prepare_ligand方法生成PDBQT
        ligand_file = os.path.join(compound_dir, "ligand.pdbqt")

        docking = AutoDockVina(
            GLOBAL_DOCKING_CONFIG['vina_executable'],
            GLOBAL_DOCKING_CONFIG['receptor_file']
        )

        # 设置结合位点
        search_space = GLOBAL_DOCKING_CONFIG.get('search_space', {})
        if search_space.get('center_x') is not None:
            docking.config.set_binding_site(
                center_x=search_space['center_x'],
                center_y=search_space['center_y'],
                center_z=search_space['center_z'],
                size_x=search_space.get('size_x', 25.0),
                size_y=search_space.get('size_y', 25.0),
                size_z=search_space.get('size_z', 25.0)
            )

        docking.config.set_exhaustiveness(GLOBAL_DOCKING_CONFIG.get('exhaustiveness', 8))

        # 设置超时
        timeout = GLOBAL_DOCKING_CONFIG.get('docking_timeout', 600)
        docking.config.config['docking_timeout'] = timeout

        # 先制备配体
        if not docking.prepare_ligand(smiles, ligand_file):
            print(f"  化合物 {idx} 配体制备失败")
            return None

        # 验证配体文件
        if not os.path.exists(ligand_file) or os.path.getsize(ligand_file) < 100:
            print(f"  化合物 {idx} 配体文件异常")
            return None

        output_file = os.path.join(compound_dir, "docking_result.pdbqt")
        log_file = os.path.join(compound_dir, "docking.log")

        result = docking.dock(ligand_file, output_file, log_file)

        if result and result.get('best_affinity') is not None:
            result['SMILES'] = smiles
            result['compound_idx'] = idx
            print(f"  化合物 {idx} 对接成功: 结合能={result['best_affinity']:.2f} kcal/mol")
        elif result:
            print(f"  化合物 {idx} 对接完成但无有效结合能")
            return None
        else:
            print(f"  化合物 {idx} 对接失败")
            return None

        return result
    except Exception as e:
        print(f"  化合物 {idx} 对接失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# 进阶功能：并行分子对接
# ============================================
def run_parallel_docking(smiles_list: List[str],
                        receptor_file: str,
                        vina_executable: str,
                        output_dir: str,
                        cpu_cores: int = 4,
                        exhaustiveness: int = 8) -> List[Dict]:
    global GLOBAL_DOCKING_CONFIG
    GLOBAL_DOCKING_CONFIG['vina_executable'] = vina_executable
    GLOBAL_DOCKING_CONFIG['receptor_file'] = receptor_file
    GLOBAL_DOCKING_CONFIG['exhaustiveness'] = exhaustiveness

    print(f"\n启动并行对接，使用 {cpu_cores} 个线程...")
    start_time = time.time()

    args_list = [(i, smiles, output_dir) for i, smiles in enumerate(smiles_list)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
        results = list(executor.map(dock_single_compound, args_list))

    results = [r for r in results if r is not None]

    elapsed_time = time.time() - start_time
    print(f"并行对接完成，耗时: {elapsed_time:.2f}秒")

    return results

# ============================================
# 进阶功能：全面模型评估
# ============================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef, average_precision_score
    )

    metrics = {}

    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        y_prob = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except Exception as e:
        print(f"  指标计算失败: {e}")

    return metrics

# ============================================
# 主工作流函数
# ============================================
def main():
    parser = argparse.ArgumentParser(description='登革病毒NS5抑制剂虚拟筛选 - 进阶版')

    parser.add_argument('--target', type=str, default='NS5', help='靶点名称')
    parser.add_argument('--library', type=str,
                      default='E:/Python/dengue_drug_discovery/src/modeling/pubchem_100k_compounds.csv',
                      help='待筛选化合物库路径')
    parser.add_argument('--training', type=str,
                      default='E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data_cleaned.csv',
                      help='训练数据路径')
    parser.add_argument('--receptor', type=str,
                      default='E:/Python/dengue_drug_discovery/virtual_screening_pipeline/data/target_structures/4V0Q.pdbqt',
                      help='受体文件路径')
    parser.add_argument('--vina', type=str,
                      default='E:/autodock/vina.exe',
                      help='Vina可执行文件路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录')

    parser.add_argument('--top_n', type=int, default=50, help='筛选前N个化合物')
    parser.add_argument('--cpu_cores', type=int, default=8, help='CPU核心数')
    parser.add_argument('--exhaustiveness', type=int, default=8, help='Vina搜索穷举度')
    parser.add_argument('--smote', action='store_true', default=True, help='是否使用SMOTE')
    parser.add_argument('--ad_threshold', type=float, default=0.6, help='适用域相似度阈值')
    parser.add_argument('--prob_threshold', type=float, default=0.5, help='活性概率阈值')

    args = parser.parse_args()

    # 检查Vina可执行文件
    if not os.path.exists(args.vina):
        print(f"\n警告: Vina可执行文件不存在: {args.vina}")
        print("分子对接步骤将被跳过。请使用 --vina 参数指定正确的路径。")
        vina_available = False
    else:
        vina_available = True

    # 检查受体文件
    if not os.path.exists(args.receptor):
        print(f"\n错误: 受体文件不存在: {args.receptor}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output) / f"{args.target}_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print(f"\n进阶参数配置:")
    print(f"  - SMOTE数据平衡: {'启用' if args.smote else '禁用'}")
    print(f"  - 适用域阈值: {args.ad_threshold}")
    print(f"  - CPU核心数: {args.cpu_cores}")
    print(f"  - 搜索穷举度: {args.exhaustiveness}")

    # ============================================
    # Step 1: 加载训练数据
    # ============================================
    print("\n" + "=" * 60)
    print("Step 1: 加载训练数据")
    print("=" * 60)

    training_df = pd.read_csv(args.training)
    print(f"训练数据: {len(training_df)} 条")
    print(f"  活性 (Label=1): {sum(training_df['Label'])} 条")
    print(f"  非活性 (Label=0): {len(training_df) - sum(training_df['Label'])} 条")

    # ============================================
    # Step 2: 特征工程（多维度特征融合）
    # ============================================
    print("\n" + "=" * 60)
    print("Step 2: 多维度特征融合")
    print("=" * 60)

    fe = FeatureEngineering()

    mols_train = []
    labels_train = []

    for smiles, label in zip(training_df['SMILES'], training_df['Label']):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols_train.append(mol)
            labels_train.append(label)

    print(f"有效分子: {len(mols_train)}")

    features_train, _, _ = fe.calculate_all_features(mols_train)
    print(f"特征矩阵: {features_train.shape}")
    print(f"  - 包含指纹和2D描述符（分子量、LogP、TPSA等）")

    X = features_train
    y = np.array(labels_train)

    # ============================================
    # Step 3: SMOTE数据平衡
    # ============================================
    if args.smote:
        print("\n" + "=" * 60)
        print("Step 3: SMOTE数据平衡")
        print("=" * 60)
        X, y = apply_smote(X, y)

    # ============================================
    # Step 4: 模型训练与评估
    # ============================================
    print("\n" + "=" * 60)
    print("Step 4: 模型训练（含全面评估指标）")
    print("=" * 60)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    screening = VirtualScreening()
    train_results = screening.train_models(X_train, y_train, X_val, y_val)

    # 全面评估最佳模型
    best_model = train_results.get('best_model', 'XGBoost')
    metrics = {}
    if best_model in screening.trained_models:
        trainer = screening.trained_models[best_model]
        y_pred = trainer.predict(X_val)
        y_proba = trainer.predict_proba(X_val)

        metrics = calculate_metrics(y_val, y_pred, y_proba)
        print(f"\n{best_model} 模型评估指标:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1: {metrics.get('f1', 0):.4f}")
        print(f"  AUC: {metrics.get('auc', 0):.4f}")
        print(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")
        print(f"  MCC: {metrics.get('mcc', 0):.4f}")

    # ============================================
    # Step 5: 加载并处理化合物库
    # ============================================
    print("\n" + "=" * 60)
    print("Step 5: 加载化合物库")
    print("=" * 60)

    library = CompoundLibrary()
    library.load_from_smiles(args.library, smiles_column='SMILES')
    print(f"加载化合物: {len(library.compounds)}")

    library.deduplicate()
    print(f"去重后: {len(library.compounds)}")

    library.filter_drug_likeness()
    print(f"类药过滤后: {len(library.compounds)}")

    # ============================================
    # Step 6: 虚拟筛选（含适用域检查）
    # ============================================
    print("\n" + "=" * 60)
    print("Step 6: 虚拟筛选（含适用域检查）")
    print("=" * 60)

    mols_screen = [comp['mol'] for comp in library.compounds if 'mol' in comp]
    features_screen, _, _ = fe.calculate_all_features(mols_screen)

    print(f"待筛选特征矩阵: {features_screen.shape}")

    print(f"\n适用域检查（阈值: {args.ad_threshold}）...")
    in_domain = calculate_applicability_domain(X_train, features_screen, threshold=args.ad_threshold)
    print(f"在适用域内的化合物: {sum(in_domain)}/{len(in_domain)}")

    screen_results = screening.screen_compounds(
        features_screen,
        model_name=best_model,
        probability_threshold=args.prob_threshold
    )

    probabilities = np.array(screen_results['probabilities'])
    scores = probabilities[:, 1] if probabilities.ndim > 1 else probabilities

    final_scores = scores.copy()
    final_scores[~in_domain] = 0

    top_indices = np.argsort(final_scores)[::-1][:args.top_n]

    print(f"\n筛选结果:")
    print(f"  预测活性化合物: {sum(scores >= args.prob_threshold)}")
    print(f"  适用域内活性化合物: {sum(in_domain & (scores >= args.prob_threshold))}")
    print(f"  Top {args.top_n} 分数范围: [{final_scores[top_indices].min():.4f}, {final_scores[top_indices].max():.4f}]")

    # ============================================
    # Step 7: ADMET评估
    # ============================================
    print("\n" + "=" * 60)
    print("Step 7: ADMET评估")
    print("=" * 60)

    evaluator = ADMETCalculator()
    admet_results = []

    for idx in top_indices[:20]:
        comp = library.compounds[idx]
        if 'mol' in comp:
            try:
                result = evaluator.calculate_all_admet(comp['mol'])
                result['SMILES'] = Chem.MolToSmiles(comp['mol'])
                result['Score'] = float(final_scores[idx])
                result['InDomain'] = bool(in_domain[idx])
                admet_results.append(result)
            except Exception as e:
                print(f"  ADMET评估失败: {e}")

    print(f"完成 {len(admet_results)} 个化合物的ADMET评估")

    # ============================================
    # Step 8: 并行分子对接
    # ============================================
    print("\n" + "=" * 60)
    print("Step 8: 并行分子对接（ETKDG构象搜索）")
    print("=" * 60)

    docking_results = []
    if vina_available:
        top_smiles = []
        for idx in top_indices[:10]:
            comp = library.compounds[idx]
            if 'mol' in comp:
                top_smiles.append(Chem.MolToSmiles(comp['mol']))

        docking_results = run_parallel_docking(
            top_smiles,
            args.receptor,
            args.vina,
            str(output_dir / "docking_results"),
            cpu_cores=args.cpu_cores,
            exhaustiveness=args.exhaustiveness
        )

        print(f"对接完成: {len(docking_results)} 个化合物")
        if docking_results:
            docking_results.sort(key=lambda x: x.get('best_affinity', float('inf')))
            print("Top 5 对接结果:")
            for i, res in enumerate(docking_results[:5]):
                aff = res.get('best_affinity', 'N/A')
                print(f"  {i+1}. 结合能: {aff} kcal/mol")
    else:
        print("Vina不可用，跳过分子对接")

    # ============================================
    # Step 9: 生成报告
    # ============================================
    print("\n" + "=" * 60)
    print("Step 9: 生成报告")
    print("=" * 60)

    reporter = VirtualScreeningReporter(args.target, output_dir)

    admet_df = pd.DataFrame(admet_results)
    docking_df = pd.DataFrame(docking_results)

    model_performance = {}
    if best_model in screening.trained_models:
        model_performance[best_model] = metrics

    report_path = reporter.generate_summary_report(
        pipeline_results={
            "target": args.target,
            "compounds_screened": len(library.compounds),
            "active_predictions": int(sum(scores >= args.prob_threshold)),
            "in_domain_count": int(sum(in_domain)),
            "top_compounds": args.top_n,
            "model_metrics": metrics,
            "docking_results": docking_results
        },
        model_performance=model_performance,
        admet_results=admet_df
    )

    print(f"报告已生成: {report_path}")

    # ============================================
    # Step 10: 保存结果
    # ============================================
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "smote_used": args.smote,
            "ad_threshold": args.ad_threshold,
            "prob_threshold": args.prob_threshold,
            "cpu_cores": args.cpu_cores
        },
        "screening": {
            "total_compounds": len(library.compounds),
            "active_predictions": int(sum(scores >= args.prob_threshold)),
            "in_domain_active": int(sum(in_domain & (scores >= args.prob_threshold)))
        },
        "model_metrics": metrics,
        "top_compounds": [
            {
                "SMILES": Chem.MolToSmiles(library.compounds[idx]['mol']),
                "score": float(final_scores[idx]),
                "in_domain": bool(in_domain[idx])
            } for idx in top_indices[:20]
        ],
        "docking_results": docking_results
    }

    with open(output_dir / "advanced_screening_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n进阶版工作流完成!")
    print(f"结果目录: {output_dir}")

if __name__ == "__main__":
    main()