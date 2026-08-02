#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
登革病毒NS5抑制剂虚拟筛选 - 完整ML流程

使用ChEMBL下载的训练数据进行模型训练和筛选
靶点: NS5 (PDB: 4V0Q, CHEMBL3130)
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

print("=" * 60)
print("登革病毒NS5抑制剂虚拟筛选 - 完整ML流程")
print("=" * 60)
print("Python version:", sys.version.split()[0])

sys.path.append(os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize as MolStandardize
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
    HAS_RDKIT = True
    print("RDKit: OK")
except ImportError as e:
    HAS_RDKIT = False
    print(f"RDKit Error: {e}")
    sys.exit(1)

from src.compound_library import CompoundLibrary
from src.molecular_features import FeatureEngineering
from src.ml_screening import VirtualScreening
from src.admet_evaluation import ADMETCalculator
from src.result_analysis import VirtualScreeningReporter

def load_training_data(training_file):
    """加载训练数据（SMILES + Label）"""
    import pandas as pd
    print(f"\n加载训练数据: {training_file}")
    df = pd.read_csv(training_file)
    print(f"  总数据: {len(df)} 条")
    print(f"  活性 (Label=1): {sum(df['Label'])} 条")
    print(f"  非活性 (Label=0): {len(df) - sum(df['Label'])} 条")
    return df

def train_models_with_data(training_df, target_name):
    """使用训练数据训练ML模型"""
    print("\n" + "=" * 60)
    print("训练机器学习模型")
    print("=" * 60)

    fe = FeatureEngineering()

    smiles_list = training_df['SMILES'].tolist()
    labels = training_df['Label'].values

    print(f"\n生成分子特征 ({len(smiles_list)} 个化合物)...")

    mols = []
    valid_labels = []
    for smiles, label in zip(smiles_list, labels):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_labels.append(label)
        except:
            pass

    print(f"  有效分子: {len(mols)} / {len(smiles_list)}")

    if len(mols) < 10:
        print("  错误: 有效分子数量不足")
        return None

    features, _, _ = fe.calculate_all_features(mols)
    print(f"  特征矩阵: {features.shape}")

    X = features
    y = np.array(valid_labels)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")
    print(f"  训练集活性比例: {sum(y_train)/len(y_train):.2%}")
    print(f"  验证集活性比例: {sum(y_val)/len(y_val):.2%}")

    screening = VirtualScreening()

    print("\n训练多个模型...")
    results = screening.train_models(X_train, y_train, X_val, y_val)

    if results.get("success"):
        print(f"\n模型训练完成!")
        print(f"最佳模型: {results.get('best_model')} (AUC: {results.get('best_auc', 0):.4f})")
    else:
        print("\n模型训练可能有问题，但继续...")

    return {'screening': screening, 'feature_engineering': fe, 'X_val': X_val, 'y_val': y_val, 'train_results': results}

def screen_compounds(library, models_info):
    """使用训练好的模型筛选化合物"""
    screening = models_info['screening']
    feature_engineering = models_info['feature_engineering']

    print("\n" + "=" * 60)
    print("筛选化合物")
    print("=" * 60)

    print(f"\n待筛选化合物: {len(library.compounds)}")

    if len(library.compounds) == 0:
        print("  没有化合物可筛选")
        return None

    mols = [comp['mol'] for comp in library.compounds if 'mol' in comp]
    print(f"  有效分子: {len(mols)}")

    if len(mols) == 0:
        return None

    print("\n生成分子特征...")
    features, _, _ = feature_engineering.calculate_all_features(mols)
    print(f"  特征矩阵: {features.shape}")

    print("\n计算预测分数...")
    try:
        best_model_name = models_info['train_results'].get('best_model', 'RandomForest')
        print(f"使用最佳模型: {best_model_name}")
        
        # 检查模型是否存在
        if best_model_name not in screening.trained_models:
            raise ValueError(f"模型 {best_model_name} 不存在")

        # 检查特征维度是否匹配
        trainer = screening.trained_models[best_model_name]
        if hasattr(trainer, 'scaler') and features.shape[1] != trainer.scaler.mean_.shape[0]:
            raise ValueError(f"特征维度不匹配: 模型期望 {trainer.scaler.mean_.shape[0]} 个特征，实际 {features.shape[1]} 个")

        results = screening.screen_compounds(
            features,
            model_name=best_model_name,
            use_ensemble=False,
            probability_threshold=0.5
        )

        # 检查结果是否有效
        if not results.get('success', False):
            raise ValueError(f"筛选失败: {results.get('error', '未知错误')}")

        predictions = results.get('predictions', [])
        probabilities = results.get('probabilities', [])

        if probabilities is not None and len(probabilities) > 0:
            probs_array = np.array(probabilities)
            if probs_array.ndim > 1 and probs_array.shape[1] > 1:
                scores = probs_array[:, 1]
            else:
                scores = probs_array.astype(float)
        else:
            scores = np.array(predictions).astype(float)

        print(f"  预测完成: {len(scores)} 个化合物")
        print(f"  分数范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print(f"  预测活性: {int(np.sum(scores >= 0.5))}/{len(scores)}")

    except Exception as e:
        print(f"  ❌ 预测出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("  ⚠️ 无法继续筛选，返回None")
        return None

    top_indices = np.argsort(scores)[::-1][:100]

    print(f"\n筛选出预测分数最高的前100个化合物")
    print("Top 10 化合物:")
    for i, idx in enumerate(top_indices[:10]):
        idx_int = int(idx) if not hasattr(idx, '__len__') else int(idx[0])
        score_val = scores[idx_int]
        if hasattr(score_val, '__len__') and len(score_val) > 0:
            score_val = score_val[0]
        comp = library.compounds[idx_int]
        cid = comp.get('CID', 'N/A')
        score = float(score_val)
        print(f"  {i+1}. CID={cid}, Score={score:.4f}")

    top_indices_list = []
    scores_list = []
    for i in top_indices:
        idx_int = int(i) if not hasattr(i, '__len__') else int(i[0])
        score_val = scores[idx_int]
        if hasattr(score_val, '__len__') and len(score_val) > 0:
            score_val = score_val[0]
        top_indices_list.append(idx_int)
        scores_list.append(float(score_val))

    return {
        'top_indices': top_indices_list,
        'scores': scores_list,
        'all_scores': [float(s) if not hasattr(s, '__len__') else float(s[0]) for s in scores]
    }

def main():
    parser = argparse.ArgumentParser(description='登革病毒NS5抑制剂虚拟筛选')
    parser.add_argument('--target', type=str, default='NS5', help='靶点名称')
    parser.add_argument('--library', type=str,
                      default='E:/Python/dengue_drug_discovery/src/modeling/pubchem_100k_compounds.csv',
                      help='待筛选化合物库路径')
    parser.add_argument('--training', type=str,
                      default='E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data_cleaned.csv',
                      help='训练数据路径')
    parser.add_argument('--pdb_id', type=str, default='4V0Q', help='PDB ID')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--top_n', type=int, default=100, help='筛选前N个化合物')

    args = parser.parse_args()

    output_dir = Path(args.output) / f"{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n靶点: {args.target}")
    print(f"PDB ID: {args.pdb_id}")
    print(f"化合物库: {args.library}")
    print(f"训练数据: {args.training}")
    print(f"输出目录: {output_dir}")

    import pandas as pd

    training_df = load_training_data(args.training)

    models_info = train_models_with_data(training_df, args.target)
    if models_info is None:
        print("\n模型训练失败!")
        return

    print("\n" + "=" * 60)
    print("加载和处理待筛选化合物库")
    print("=" * 60)

    library = CompoundLibrary()
    print(f"\n加载化合物库: {args.library}")
    count = library.load_from_smiles(args.library, smiles_column='SMILES')
    print(f"  加载: {count} 个化合物")

    if len(library.compounds) > 0:
        print("\n预处理化合物...")
        removed = library.deduplicate()
        print(f"  去重后: {len(library.compounds)} 个")
        removed = library.filter_drug_likeness()
        print(f"  类药过滤后: {len(library.compounds)} 个")

    if len(library.compounds) == 0:
        print("\n没有化合物可筛选!")
        return

    screening_results = screen_compounds(library, models_info)

    print("\n" + "=" * 60)
    print("ADMET评估")
    print("=" * 60)

    evaluator = ADMETCalculator()
    admet_results = []

    if screening_results and screening_results['top_indices'] is not None:
        top_comps = [library.compounds[i] for i in screening_results['top_indices'][:20]]
    else:
        top_comps = library.compounds[:20]

    print(f"\n评估Top {len(top_comps)} 化合物的ADMET性质...")

    for i, comp in enumerate(top_comps):
        if 'mol' in comp:
            try:
                result = evaluator.calculate_all_admet(comp['mol'])
                result['CID'] = comp.get('CID', 0)
                result['Score'] = screening_results['scores'][i] if screening_results and i < len(screening_results['scores']) else 0
                result['SMILES'] = Chem.MolToSmiles(comp['mol'])
                admet_results.append(result)
            except Exception as e:
                print(f"  ADMET评估错误: {e}")

    print(f"\n完成 {len(admet_results)} 个化合物的ADMET评估")

    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)

    reporter = VirtualScreeningReporter(args.target, output_dir)

    admet_df = pd.DataFrame(admet_results) if admet_results else None

    report_path = reporter.generate_summary_report(
        pipeline_results={
            "target": args.target,
            "pdb_id": args.pdb_id,
            "training_samples": len(training_df),
            "compounds_screened": len(library.compounds),
            "top_compounds": len(screening_results['top_indices']) if screening_results else 0
        },
        admet_results=admet_df
    )

    results_file = output_dir / "screening_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "target": args.target,
            "pdb_id": args.pdb_id,
            "compounds_screened": len(library.compounds),
            "top_indices": [int(i) for i in screening_results['top_indices']] if screening_results else [],
            "top_scores": [float(s) for s in screening_results['scores']] if screening_results else [],
            "admet_results": admet_results
        }, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("虚拟筛选完成!")
    print("=" * 60)
    print(f"报告: {report_path}")
    print(f"结果: {results_file}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()