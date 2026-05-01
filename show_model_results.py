#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看模型构建结果的详细信息
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

from src.ml_screening import VirtualScreening
from src.molecular_features import FeatureEngineering
from rdkit import Chem
import pandas as pd

def load_training_data():
    """加载训练数据"""
    training_file = "E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data_cleaned.csv"
    df = pd.read_csv(training_file)
    
    fe = FeatureEngineering()
    mols = []
    labels = []
    
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            labels.append(row['Label'])
    
    features, _, _ = fe.calculate_all_features(mols)
    X = features
    y = np.array(labels)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def show_model_results():
    """显示模型构建结果"""
    print("=" * 70)
    print("模型构建结果详细信息")
    print("=" * 70)
    
    # 加载数据
    X_train, X_val, y_train, y_val = load_training_data()
    print(f"\n训练数据: {len(X_train)} 样本, {X_train.shape[1]} 特征")
    print(f"验证数据: {len(X_val)} 样本")
    print(f"训练集活性比例: {sum(y_train)/len(y_train):.2%}")
    
    # 训练模型
    screening = VirtualScreening()
    results = screening.train_models(X_train, y_train, X_val, y_val)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("各模型性能:")
    print("=" * 70)
    
    for model_name, model_result in results['models'].items():
        if model_result.get('trained'):
            auc = model_result.get('validation_auc', 0)
            print(f"  OK {model_name}: AUC = {auc:.4f}")
        else:
            print(f"  FAIL {model_name}: {model_result.get('error', '训练失败')}")
    
    print("\n" + "=" * 70)
    print(f"最佳模型: {results['best_model']}")
    print(f"最佳AUC: {results['best_auc']:.4f}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    results = show_model_results()
    
    # 可以进一步使用训练好的模型
    # screening = VirtualScreening()
    # predictions = screening.predict(X_test)
