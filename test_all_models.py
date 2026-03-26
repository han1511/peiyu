#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有模型是否正常运行
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 测试数据
test_data = [
    {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O", "active": 1},  # 阿司匹林
    {"SMILES": "C1=CC=CC=C1", "active": 0},  # 苯
    {"SMILES": "CC(=O)Nc1ccc(O)cc1", "active": 1},  # 对乙酰氨基酚
    {"SMILES": "CCO", "active": 0},  # 乙醇
    {"SMILES": "CC(=O)Oc1ccccc1C(=O)O", "active": 1},  # 邻苯二甲酸二乙酯
    {"SMILES": "C1CCCCC1", "active": 0},  # 环己烷
    {"SMILES": "C1=CN=CC=C1", "active": 1},  # 吡啶
    {"SMILES": "CCOC(=O)C1=CC=CC=C1", "active": 0},  # 苯甲酸乙酯
    {"SMILES": "C1=CC=C(C=C1)O", "active": 1},  # 苯酚
    {"SMILES": "C1=CC=C(C=C1)Cl", "active": 0}   # 氯苯
]

def test_traditional_models():
    """测试传统模型"""
    print("=== 测试传统模型 ===")
    
    # 导入传统模型
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        # 准备数据
        df = pd.DataFrame(test_data)
        
        # 计算特征
        def calculate_features(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                
                # Morgan 指纹
                morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256))
                
                # MACCS 指纹
                maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))
                
                # 合并特征
                return np.concatenate([morgan_fp, maccs_fp])
            except:
                return None
        
        features = []
        labels = []
        for _, row in df.iterrows():
            feat = calculate_features(row['SMILES'])
            if feat is not None:
                features.append(feat)
                labels.append(row['active'])
        
        X = np.array(features)
        y = np.array(labels)
        
        # 测试 RandomForest
        print("测试 RandomForest...")
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
        rf_model.fit(X, y)
        rf_score = rf_model.score(X, y)
        print(f"RandomForest 得分: {rf_score:.4f}")
        
        # 测试 SVM
        print("测试 SVM...")
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X, y)
        svm_score = svm_model.score(X, y)
        print(f"SVM 得分: {svm_score:.4f}")
        
        # 测试 LogisticRegression
        print("测试 LogisticRegression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X, y)
        lr_score = lr_model.score(X, y)
        print(f"LogisticRegression 得分: {lr_score:.4f}")
        
        # 测试 XGBoost
        try:
            from xgboost import XGBClassifier
            print("测试 XGBoost...")
            xgb_model = XGBClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
            xgb_model.fit(X, y)
            xgb_score = xgb_model.score(X, y)
            print(f"XGBoost 得分: {xgb_score:.4f}")
        except ImportError:
            print("XGBoost 未安装，跳过测试")
        
        print("传统模型测试完成！\n")
        
    except Exception as e:
        print(f"传统模型测试失败: {str(e)}")

def test_deep_learning_models():
    """测试深度学习模型"""
    print("=== 测试深度学习模型 ===")
    
    try:
        import torch
        
        # 测试 GNN
        try:
            from torch_geometric.data import Data, DataLoader
            from src.modeling.deep_learning_models import GNNModel, mol_to_graph
            
            print("测试 GNN...")
            
            # 准备数据
            data_list = []
            for item in test_data:
                mol = Chem.MolFromSmiles(item['SMILES'])
                if mol:
                    data = mol_to_graph(mol)
                    if data:
                        data.y = torch.tensor([item['active']], dtype=torch.long)
                        data_list.append(data)
            
            if data_list:
                # 创建模型
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = GNNModel().to(device)
                
                # 测试前向传播
                loader = DataLoader(data_list, batch_size=2)
                for batch in loader:
                    batch = batch.to(device)
                    output = model(batch.x, batch.edge_index, batch.batch)
                    print(f"GNN 输出形状: {output.shape}")
                print("GNN 测试成功！")
            else:
                print("GNN 数据准备失败")
        except ImportError as e:
            print(f"GNN 测试失败: {str(e)}")
        
        # 测试 Transformer
        try:
            from src.modeling.deep_learning_models import SMILESTransformer, smiles_to_tensor
            
            print("测试 Transformer...")
            
            # 准备数据
            tensors = []
            for item in test_data:
                tensor = smiles_to_tensor(item['SMILES'])
                tensors.append(tensor)
            
            if tensors:
                X = torch.stack(tensors)
                
                # 创建模型
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = SMILESTransformer().to(device)
                
                # 测试前向传播
                output = model(X.to(device))
                print(f"Transformer 输出形状: {output.shape}")
                print("Transformer 测试成功！")
            else:
                print("Transformer 数据准备失败")
        except ImportError as e:
            print(f"Transformer 测试失败: {str(e)}")
        
        # 测试 DeepChem
        try:
            import deepchem as dc
            
            print("测试 DeepChem...")
            
            # 准备数据
            df = pd.DataFrame(test_data)
            featurizer = dc.feat.CircularFingerprint(size=256)
            X = featurizer.featurize(df['SMILES'].tolist())
            y = df['active'].values.reshape(-1, 1)
            
            # 创建数据集
            dataset = dc.data.NumpyDataset(X=X, y=y)
            
            # 创建模型
            model = dc.models.MultitaskClassifier(
                n_tasks=1,
                n_features=256,
                layer_sizes=[100, 100]
            )
            
            # 测试训练
            model.fit(dataset, nb_epoch=1)
            print("DeepChem 测试成功！")
        except ImportError as e:
            print(f"DeepChem 测试失败: {str(e)}")
        
        print("深度学习模型测试完成！\n")
        
    except Exception as e:
        print(f"深度学习模型测试失败: {str(e)}")

def test_gui_import():
    """测试 GUI 导入"""
    print("=== 测试 GUI 导入 ===")
    
    try:
        from screening_gui import ScreeningGUI
        print("GUI 导入成功！")
    except Exception as e:
        print(f"GUI 导入失败: {str(e)}")

def main():
    """主测试函数"""
    print("开始测试所有模型...\n")
    
    test_traditional_models()
    test_deep_learning_models()
    test_gui_import()
    
    print("所有测试完成！")

if __name__ == "__main__":
    main()
