#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型构建功能
直接调用模型构建核心代码，绕过 GUI 界面
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型构建相关模块
from src.modeling.model_training import train_classification_model
from src.feature_engineering.molecular_features import calculate_features


def test_model_building():
    """测试模型构建功能"""
    print("开始测试模型构建...")
    
    # 创建一个简单的测试数据集
    test_data = {
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # 阿司匹林
            'C1=CC=CC=C1',  # 苯
            'CCO',  # 乙醇
            'CC(=O)O',  # 乙酸
            'C1=CC=CC=C1C(=O)O',  # 苯甲酸
            'C1=CC=CC=C1OH',  # 苯酚
            'CC(=O)N',  # 乙酰胺
            'C1=CC=CC=C1NH2',  # 苯胺
            'C1=CC=CC=C1Cl',  # 氯苯
            'C1=CC=CC=C1Br'  # 溴苯
        ],
        'active': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0]
    }
    
    df = pd.DataFrame(test_data)
    
    # 过滤无效分子
    df = df[df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    
    if len(df) == 0:
        print("错误：没有有效的分子数据")
        return False
    
    print(f"有效分子数量: {len(df)}")
    
    # 提取特征
    print("正在提取特征...")
    try:
        # 重命名列名以匹配 calculate_features 函数的默认设置
        df_rename = df.rename(columns={'SMILES': 'canonical_smiles'})
        
        # 计算特征
        feature_df = calculate_features(
            df_rename,
            output_path=None,  # 不保存到文件
            smiles_column='canonical_smiles',
            features_to_calculate=['Morgan', 'MACCS'],  # 只计算 Morgan 和 MACCS 指纹
            feature_selection_method=None  # 不进行特征选择
        )
        
        if feature_df is None or len(feature_df) == 0:
            print("错误：特征提取失败")
            return False
        print(f"特征提取成功，特征维度: {feature_df.shape}")
    except Exception as e:
        print(f"特征提取出错: {str(e)}")
        return False
    
    # 测试不同模型
    models_to_test = [
        'RandomForest',
        'SVM',
        'XGBoost'
    ]
    
    print("\n开始训练和评估模型...")
    try:
        # 训练和评估模型
        trained_models, all_results, df_features = train_classification_model(
            input_df=feature_df,
            models_to_train=models_to_test,
            balance_data=True,
            cross_val=False,
            use_lazypredict=False
        )
        
        print("\n模型训练和评估完成！")
        print(f"训练的模型数量: {len(trained_models)}")
        
        # 显示结果
        for model_name, results in all_results.items():
            print(f"\n{model_name} 模型结果:")
            print(f"准确率: {results['test_metrics'].get('accuracy', 'N/A'):.4f}")
            print(f"精确率: {results['test_metrics'].get('precision', 'N/A'):.4f}")
            print(f"召回率: {results['test_metrics'].get('recall', 'N/A'):.4f}")
            print(f"F1分数: {results['test_metrics'].get('f1_score', 'N/A'):.4f}")
            print(f"ROC-AUC: {results['test_metrics'].get('roc_auc', 'N/A'):.4f}")
            
    except Exception as e:
        print(f"模型训练和评估出错: {str(e)}")
        return False
    
    # 测试深度学习模型（如果安装了相关依赖）
    print("\n测试深度学习模型...")
    try:
        import torch
        print("PyTorch 已安装，准备测试 Transformer 模型...")
        
        # 测试 Transformer 模型
        from src.modeling.deep_learning_models import SMILESTransformer, smiles_to_tensor
        
        # 准备 Transformer 数据
        smiles_list = df['SMILES'].tolist()
        labels = df['active'].tolist()
        
        # 转换为张量
        tensors = []
        for smiles in smiles_list:
            tensor = smiles_to_tensor(smiles)
            if tensor is not None:
                tensors.append(tensor)
        
        if tensors:
            print("Transformer 模型数据准备成功")
        else:
            print("Transformer 模型数据准备失败")
            
    except ImportError as e:
        print(f"深度学习依赖未安装: {str(e)}")
    except Exception as e:
        print(f"深度学习模型测试出错: {str(e)}")
    
    print("\n模型构建测试完成！")
    return True


if __name__ == "__main__":
    test_model_building()
