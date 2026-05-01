#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征生成功能

使用小样本化合物测试特征生成是否正常工作
"""

import sys
import os
import numpy as np

# 检查RDKit
print("Python version:", sys.version)

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("RDKit version:", rdkit.__version__)
    print("RDKit loaded successfully")
except ImportError as e:
    print("Error loading RDKit:", e)
    sys.exit(1)

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

# 测试分子指纹计算
def test_fingerprints():
    print("\n=== Testing Fingerprints ===")
    
    # 测试分子
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    mol = Chem.MolFromSmiles(test_smiles)
    
    if mol is None:
        print("Failed to create molecule")
        return False
    
    print("Created molecule:", test_smiles)
    
    # 计算Morgan指纹
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp)
        print("Morgan fingerprint shape:", fp_array.shape)
        print("Morgan fingerprint sum:", np.sum(fp_array))
    except Exception as e:
        print("Error calculating Morgan fingerprint:", e)
        return False
    
    return True

# 测试描述符计算
def test_descriptors():
    print("\n=== Testing Descriptors ===")
    
    # 测试分子
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    mol = Chem.MolFromSmiles(test_smiles)
    
    if mol is None:
        print("Failed to create molecule")
        return False
    
    from rdkit.Chem import Descriptors
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        print(f"Molecular Weight: {mw}")
        print(f"LogP: {logp}")
        print(f"TPSA: {tpsa}")
        print(f"H-Bond Donors: {hbd}")
        print(f"H-Bond Acceptors: {hba}")
    except Exception as e:
        print("Error calculating descriptors:", e)
        return False
    
    return True

# 测试FeatureEngineering类
def test_feature_engineering():
    print("\n=== Testing Feature Engineering ===")
    
    try:
        from src.molecular_features import FeatureEngineering
        
        # 测试分子
        test_smiles_list = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
            "C1=CC=CC=C1",  # 苯
            "CCO"
        ]  # 乙醇
        
        mols = []
        for smiles in test_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
        
        print(f"Created {len(mols)} molecules")
        
        if not mols:
            print("No valid molecules")
            return False
        
        # 测试特征工程
        fe = FeatureEngineering(fingerprint_types=["Morgan", "MACCS"])
        features, valid_indices, feature_names = fe.calculate_all_features(mols)
        
        print(f"Features shape: {features.shape}")
        print(f"Valid indices: {valid_indices}")
        print(f"Feature names count: {len(feature_names)}")
        
        if features.size > 0:
            print("Feature generation successful!")
            return True
        else:
            print("Feature generation failed - empty features")
            return False
            
    except Exception as e:
        print("Error in feature engineering:", e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing feature generation functionality...")
    
    test1 = test_fingerprints()
    test2 = test_descriptors()
    test3 = test_feature_engineering()
    
    print("\n=== Test Results ===")
    print(f"Fingerprints: {'PASS' if test1 else 'FAIL'}")
    print(f"Descriptors: {'PASS' if test2 else 'FAIL'}")
    print(f"Feature Engineering: {'PASS' if test3 else 'FAIL'}")
    
    if all([test1, test2, test3]):
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
