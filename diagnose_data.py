#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：检查训练数据和模型训练问题
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

# 读取训练数据
training_file = "E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data.csv"
df = pd.read_csv(training_file)

print("=" * 60)
print("训练数据诊断")
print("=" * 60)
print(f"总样本数: {len(df)}")
print(f"活性化合物 (Label=1): {sum(df['Label'])}")
print(f"非活性化合物 (Label=0): {len(df) - sum(df['Label'])}")
print(f"活性比例: {sum(df['Label'])/len(df):.2%}")

# 检查数据分布
print("\n数据分布:")
print(df['Label'].value_counts())

# 检查SMILES质量
from rdkit import Chem
valid_smiles = []
invalid_smiles = []

for i, row in df.iterrows():
    smiles = row['SMILES']
    label = row['Label']
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((i, smiles, label))
        else:
            valid_smiles.append((i, smiles, label))
    except Exception as e:
        invalid_smiles.append((i, smiles, label))

print(f"\n有效SMILES: {len(valid_smiles)}")
print(f"无效SMILES: {len(invalid_smiles)}")

# 检查有效数据中的类别分布
valid_labels = [v[2] for v in valid_smiles]
print(f"\n有效数据中活性化合物: {sum(valid_labels)}")
print(f"有效数据中非活性化合物: {len(valid_labels) - sum(valid_labels)}")

# 检查是否有重复的SMILES
smiles_counts = df['SMILES'].value_counts()
duplicate_smiles = smiles_counts[smiles_counts > 1]
print(f"\n重复SMILES数量: {len(duplicate_smiles)}")

# 如果有重复，检查它们的标签是否一致
if len(duplicate_smiles) > 0:
    print("\n重复SMILES的标签一致性检查:")
    for smiles, count in duplicate_smiles.head(5).items():
        labels = df[df['SMILES'] == smiles]['Label'].unique()
        if len(labels) > 1:
            print(f"  警告: {smiles} 有多个标签: {labels}")
        else:
            print(f"  OK: {smiles} 标签一致: {labels[0]}")

print("\n" + "=" * 60)
print("检查完成!")
print("=" * 60)
