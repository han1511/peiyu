#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复训练数据：清理重复SMILES并平衡数据集
"""

import os
import numpy as np
import pandas as pd

# 读取原始数据
training_file = "E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data.csv"
df = pd.read_csv(training_file)

print("=" * 60)
print("修复训练数据")
print("=" * 60)
print(f"原始数据: {len(df)} 条")
print(f"活性化合物: {sum(df['Label'])}")
print(f"非活性化合物: {len(df) - sum(df['Label'])}")

# 步骤1: 处理重复SMILES
print("\n步骤1: 处理重复SMILES")
smiles_counts = df['SMILES'].value_counts()
duplicate_smiles = smiles_counts[smiles_counts > 1]
print(f"重复SMILES数量: {len(duplicate_smiles)}")

# 找出有矛盾标签的SMILES
conflict_smiles = []
for smiles, count in duplicate_smiles.items():
    labels = df[df['SMILES'] == smiles]['Label'].unique()
    if len(labels) > 1:
        conflict_smiles.append(smiles)
        print(f"  删除矛盾SMILES: {smiles} (标签: {labels})")

# 删除有矛盾的SMILES
df_clean = df[~df['SMILES'].isin(conflict_smiles)]

# 移除重复（保留第一个出现的）
df_clean = df_clean.drop_duplicates(subset='SMILES', keep='first')

print(f"\n处理后数据: {len(df_clean)} 条")
print(f"活性化合物: {sum(df_clean['Label'])}")
print(f"非活性化合物: {len(df_clean) - sum(df_clean['Label'])}")

# 步骤2: 平衡数据集
print("\n步骤2: 平衡数据集")
active_df = df_clean[df_clean['Label'] == 1]
inactive_df = df_clean[df_clean['Label'] == 0]

print(f"活性化合物: {len(active_df)}")
print(f"非活性化合物: {len(inactive_df)}")

# 取较小的数量作为采样数量
min_count = min(len(active_df), len(inactive_df))

# 随机采样
active_sample = active_df.sample(n=min_count, random_state=42)
inactive_sample = inactive_df.sample(n=min_count, random_state=42)

# 合并平衡后的数据集
balanced_df = pd.concat([active_sample, inactive_sample])

# 打乱顺序
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n平衡后数据: {len(balanced_df)} 条")
print(f"活性化合物: {sum(balanced_df['Label'])}")
print(f"非活性化合物: {len(balanced_df) - sum(balanced_df['Label'])}")
print(f"活性比例: {sum(balanced_df['Label'])/len(balanced_df):.2%}")

# 保存修复后的数据
output_file = "E:/Python/dengue_drug_discovery/src/modeling/DENV_NS5_training_data_cleaned.csv"
balanced_df.to_csv(output_file, index=False)

print(f"\n修复后的数据已保存到: {output_file}")

print("\n" + "=" * 60)
print("数据修复完成!")
print("=" * 60)
