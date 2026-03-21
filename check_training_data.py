#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练数据中的活性化合物数量
"""

import pandas as pd

# 读取处理后的数据
df = pd.read_csv('data/processed/processed_dengue_data.csv')

print("=== 训练数据统计 ===")
print(f"总化合物数: {len(df)}")
print(f"活性化合物数: {df['active'].sum()}")
print(f"非活性化合物数: {len(df) - df['active'].sum()}")
print(f"活性比例: {df['active'].sum() / len(df) * 100:.2f}%")

# 检查原始数据
print("\n=== 原始数据统计 ===")
raw_df = pd.read_csv('data/raw/dengue_antiviral_data.csv')
print(f"总化合物数: {len(raw_df)}")
if 'active' in raw_df.columns:
    print(f"活性化合物数: {raw_df['active'].sum()}")
    print(f"非活性化合物数: {len(raw_df) - raw_df['active'].sum()}")
    print(f"活性比例: {raw_df['active'].sum() / len(raw_df) * 100:.2f}%")
else:
    print("原始数据中没有active列")
