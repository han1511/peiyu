#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json

# 添加项目根目录到sys.path
sys.path.append('E:\\Python\\dengue_drug_discovery\\virtual_screening_pipeline')

from src.molecular_docking import AutoDockVina, DockingConfig

def find_latest_results():
    """查找最新的筛选结果目录"""
    results_dir = "E:\\Python\\dengue_drug_discovery\\results"
    ns5_dirs = []
    
    for item in os.listdir(results_dir):
        if item.startswith("NS5_") and os.path.isdir(os.path.join(results_dir, item)):
            ns5_dirs.append(os.path.join(results_dir, item))
    
    if not ns5_dirs:
        print("未找到筛选结果目录")
        return None
    
    # 按修改时间排序，返回最新的
    ns5_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return ns5_dirs[0]

# 查找最新结果
latest_dir = find_latest_results()
if not latest_dir:
    sys.exit(1)

results_file = os.path.join(latest_dir, "screening_results.json")
print(f"使用筛选结果: {results_file}")

# 初始化Vina
vina = AutoDockVina(
    vina_executable="E:\\autodock\\vina.exe"
)

# 加载筛选结果
with open(results_file, 'r') as f:
    screening_results = json.load(f)

print("Screening results:")
print(f"Top compounds: {len(screening_results.get('top_indices', []))}")
print(f"Top scores: {len(screening_results.get('top_scores', []))}")

# 显示Top 5化合物分数
print("\nTop 5 compound scores:")
for i, score in enumerate(screening_results.get('top_scores', [])[:5]):
    print(f"Compound {i+1}: Score = {score:.4f}")

# 准备一个简单的配体（布洛芬）
print("\nTesting molecular docking with ibuprofen...")
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # 布洛芬

# 准备配体
ligand_file = "ibuprofen.pdbqt"
success = vina.prepare_ligand(smiles, ligand_file)
print(f"Ligand preparation: {'Success' if success else 'Failed'}")

# 检查4V0Q.pdb文件
pdb_file = "E:\\Python\\dengue_drug_discovery\\virtual_screening_pipeline\\data\\target_structures\\4V0Q.pdb"
if os.path.exists(pdb_file):
    print(f"PDB file exists: {pdb_file}")
    # 检查文件大小
    size = os.path.getsize(pdb_file)
    print(f"PDB file size: {size} bytes")
else:
    print("PDB file not found!")
