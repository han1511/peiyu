#!/usr/bin/env python3
"""测试中文文件名是否影响OpenBabel"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

from rdkit import Chem
from rdkit.Chem import AllChem

# 使用与test_docking_fix相同的路径
output_dir = 'test_docking_output'
os.makedirs(output_dir, exist_ok=True)

# 测试分子
smiles = 'CC(=O)O'
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

# 测试1: 中文文件名
sdf_file_cn = os.path.join(output_dir, '乙酸_ligand.sdf')
pdbqt_file_cn = os.path.join(output_dir, '乙酸_ligand.pdbqt')

writer = Chem.SDWriter(sdf_file_cn)
writer.write(mol)
writer.close()

print("=== 测试中文文件名 ===")
print(f"SDF文件: {sdf_file_cn}")
print(f"SDF存在: {os.path.exists(sdf_file_cn)}")

result = subprocess.run(
    ['obabel', sdf_file_cn, '-O', pdbqt_file_cn],
    capture_output=True, timeout=30
)
print(f"obabel返回码: {result.returncode}")
print(f"PDBQT存在: {os.path.exists(pdbqt_file_cn)}")
if os.path.exists(pdbqt_file_cn):
    print(f"PDBQT大小: {os.path.getsize(pdbqt_file_cn)} 字节")
    with open(pdbqt_file_cn, 'r', encoding='gbk', errors='replace') as f:
        content = f.read()
    print(f"ROOT: {'ROOT' in content}, ENDROOT: {'ENDROOT' in content}")

# 测试2: 英文文件名
sdf_file_en = os.path.join(output_dir, 'test_ligand.sdf')
pdbqt_file_en = os.path.join(output_dir, 'test_ligand.pdbqt')

writer2 = Chem.SDWriter(sdf_file_en)
writer2.write(mol)
writer2.close()

print("\n=== 测试英文文件名 ===")
result2 = subprocess.run(
    ['obabel', sdf_file_en, '-O', pdbqt_file_en],
    capture_output=True, timeout=30
)
print(f"obabel返回码: {result2.returncode}")
print(f"PDBQT存在: {os.path.exists(pdbqt_file_en)}")
if os.path.exists(pdbqt_file_en):
    print(f"PDBQT大小: {os.path.getsize(pdbqt_file_en)} 字节")

# 清理
for f in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, f))
os.rmdir(output_dir)
print("\n测试完成")