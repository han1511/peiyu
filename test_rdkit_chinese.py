#!/usr/bin/env python3
"""测试RDKit对中文文件名的支持"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

from rdkit import Chem
from rdkit.Chem import AllChem

output_dir = 'test_docking_output'
os.makedirs(output_dir, exist_ok=True)

smiles = 'CC(=O)O'
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

# 测试1: 中文文件名 - SDWriter
cn_path = os.path.join(output_dir, '乙酸_ligand.sdf')
print(f"尝试写入: {cn_path}")
print(f"目录内容(前): {os.listdir(output_dir)}")

writer = Chem.SDWriter(cn_path)
writer.write(mol)
writer.close()

print(f"目录内容(后): {os.listdir(output_dir)}")
print(f"文件存在: {os.path.exists(cn_path)}")

# 列出所有文件
for f in os.listdir(output_dir):
    full = os.path.join(output_dir, f)
    print(f"  文件: {f}, 大小: {os.path.getsize(full)}")

# 测试2: 使用英文文件名
en_path = os.path.join(output_dir, 'ligand_en.sdf')
writer2 = Chem.SDWriter(en_path)
writer2.write(mol)
writer2.close()
print(f"\n英文文件存在: {os.path.exists(en_path)}, 大小: {os.path.getsize(en_path)}")

# 测试3: 直接用open()写中文路径
test_path = os.path.join(output_dir, '测试文件.txt')
try:
    with open(test_path, 'w') as f:
        f.write("test")
    print(f"\nopen()中文文件: 存在={os.path.exists(test_path)}, 大小={os.path.getsize(test_path)}")
except Exception as e:
    print(f"\nopen()中文文件失败: {e}")

# 清理
import shutil
shutil.rmtree(output_dir)
print("\n测试完成")