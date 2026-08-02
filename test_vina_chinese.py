#!/usr/bin/env python3
"""测试Vina对中文路径的支持"""
import sys
import os
import subprocess

output_dir = 'test_docking_output'
os.makedirs(output_dir, exist_ok=True)

# 使用已经验证可以工作的PDBQT文件
# 先用英文文件名生成一个
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = 'CC(=O)O'
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

# 生成英文PDBQT作为测试
sdf_file = os.path.join(output_dir, 'ligand_test.sdf')
mol_block = Chem.MolToMolBlock(mol)
with open(sdf_file, 'w', encoding='utf-8') as f:
    f.write(mol_block)

pdbqt_en = os.path.join(output_dir, 'ligand_en.pdbqt')
result = subprocess.run(['obabel', sdf_file, '-O', pdbqt_en], capture_output=True, timeout=30)
print(f"英文PDBQT: {os.path.exists(pdbqt_en)}, 大小: {os.path.getsize(pdbqt_en)}")

# 复制为中文文件名
pdbqt_cn = os.path.join(output_dir, '乙酸_ligand.pdbqt')
import shutil
shutil.copy(pdbqt_en, pdbqt_cn)
print(f"中文PDBQT: {os.path.exists(pdbqt_cn)}, 大小: {os.path.getsize(pdbqt_cn)}")

# 测试Vina with English path
vina_exe = 'E:/autodock/vina.exe'
receptor = 'virtual_screening_pipeline/data/target_structures/4V0Q.pdbqt'

print("\n=== 测试Vina + 英文路径 ===")
cmd_en = [
    vina_exe,
    '--receptor', receptor,
    '--ligand', pdbqt_en,
    '--out', os.path.join(output_dir, 'result_en.pdbqt'),
    '--exhaustiveness', '8',
    '--num_modes', '20',
    '--center_x', '25.0',
    '--center_y', '162.0',
    '--center_z', '25.0',
    '--size_x', '25.0',
    '--size_y', '25.0',
    '--size_z', '25.0',
    '--log', os.path.join(output_dir, 'log_en.txt')
]
result_en = subprocess.run(cmd_en, capture_output=True, timeout=60)
stdout_en = result_en.stdout.decode('gbk', errors='replace') if result_en.stdout else ""
print(f"返回码: {result_en.returncode}")
print(f"stdout (前300字符): {stdout_en[:300]}")

print("\n=== 测试Vina + 中文路径 ===")
cmd_cn = [
    vina_exe,
    '--receptor', receptor,
    '--ligand', pdbqt_cn,
    '--out', os.path.join(output_dir, 'result_cn.pdbqt'),
    '--exhaustiveness', '8',
    '--num_modes', '20',
    '--center_x', '25.0',
    '--center_y', '162.0',
    '--center_z', '25.0',
    '--size_x', '25.0',
    '--size_y', '25.0',
    '--size_z', '25.0',
    '--log', os.path.join(output_dir, 'log_cn.txt')
]
result_cn = subprocess.run(cmd_cn, capture_output=True, timeout=60)
stdout_cn = result_cn.stdout.decode('gbk', errors='replace') if result_cn.stdout else ""
print(f"返回码: {result_cn.returncode}")
print(f"stdout (前300字符): {stdout_cn[:300]}")

# 清理
shutil.rmtree(output_dir, ignore_errors=True)
print("\n测试完成")