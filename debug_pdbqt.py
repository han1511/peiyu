#!/usr/bin/env python3
"""检查PDBQT文件生成情况 - 使用命令行工具"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

from rdkit import Chem
from rdkit.Chem import AllChem

output_dir = 'debug_pdbqt'
os.makedirs(output_dir, exist_ok=True)

# 测试分子
smiles = 'CC(=O)O'
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
conf = mol.GetConformer()

# 方法1: SDF -> PDBQT via OpenBabel CLI
sdf_file = os.path.join(output_dir, 'test.sdf')
writer = Chem.SDWriter(sdf_file)
writer.write(mol)
writer.close()

pdbqt_file = os.path.join(output_dir, 'test_via_sdf.pdbqt')
result = subprocess.run(
    ['obabel', sdf_file, '-O', pdbqt_file],
    capture_output=True, timeout=30
)
print(f"方法1 (SDF->PDBQT): 返回码={result.returncode}")
if result.returncode == 0 and os.path.exists(pdbqt_file):
    print(f"  生成 {os.path.getsize(pdbqt_file)} 字节")
else:
    print(f"  输出: {result.stdout.decode('gbk', errors='replace') if result.stdout else 'N/A'}")
    print(f"  错误: {result.stderr.decode('gbk', errors='replace') if result.stderr else 'N/A'}")

# 方法2: PDB -> PDBQT via OpenBabel CLI
pdb_file = os.path.join(output_dir, 'test.pdb')
Chem.MolToPDBFile(mol, pdb_file)

pdbqt_file2 = os.path.join(output_dir, 'test_via_pdb.pdbqt')
result2 = subprocess.run(
    ['obabel', pdb_file, '-O', pdbqt_file2],
    capture_output=True, timeout=30
)
print(f"方法2 (PDB->PDBQT): 返回码={result2.returncode}")
if result2.returncode == 0:
    print(f"  生成 {os.path.getsize(pdbqt_file2)} 字节")

# 检查文件内容
for fname in ['test_via_sdf.pdbqt', 'test_via_pdb.pdbqt']:
    fpath = os.path.join(output_dir, fname)
    if not os.path.exists(fpath):
        print(f"\n文件 {fname} 不存在")
        continue
    
    print(f"\n{'='*60}")
    print(f"文件: {fname}")
    print(f"{'='*60}")
    with open(fpath, 'r', encoding='gbk', errors='replace') as f:
        content = f.read()
    lines = content.split('\n')
    print(f"总行数: {len(lines)}")
    
    # 显示前25行
    print(f"\n前25行:")
    for i, line in enumerate(lines[:25]):
        print(f"  {i:2d}: [{len(line):3d} chars] |{line}|")
    
    # 检查关键标记
    has_root = 'ROOT' in content
    has_endroot = 'ENDROOT' in content
    atom_lines = [l for l in lines if l.startswith('ATOM')]
    print(f"\nROOT: {has_root}, ENDROOT: {has_endroot}, ATOM行数: {len(atom_lines)}")
    
    # 检查ATOM行格式
    if atom_lines:
        print(f"\nATOM行格式检查:")
        for i, line in enumerate(atom_lines[:3]):
            print(f"  ATOM行{i}: len={len(line)}")
            print(f"    坐标字段(30:38): '{line[30:38]}'")
            print(f"    坐标字段(38:46): '{line[38:46]}'")
            print(f"    坐标字段(46:54): '{line[46:54]}'")

# 清理SDF和PDB中间文件
if os.path.exists(sdf_file):
    os.remove(sdf_file)
if os.path.exists(pdb_file):
    os.remove(pdb_file)
print("\n调试完成")