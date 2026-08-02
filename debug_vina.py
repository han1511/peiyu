#!/usr/bin/env python3
"""详细调试prepare_ligand流程"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess

output_dir = 'test_docking_output'
os.makedirs(output_dir, exist_ok=True)

# 测试分子
smiles = 'CC(=O)O'
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
conf = mol.GetConformer()

# 方法1: SDF -> PDBQT
sdf_file = os.path.join(output_dir, 'test_ligand.sdf')
writer = Chem.SDWriter(sdf_file)
writer.write(mol)
writer.close()

output_file = os.path.join(output_dir, 'test_ligand.pdbqt')

print("=== 方法1: SDF->PDBQT ===")
result = subprocess.run(
    ['obabel', sdf_file, '-O', output_file],
    capture_output=True, timeout=30
)
print(f"返回码: {result.returncode}")
print(f"文件存在: {os.path.exists(output_file)}")
if os.path.exists(output_file):
    print(f"文件大小: {os.path.getsize(output_file)} 字节")
    # 读取文件
    with open(output_file, 'r', encoding='gbk', errors='replace') as f:
        content = f.read()
    lines = content.strip().split('\n')
    print(f"行数: {len(lines)}")
    
    # 手动验证
    has_root = 'ROOT' in content
    has_endroot = 'ENDROOT' in content
    atom_lines = [l for l in lines if l.startswith('ATOM')]
    print(f"ROOT: {has_root}, ENDROOT: {has_endroot}, ATOM: {len(atom_lines)}")
    
    valid = True
    for line in atom_lines:
        if len(line) < 50:
            print(f"  行过短: '{line}'")
            valid = False
            continue
        try:
            float(line[30:38])
            float(line[38:46])
            float(line[46:54])
        except (ValueError, IndexError):
            try:
                float(line[31:39])
                float(line[39:47])
                float(line[47:55])
            except (ValueError, IndexError):
                print(f"  坐标解析失败: '{line}'")
                valid = False
    
    print(f"手动验证: {'通过' if valid else '失败'}")

# 现在用Vina尝试读取
print("\n=== Vina测试 ===")
vina_exe = 'E:/autodock/vina.exe'
receptor = 'virtual_screening_pipeline/data/target_structures/4V0Q.pdbqt'
log_file = os.path.join(output_dir, 'test_log.txt')
result_file = os.path.join(output_dir, 'test_result.pdbqt')

vina_cmd = [
    vina_exe,
    '--receptor', receptor,
    '--ligand', output_file,
    '--out', result_file,
    '--exhaustiveness', '8',
    '--num_modes', '20',
    '--energy_range', '3.0',
    '--center_x', '25.0',
    '--center_y', '162.0',
    '--center_z', '25.0',
    '--size_x', '25.0',
    '--size_y', '25.0',
    '--size_z', '25.0',
    '--log', log_file
]

print(f"执行命令: {' '.join(vina_cmd)}")
vina_result = subprocess.run(
    vina_cmd,
    capture_output=True,
    timeout=60
)

stdout = vina_result.stdout.decode('gbk', errors='replace') if vina_result.stdout else ""
stderr = vina_result.stderr.decode('gbk', errors='replace') if vina_result.stderr else ""

print(f"返回码: {vina_result.returncode}")
print(f"stdout: {stdout[:500]}")
if stderr:
    print(f"stderr: {stderr[:500]}")

# 清理
for f in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, f))
os.rmdir(output_dir)
print("\n调试完成")