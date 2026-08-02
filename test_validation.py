#!/usr/bin/env python3
"""直接测试PDBQT验证"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

# 直接导入验证函数
from src.molecular_docking import AutoDockVina

output_dir = 'debug_pdbqt'

vina = AutoDockVina(
    vina_executable='E:/autodock/vina.exe',
    receptor_file='virtual_screening_pipeline/data/target_structures/4V0Q.pdbqt'
)

for fname in ['test_via_sdf.pdbqt', 'test_via_pdb.pdbqt']:
    fpath = os.path.join(output_dir, fname)
    if not os.path.exists(fpath):
        print(f"文件 {fname} 不存在")
        continue
    
    print(f"\n测试文件: {fname}")
    result = vina._validate_pdbqt(fpath)
    print(f"  验证结果: {result}")
    
    # 手动检查
    with open(fpath, 'r', encoding='gbk', errors='replace') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    atom_lines = [l for l in lines if l.startswith('ATOM')]
    
    print(f"  ROOT: {'ROOT' in content}")
    print(f"  ENDROOT: {'ENDROOT' in content}")
    print(f"  ATOM行数: {len(atom_lines)}")
    
    if atom_lines:
        for i, line in enumerate(atom_lines[:3]):
            print(f"  ATOM行{i}: len={len(line)}")
            print(f"    30:38 = '{line[30:38]}' -> float: {float(line[30:38])}")
            print(f"    38:46 = '{line[38:46]}' -> float: {float(line[38:46])}")
            print(f"    46:54 = '{line[46:54]}' -> float: {float(line[46:54])}")

print("\n验证测试完成")