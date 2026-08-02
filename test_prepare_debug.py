#!/usr/bin/env python3
"""检查prepare_ligand生成的实际文件"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'virtual_screening_pipeline'))

from src.molecular_docking import AutoDockVina

output_dir = 'test_docking_output'
os.makedirs(output_dir, exist_ok=True)

vina = AutoDockVina(
    vina_executable='E:/autodock/vina.exe',
    receptor_file='virtual_screening_pipeline/data/target_structures/4V0Q.pdbqt'
)

# 测试用中文文件名
smiles = 'CC(=O)O'
cn_output = os.path.join(output_dir, '乙酸_ligand.pdbqt')
print(f"尝试制备: {cn_output}")
print(f"目录: {os.listdir(output_dir)}")

success = vina.prepare_ligand(smiles, cn_output)
print(f"制备结果: {success}")

print(f"\n目录内容:")
for f in os.listdir(output_dir):
    full = os.path.join(output_dir, f)
    size = os.path.getsize(full)
    print(f"  {f}: {size} 字节")

# 检查是否有文件名编码问题
import glob
print(f"\nglob搜索: {glob.glob(os.path.join(output_dir, '*'))}")

# 如果文件存在，检查内容
if os.path.exists(cn_output):
    print(f"\n文件存在!")
    with open(cn_output, 'r', encoding='gbk', errors='replace') as f:
        content = f.read()
    print(f"内容长度: {len(content)}")
    print(f"前500字符:")
    print(content[:500])
else:
    print(f"\n文件不存在: {cn_output}")
    # 尝试找到实际生成的文件
    for f in os.listdir(output_dir):
        full = os.path.join(output_dir, f)
        if f.endswith('.pdbqt'):
            print(f"找到PDBQT文件: {f}")
            with open(full, 'r', encoding='gbk', errors='replace') as fh:
                content = fh.read()
            print(f"内容长度: {len(content)}")
            print(content[:500])

# 清理
import shutil
shutil.rmtree(output_dir, ignore_errors=True)
print("\n调试完成")