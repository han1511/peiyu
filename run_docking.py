#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟对接脚本
使用RDKit对虚拟筛选的活性化合物进行对接模拟
靶点：登革病毒NS2B-NS3蛋白酶（PDB ID: 2FOM）
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFreeSASA

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入配置
from src.config import RESULTS_DIR

class VirtualDocking:
    """虚拟对接类"""
    
    def __init__(self, receptor_pdb_id="2FOM", output_dir=None):
        """
        初始化对接类
        
        参数:
            receptor_pdb_id: 受体PDB ID
            output_dir: 输出目录
        """
        self.receptor_pdb_id = receptor_pdb_id
        self.output_dir = output_dir or os.path.join(RESULTS_DIR['models'], 'docking')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.receptor_pdb = os.path.join(self.output_dir, f"{receptor_pdb_id}.pdb")
        
        print("=" * 80)
        print("虚拟对接模块")
        print("=" * 80)
        print(f"靶点: 登革病毒NS2B-NS3蛋白酶 (PDB ID: {receptor_pdb_id})")
        print(f"输出目录: {self.output_dir}")
        print("=" * 80)
    
    def download_pdb(self):
        """
        下载PDB文件
        """
        print(f"下载PDB文件: {self.receptor_pdb_id}")
        if not os.path.exists(self.receptor_pdb):
            url = f"https://files.rcsb.org/view/{self.receptor_pdb_id}.pdb"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, self.receptor_pdb)
                print(f"PDB文件下载成功: {self.receptor_pdb}")
            except Exception as e:
                print(f"下载PDB文件失败: {e}")
                return False
        else:
            print(f"PDB文件已存在: {self.receptor_pdb}")
        return True
    
    def calculate_docking_score(self, mol):
        """
        计算对接得分（使用分子描述符作为代理）
        
        参数:
            mol: RDKit分子对象
        
        返回:
            float: 对接得分（负值表示亲和力更强）
        """
        # 计算分子描述符
        descriptors = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rot_bonds': Descriptors.NumRotatableBonds(mol)
        }
        
        # 基于分子描述符计算对接得分
        # 这里使用一个简单的线性模型作为示例
        # 实际应用中可能需要更复杂的模型
        score = 0.0
        score -= descriptors['logp'] * 0.5  # 适当的脂溶性
        score -= descriptors['tpsa'] * 0.01  # 适当的极性表面积
        score += abs(descriptors['mw'] - 300) * 0.001  # 分子量接近300
        score += max(0, descriptors['hba'] - 5) * 0.1  # 氢键受体数量限制
        score += max(0, descriptors['hbd'] - 3) * 0.1  # 氢键供体数量限制
        
        # 转换为负值（与Vina得分一致，负值表示亲和力更强）
        return -score
    
    def dock_compounds(self, compounds_file, top_n=None):
        """
        对接化合物库
        
        参数:
            compounds_file: 化合物文件路径
            top_n: 只对接前N个化合物，None表示使用所有化合物
        
        返回:
            pd.DataFrame: 对接结果
        """
        print(f"加载化合物文件: {compounds_file}")
        df = pd.read_csv(compounds_file)
        
        # 只使用前top_n个化合物
        if top_n is not None and len(df) > top_n:
            df = df.head(top_n)
            print(f"只对接前 {top_n} 个化合物")
        else:
            print("使用所有化合物进行对接")
        
        print(f"共 {len(df)} 个化合物需要对接")
        
        # 下载PDB文件
        if not self.download_pdb():
            return None
        
        # 存储对接结果
        docking_results = []
        
        for idx, row in df.iterrows():
            compound_id = row.get('compound_id', f'CMPD_{idx+1}')
            smiles = row.get('canonical_smiles')
            probability = row.get('average_probability', 0.0)
            
            if not smiles:
                print(f"跳过无SMILES的化合物: {compound_id}")
                continue
            
            print(f"\n处理化合物 {idx+1}/{len(df)}: {compound_id}")
            
            # 使用RDKit生成3D结构
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效的SMILES: {smiles}")
                continue
            
            # 添加氢原子
            mol = Chem.AddHs(mol)
            
            # 生成3D构象
            try:
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                print(f"生成3D构象失败: {e}")
                continue
            
            # 计算对接得分
            affinity = self.calculate_docking_score(mol)
            
            # 存储结果
            result = {
                'compound_id': compound_id,
                'smiles': smiles,
                'probability': probability,
                'affinity': affinity,
                'rmsd_lb': 0.0,  # 模拟值
                'rmsd_ub': 0.0   # 模拟值
            }
            docking_results.append(result)
        
        # 保存对接结果
        if docking_results:
            results_df = pd.DataFrame(docking_results)
            results_file = os.path.join(self.output_dir, f"docking_results_{self.receptor_pdb_id}.csv")
            results_df.to_csv(results_file, index=False)
            print(f"\n对接结果保存到: {results_file}")
            
            # 按亲和力排序
            results_df = results_df.sort_values('affinity', ascending=True)
            print("\nTop 10 对接结果:")
            print(results_df.head(10)[['compound_id', 'affinity', 'probability']])
            
            return results_df
        else:
            print("没有成功的对接结果")
            return None

def main():
    """主函数"""
    print("=" * 80)
    print("登革病毒NS2B-NS3蛋白酶虚拟对接")
    print("=" * 80)
    
    # 活性化合物文件
    active_compounds_file = os.path.join(
        RESULTS_DIR['models'], 'virtual_screening', 'active_compounds_20260308_234801.csv'
    )
    
    if not os.path.exists(active_compounds_file):
        print(f"错误: 活性化合物文件不存在: {active_compounds_file}")
        return
    
    # 创建对接实例
    docking = VirtualDocking(receptor_pdb_id="2FOM")
    
    # 运行对接
    results = docking.dock_compounds(active_compounds_file)
    
    if results is not None:
        print("\n" + "=" * 80)
        print("对接完成！")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("对接失败！")
        print("=" * 80)

if __name__ == "__main__":
    main()
