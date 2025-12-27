#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从PubChem数据库下载化合物的测试脚本

这个脚本演示了如何使用fetch_pubchem_data模块从PubChem数据库下载化合物数据，
支持多种下载方式，包括指定CID范围、按物质名称搜索、从本地文件读取SMILES列表等。
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_acquisition.fetch_pubchem_data import fetch_pubchem_compounds, fetch_pubchem_substance_list
from src.config import DATA_DIR


def main():
    print("=" * 80)
    print("PubChem化合物下载工具")
    print("=" * 80)
    
    # 选项菜单
    print("请选择下载方式：")
    print("1. 下载特定CID范围的化合物")
    print("2. 按物质名称搜索并下载化合物")
    print("3. 从本地文件读取SMILES列表并下载对应化合物")
    print("4. 下载PubChem前1000个化合物（默认示例）")
    
    choice = input("请输入选项 (1-4): ").strip()
    
    if choice == '1':
        # 下载特定CID范围的化合物
        try:
            start_cid = int(input("请输入起始CID: ").strip())
            end_cid = int(input("请输入结束CID: ").strip())
            batch_size = int(input("请输入批次大小 (建议100-500): ").strip())
            
            cid_list = list(range(start_cid, end_cid + 1))
            print(f"\n准备下载CID {start_cid} 到 {end_cid} 的化合物...")
            
            df = fetch_pubchem_compounds(
                compound_list=cid_list,
                identifier_type='cid',
                batch_size=batch_size,
                output_file=f'pubchem_cids_{start_cid}_{end_cid}.csv'
            )
            
        except ValueError as e:
            print(f"输入错误: {e}")
            return
    
    elif choice == '2':
        # 按物质名称搜索并下载化合物
        substance_name = input("请输入物质名称 (如 'Dengue inhibitor'): ").strip()
        max_compounds = int(input("请输入最大下载数量 (建议不超过1000): ").strip())
        batch_size = int(input("请输入批次大小 (建议100-500): ").strip())
        
        print(f"\n正在搜索并下载与 '{substance_name}' 相关的化合物...")
        
        df = fetch_pubchem_substance_list(
            substance_name=substance_name,
            max_compounds=max_compounds,
            output_file=f'pubchem_{substance_name.replace(" ", "_")}_compounds.csv'
        )
    
    elif choice == '3':
        # 从本地文件读取SMILES列表并下载对应化合物
        file_path = input("请输入包含SMILES的文件路径: ").strip()
        batch_size = int(input("请输入批次大小 (建议100-500): ").strip())
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        try:
            # 读取SMILES列表
            df_smiles = pd.read_csv(file_path)
            
            if 'SMILES' not in df_smiles.columns:
                print("文件中没有 'SMILES' 列，请检查文件格式")
                return
            
            smiles_list = df_smiles['SMILES'].tolist()
            print(f"\n从文件中读取到 {len(smiles_list)} 个SMILES")
            print(f"准备下载对应化合物数据...")
            
            df = fetch_pubchem_compounds(
                compound_list=smiles_list,
                identifier_type='smiles',
                batch_size=batch_size,
                output_file='pubchem_smiles_list_compounds.csv'
            )
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
            return
    
    elif choice == '4':
        # 下载PubChem前1000个化合物
        batch_size = int(input("请输入批次大小 (建议100-500): ").strip())
        
        print("\n正在下载PubChem前1000个化合物...")
        
        df = fetch_pubchem_compounds(
            batch_size=batch_size,
            output_file='pubchem_first_1000_compounds.csv'
        )
    
    else:
        print("无效选项，请重新运行脚本并选择1-4之间的选项")
        return
    
    # 显示下载结果
    if not df.empty:
        print("\n" + "=" * 80)
        print("下载完成！")
        print("=" * 80)
        print(f"共成功获取 {len(df)} 个化合物")
        print(f"数据已保存到: {os.path.join(DATA_DIR['raw'], df.attrs.get('output_file', 'pubchem_compounds.csv'))}")
        print("\n数据基本信息:")
        print(df.info())
        print("\n前5行数据:")
        print(df.head())
    else:
        print("\n下载失败，未获取到任何化合物数据")
    
    print("\n" + "=" * 80)
    print("脚本执行完毕")
    print("=" * 80)


if __name__ == "__main__":
    main()
