#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试PubChem化合物下载功能
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_acquisition.fetch_pubchem_data import fetch_pubchem_compounds

def main():
    print("=" * 80)
    print("测试PubChem化合物下载功能")
    print("=" * 80)
    
    # 下载PubChem前100个化合物
    print("正在下载PubChem前100个化合物...")
    
    try:
        df = fetch_pubchem_compounds(
            compound_list=list(range(1, 101)),  # 下载CID 1-100的化合物
            identifier_type='cid',
            batch_size=50,  # 每批次50个化合物
            delay=0.5,  # 增加延迟以避免API限制
            output_file='test_pubchem_compounds.csv'
        )
        
        if not df.empty:
            print("\n" + "=" * 80)
            print("下载成功！")
            print("=" * 80)
            print(f"共成功获取 {len(df)} 个化合物")
            print("\n数据基本信息:")
            print(df.info())
            print("\n前5行数据:")
            print(df.head())
            print("\n数据已保存到: data/raw/test_pubchem_compounds.csv")
        else:
            print("\n下载失败，未获取到任何化合物数据")
            
    except Exception as e:
        print(f"\n下载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
