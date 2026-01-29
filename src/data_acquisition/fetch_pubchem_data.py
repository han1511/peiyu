#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从PubChem数据库获取化合物数据
"""

import os
import sys
import time
import pandas as pd
import requests
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置
from src.config import DATA_DIR, DATA_CONFIG, PUBCHEM_CONFIG


def fetch_pubchem_compounds(compound_list=None, identifier_type='cid', output_dir=None, output_file=None, 
                           batch_size=None, delay=None, max_retries=None, timeout=None, 
                           large_scale=False, resume=False, checkpoint_interval=1000):
    """
    从PubChem数据库获取化合物数据
    
    参数:
        compound_list: 化合物标识符列表 (默认下载前1000个化合物)
        identifier_type: 标识符类型 ('cid', 'name', 'smiles', 'inchi', 'inchikey', 'sdf', 'cas')
        output_dir: 输出目录路径 (默认使用配置中的路径)
        output_file: 输出文件名 (默认使用配置中的文件名)
        batch_size: 批量处理大小 (PubChem API限制)
        delay: 请求之间的延迟 (秒)
        max_retries: 最大重试次数
        timeout: 请求超时时间 (秒)
        large_scale: 是否进行大规模下载 (启用断点续传和内存优化)
        resume: 是否从上次中断的位置继续下载
        checkpoint_interval: 保存检查点的间隔 (每下载多少个化合物保存一次)
    
    返回:
        pd.DataFrame: 包含化合物数据的DataFrame
    """
    # 使用配置中的默认值
    if output_dir is None:
        output_dir = DATA_DIR['raw']
    if output_file is None:
        output_file = 'pubchem_compounds.csv'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用配置中的默认值
    if batch_size is None:
        batch_size = PUBCHEM_CONFIG['default_batch_size']
    if delay is None:
        delay = PUBCHEM_CONFIG['default_delay']
    if max_retries is None:
        max_retries = PUBCHEM_CONFIG['max_retries']
    if timeout is None:
        timeout = PUBCHEM_CONFIG['timeout']
    
    # 如果没有提供化合物列表，下载前1000个化合物
    if compound_list is None:
        print("未提供化合物列表，将下载PubChem前1000个化合物...")
        compound_list = list(range(1, 1001))
        identifier_type = 'cid'
    
    total_compounds = len(compound_list)
    print(f"准备获取 {total_compounds} 个化合物的数据...")
    print(f"标识符类型: {identifier_type}")
    
    # 检查点文件路径
    checkpoint_file = None
    if large_scale:
        checkpoint_file = os.path.join(output_dir or DATA_DIR['raw'], f"{output_file or 'pubchem_compounds'}_checkpoint.txt")
    
    # 断点续传处理
    start_index = 0
    if resume and checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                start_index = int(f.read().strip())
            if start_index >= total_compounds:
                print("所有化合物已下载完成")
                return pd.read_csv(os.path.join(output_dir or DATA_DIR['raw'], output_file or 'pubchem_compounds.csv'))
            print(f"从第 {start_index} 个化合物开始继续下载")
        except Exception as e:
            print(f"读取检查点文件失败: {e}")
            start_index = 0
    
    # 使用剩余的化合物列表
    compound_list = compound_list[start_index:]
    
    # 拆分化合物列表为多个批次
    compound_batches = [compound_list[i:i+batch_size] for i in range(0, len(compound_list), batch_size)]
    print(f"共分为 {len(compound_batches)} 个批次，每批次 {batch_size} 个化合物")
    
    # PubChem API端点
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # 结果列表
    all_compounds = []
    
    # 已处理的化合物数量
    processed_count = start_index
    
    # 处理每个批次
    for batch_idx, batch in enumerate(tqdm(compound_batches, desc="处理批次")):
        batch_success = False
        retry_count = 0
        
        while not batch_success and retry_count < max_retries:
            try:
                # 构建请求URL
                ids = ','.join(map(str, batch))
                url = f"{base_url}/compound/{identifier_type}/{ids}/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,XLogP,TPSA,HeavyAtomCount,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/CSV"
                
                # 发送请求
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()  # 检查请求是否成功
                
                # 解析CSV响应
                from io import StringIO
                df_batch = pd.read_csv(StringIO(response.text))
                
                # 添加到结果列表
                all_compounds.append(df_batch)
                
                # 更新已处理的化合物数量
                processed_count += len(df_batch)
                
                batch_success = True
                
                # 大规模下载时，定期保存检查点
                if large_scale and processed_count % checkpoint_interval == 0:
                    if checkpoint_file:
                        with open(checkpoint_file, 'w') as f:
                            f.write(str(processed_count))
                        print(f"\n已保存检查点: {processed_count} 个化合物")
                
                # 请求之间添加延迟，避免超过PubChem API限制
                time.sleep(delay)
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                print(f"批次 {batch_idx+1} 请求失败 (重试 {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    wait_time = delay * 2 * retry_count
                    print(f"{wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"批次 {batch_idx+1} 达到最大重试次数，跳过该批次")
                    break
    
    # 合并所有批次的数据
    if all_compounds:
        print(f"\n正在合并数据...")
        
        # 大规模下载时，分块合并以减少内存使用
        if large_scale:
            # 先保存每个批次的数据，然后合并
            temp_files = []
            for i, df_batch in enumerate(all_compounds):
                temp_file = os.path.join(output_dir or DATA_DIR['raw'], f"temp_batch_{i}.csv")
                df_batch.to_csv(temp_file, index=False, encoding='utf-8')
                temp_files.append(temp_file)
            
            # 合并所有临时文件
            print(f"正在合并 {len(temp_files)} 个临时文件...")
            df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
            
            # 删除临时文件
            for f in temp_files:
                os.remove(f)
        else:
            # 常规合并
            df = pd.concat(all_compounds, ignore_index=True)
        
        print(f"共成功获取 {len(df)} 个化合物的数据")
        
        # 数据预处理
        print("正在预处理数据...")
        
        # 移除重复的化合物
        df = df.drop_duplicates(subset=['CID'], keep='first')
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 保存数据
        output_path = os.path.join(output_dir or DATA_DIR['raw'], output_file or 'pubchem_compounds.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"数据已保存到: {output_path}")
        
        # 删除检查点文件（如果存在）
        if large_scale and checkpoint_file and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        return df
    else:
        print("未获取到任何化合物数据")
        return pd.DataFrame()


def fetch_pubchem_substance_list(substance_name, max_compounds=None, output_dir=None, output_file=None):
    """
    根据物质名称从PubChem搜索化合物并下载数据
    
    参数:
        substance_name: 要搜索的物质名称 (如 "Dengue inhibitor")
        max_compounds: 最大下载化合物数量
        output_dir: 输出目录路径 (默认使用配置中的路径)
        output_file: 输出文件名 (默认使用配置中的文件名)
    
    返回:
        pd.DataFrame: 包含搜索到的化合物数据的DataFrame
    """
    # 使用配置中的默认值
    if output_dir is None:
        output_dir = DATA_DIR['raw']
    if output_file is None:
        output_file = f'pubchem_{substance_name.replace(" ", "_")}_compounds.csv'
    if max_compounds is None:
        max_compounds = PUBCHEM_CONFIG['max_compounds_per_search']
    
    print(f"正在搜索 PubChem 中与 '{substance_name}' 相关的化合物...")
    
    # PubChem API端点
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # 搜索化合物
    search_url = f"{base_url}/compound/name/{substance_name}/cids/JSON"
    
    try:
        response = requests.get(search_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
            cids = data['IdentifierList']['CID']
            
            # 限制化合物数量
            if len(cids) > max_compounds:
                cids = cids[:max_compounds]
                print(f"找到 {len(cids)} 个相关化合物，仅下载前 {max_compounds} 个")
            else:
                print(f"找到 {len(cids)} 个相关化合物")
            
            # 下载化合物数据
            return fetch_pubchem_compounds(cids, identifier_type='cid', output_dir=output_dir, output_file=output_file)
        else:
            print("未找到相关化合物")
            return pd.DataFrame()
    
    except requests.exceptions.RequestException as e:
        print(f"搜索失败: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # 示例1: 下载特定CID范围的化合物
    # df1 = fetch_pubchem_compounds(list(range(1, 101)), identifier_type='cid')
    
    # 示例2: 搜索并下载与登革热抑制剂相关的化合物
    df2 = fetch_pubchem_substance_list("Dengue inhibitor", max_compounds=500)
    
    # 显示数据信息
    if not df2.empty:
        print("\n数据基本信息:")
        print(df2.info())
        print("\n前5行数据:")
        print(df2.head())
    
    # 示例3: 从本地文件读取SMILES列表并下载对应化合物数据
    # smiles_list = pd.read_csv('data/raw/smiles_list.csv')['SMILES'].tolist()
    # df3 = fetch_pubchem_compounds(smiles_list, identifier_type='smiles')
