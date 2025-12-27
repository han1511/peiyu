#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子特征计算模块
使用RDKit计算分子描述符和指纹
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem import PandasTools
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置
from src.config import DATA_DIR, DATA_CONFIG, FEATURE_CONFIG, ACTIVITY_CONFIG


def load_molecules(data_path: str, smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
    """
    从CSV文件加载分子数据
    
    参数:
        data_path: CSV文件路径
        smiles_column: SMILES字符串所在的列名
    
    返回:
        pd.DataFrame: 包含分子对象的DataFrame
    """
    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 添加分子对象
    print("正在转换SMILES为分子对象...")
    tqdm.pandas(desc="转换SMILES")
    df['mol'] = df[smiles_column].progress_apply(Chem.MolFromSmiles)
    
    # 移除无效分子
    initial_count = len(df)
    df = df.dropna(subset=['mol'])
    valid_count = len(df)
    
    print(f"有效分子数: {valid_count}/{initial_count}")
    print(f"无效分子数: {initial_count - valid_count}")
    
    return df


def calculate_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    计算Morgan指纹
    
    参数:
        mol: RDKit分子对象
        radius: 指纹半径
        n_bits: 指纹位数
    
    返回:
        np.ndarray: Morgan指纹向量
    """
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))


def calculate_maccs_fingerprint(mol) -> np.ndarray:
    """
    计算MACCS指纹
    
    参数:
        mol: RDKit分子对象
    
    返回:
        np.ndarray: MACCS指纹向量
    """
    if mol is None:
        return None
    return np.array(MACCSkeys.GenMACCSKeys(mol))


def calculate_rdkit_descriptors(mol) -> dict:
    """
    计算RDKit描述符
    
    参数:
        mol: RDKit分子对象
    
    返回:
        dict: 描述符名称到值的映射
    """
    if mol is None:
        return None
    
    descriptors = {}
    for desc_name, desc_func in Descriptors._descList:
        try:
            value = desc_func(mol)
            descriptors[desc_name] = value
        except:
            descriptors[desc_name] = np.nan
    
    return descriptors


def calculate_fingerprints(df, fingerprint_types: list = None) -> pd.DataFrame:
    """
    计算分子指纹
    
    参数:
        df: 包含分子对象的DataFrame
        fingerprint_types: 要计算的指纹类型列表
    
    返回:
        pd.DataFrame: 包含指纹特征的DataFrame
    """
    if fingerprint_types is None:
        fingerprint_types = FEATURE_CONFIG['fingerprint_types']
    
    result_df = df.copy()
    
    for fp_type in fingerprint_types:
        print(f"正在计算{fp_type}指纹...")
        
        if fp_type == 'Morgan':
            radius = FEATURE_CONFIG['morgan_radius']
            n_bits = FEATURE_CONFIG['morgan_bits']
            
            tqdm.pandas(desc=f"计算{fp_type}指纹")
            fingerprints = result_df['mol'].progress_apply(
                lambda x: calculate_morgan_fingerprint(x, radius=radius, n_bits=n_bits)
            )
            
            # 将指纹向量展开为列
            fp_df = pd.DataFrame(fingerprints.tolist(), columns=[f"Morgan_{i}" for i in range(n_bits)])
            
        elif fp_type == 'MACCS':
            tqdm.pandas(desc=f"计算{fp_type}指纹")
            fingerprints = result_df['mol'].progress_apply(calculate_maccs_fingerprint)
            
            # 将指纹向量展开为列
            fp_df = pd.DataFrame(fingerprints.tolist(), columns=[f"MACCS_{i}" for i in range(167)])
            
        else:
            print(f"警告: 不支持的指纹类型: {fp_type}")
            continue
            
        # 合并到结果DataFrame
        result_df = pd.concat([result_df, fp_df], axis=1)
    
    return result_df


def calculate_descriptors(df, desc_types: list = None) -> pd.DataFrame:
    """
    计算分子描述符
    
    参数:
        df: 包含分子对象的DataFrame
        desc_types: 要计算的描述符类型列表
    
    返回:
        pd.DataFrame: 包含描述符特征的DataFrame
    """
    if desc_types is None:
        desc_types = FEATURE_CONFIG['desc_types']
    
    result_df = df.copy()
    
    for desc_type in desc_types:
        print(f"正在计算{desc_type}描述符...")
        
        if desc_type == 'rdkit_desc':
            tqdm.pandas(desc=f"计算{desc_type}")
            descriptors = result_df['mol'].progress_apply(calculate_rdkit_descriptors)
            
            # 将描述符字典展开为列
            desc_df = pd.DataFrame(descriptors.tolist())
            
        else:
            print(f"警告: 不支持的描述符类型: {desc_type}")
            continue
            
        # 合并到结果DataFrame
        result_df = pd.concat([result_df, desc_df], axis=1)
    
    return result_df


def preprocess_features(df, drop_constant: bool = True, drop_correlated: bool = True, corr_threshold: float = 0.95) -> pd.DataFrame:
    """
    预处理特征数据
    
    参数:
        df: 包含特征的DataFrame
        drop_constant: 是否移除常数特征
        drop_correlated: 是否移除高相关特征
        corr_threshold: 相关系数阈值
    
    返回:
        pd.DataFrame: 预处理后的DataFrame
    """
    result_df = df.copy()
    
    # 获取所有数值特征列，但排除活性标签列
    exclude_cols = ['active', 'standard_value', 'standard_value_nM']
    feature_cols = [col for col in result_df.select_dtypes(include=[np.number]).columns.tolist() if col not in exclude_cols]
    
    # 移除常数特征
    if drop_constant:
        print("正在移除常数特征...")
        constant_cols = [col for col in feature_cols if result_df[col].nunique() == 1]
        print(f"移除的常数特征数: {len(constant_cols)}")
        result_df = result_df.drop(columns=constant_cols)
        feature_cols = [col for col in feature_cols if col not in constant_cols]
    
    # 移除高相关特征
    if drop_correlated:
        print("正在移除高相关特征...")
        corr_matrix = result_df[feature_cols].corr().abs()
        
        # 选择上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找到相关系数大于阈值的特征对
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        
        print(f"移除的高相关特征数: {len(to_drop)}")
        result_df = result_df.drop(columns=to_drop)
    
    return result_df


def calculate_features(input_data, output_path: str = None, 
                     smiles_column: str = 'canonical_smiles',
                     features_to_calculate: list = None) -> pd.DataFrame:
    """
    生成分子特征的完整流程
    
    参数:
        input_data: 输入CSV文件路径或DataFrame对象
        output_path: 输出CSV文件路径
        smiles_column: SMILES字符串所在的列名
        features_to_calculate: 要计算的特征类型列表，默认为None（使用配置中的默认值）
    
    返回:
        pd.DataFrame: 包含分子特征的DataFrame
    """
    # 使用配置中的默认值
    if features_to_calculate is None:
        features_to_calculate = FEATURE_CONFIG['fingerprint_types'] + FEATURE_CONFIG['desc_types']
    
    if output_path is None:
        output_path = os.path.join(DATA_DIR['processed'], DATA_CONFIG['processed_data_file'])
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. 加载分子
    if isinstance(input_data, str):
        # 从文件路径加载
        df = load_molecules(input_data, smiles_column=smiles_column)
    else:
        # 直接使用DataFrame
        df = input_data.copy()
        # 添加分子对象
        print("正在转换SMILES为分子对象...")
        tqdm.pandas(desc="转换SMILES")
        df['mol'] = df[smiles_column].progress_apply(Chem.MolFromSmiles)
        
        # 移除无效分子
        initial_count = len(df)
        df = df.dropna(subset=['mol'])
        valid_count = len(df)
        
        print(f"有效分子数: {valid_count}/{initial_count}")
        print(f"无效分子数: {initial_count - valid_count}")
    
    # 2. 计算指纹
    fingerprint_types = [ft for ft in features_to_calculate if ft in ['Morgan', 'MACCS', 'morgan', 'maccs']]
    if fingerprint_types:
        df = calculate_fingerprints(df, fingerprint_types=[ft.capitalize() for ft in fingerprint_types])
    
    # 3. 计算描述符
    desc_types = [dt for dt in features_to_calculate if dt in ['rdkit_desc']]
    if desc_types:
        df = calculate_descriptors(df, desc_types=desc_types)
    
    # 4. 添加活性标签 (如果有standard_value列)
    if 'standard_value' in df.columns and 'standard_units' in df.columns:
        print("正在添加活性标签...")
        # 检查单位并转换为nM（如果需要）
        df['standard_value_nM'] = df['standard_value'].copy()
        for idx, row in df.iterrows():
            if row['standard_units'] == 'uM':
                df.loc[idx, 'standard_value_nM'] = row['standard_value'] * 1000  # 转换为nM
            elif row['standard_units'] == 'mM':
                df.loc[idx, 'standard_value_nM'] = row['standard_value'] * 1000000  # 转换为nM
        
        # 根据IC50阈值生成活性标签 (IC50 <= 1000 nM 视为活性)
        df['active'] = (df['standard_value_nM'] <= ACTIVITY_CONFIG['ic50_threshold']).astype(int)
        print(f"活性化合物数: {df['active'].sum()}")
        print(f"非活性化合物数: {len(df) - df['active'].sum()}")
    
    # 4. 预处理特征
    df = preprocess_features(df)
    
    # 5. 移除分子对象列（无法保存到CSV）
    if 'mol' in df.columns:
        df = df.drop(columns=['mol'])
    
    # 6. 保存处理后的数据
    print(f"正在保存处理后的数据: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"特征生成完成！")
    print(f"特征总数: {len(df.columns) - 1} (不包括SMILES和活性标签)")
    
    return df


if __name__ == "__main__":
    # 生成特征
    input_path = os.path.join(DATA_DIR['raw'], DATA_CONFIG['chembl_data_file'])
    df = calculate_features(input_path)
    
    # 显示数据信息
    print("\n数据信息:")
    print(df.info())
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df.head())
    
    # 显示特征统计信息
    print("\n特征统计信息:")
    print(df.describe())
