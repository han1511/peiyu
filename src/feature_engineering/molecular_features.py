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


def calculate_padel_descriptors(df, smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
    """
    使用PADEL计算分子描述符
    
    参数:
        df: 包含SMILES的DataFrame
        smiles_column: SMILES列名
    
    返回:
        pd.DataFrame: 包含PADEL描述符的DataFrame
    """
    try:
        from padelpy import padeldescriptor
        import tempfile
        import os
        
        print("正在计算PADEL描述符...")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            for i, smiles in enumerate(df[smiles_column]):
                f.write(f"{i} {smiles}\n")
            temp_smi = f.name
        
        # 定义输出文件
        temp_csv = temp_smi.replace('.smi', '.csv')
        
        # 计算描述符
        padeldescriptor(
            mol_dir=temp_smi,
            d_file=temp_csv,
            descriptortypes='./descriptors.xml',
            retainorder=True,
            fingerprints=True,
            d_2d=True,
            d_3d=False
        )
        
        # 读取结果
        padel_df = pd.read_csv(temp_csv)
        
        # 清理临时文件
        os.unlink(temp_smi)
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)
        
        # 移除第一列（索引列）
        if 'Name' in padel_df.columns:
            padel_df = padel_df.drop('Name', axis=1)
        
        # 合并到原始DataFrame
        result_df = pd.concat([df, padel_df], axis=1)
        print(f"PADEL描述符计算完成，添加了{len(padel_df.columns)}个特征")
        
        return result_df
    except ImportError:
        print("警告: padelpy库未安装，跳过PADEL描述符计算")
        return df
    except Exception as e:
        print(f"计算PADEL描述符时出错: {e}")
        return df


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
            
        elif desc_type == 'padel_desc':
            result_df = calculate_padel_descriptors(result_df)
            continue
            
        else:
            print(f"警告: 不支持的描述符类型: {desc_type}")
            continue
            
        # 合并到结果DataFrame
        result_df = pd.concat([result_df, desc_df], axis=1)
    
    return result_df


def analyze_features(df, target_column: str = 'active') -> dict:
    """
    分析特征数据
    
    参数:
        df: 包含特征的DataFrame
        target_column: 目标列名
    
    返回:
        dict: 特征分析结果
    """
    print("正在分析特征数据...")
    
    # 获取所有数值特征列
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in feature_cols:
        feature_cols.remove(target_column)
    
    analysis_results = {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'feature_stats': {}
    }
    
    # 计算每个特征的基本统计信息
    for col in feature_cols[:50]:  # 限制分析前50个特征
        analysis_results['feature_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'nan_count': int(df[col].isna().sum())
        }
    
    # 特征相关性分析
    if len(feature_cols) > 1:
        print("计算特征相关性...")
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [(i, j, round(upper.loc[i, j], 2)) 
                           for i in upper.columns 
                           for j in upper.index 
                           if upper.loc[i, j] > 0.95]
        analysis_results['highly_correlated'] = highly_correlated
        print(f"找到{len(highly_correlated)}对高度相关的特征")
    
    # 特征重要性分析（基于随机森林）
    print("分析特征重要性...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    if target_column in df.columns:
        X = df[feature_cols]
        y = df[target_column]
        
        # 处理NaN值
        X = X.fillna(0)
        
        # 划分训练集
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:20]
        analysis_results['top_features'] = feature_importance
        
        print("Top 10 重要特征:")
        for feat, imp in feature_importance[:10]:
            print(f"{feat}: {imp:.4f}")
    
    return analysis_results


def feature_selection(df, target_column: str = 'active', method: str = 'all', n_features: int = 100) -> pd.DataFrame:
    """
    特征选择
    
    参数:
        df: 包含特征和目标变量的DataFrame
        target_column: 目标列名
        method: 特征选择方法 ('variance', 'correlation', 'importance', 'rfe', 'all')
        n_features: 选择的特征数量
    
    返回:
        pd.DataFrame: 包含选择后特征的DataFrame
    """
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
    from sklearn.ensemble import RandomForestClassifier
    
    print(f"正在使用{method}方法进行特征选择...")
    
    # 获取所有数值特征列，但排除活性标签列
    exclude_cols = ['active', 'standard_value', 'standard_value_nM', 'pIC50', 'activity_class']
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_column]
    
    # 处理NaN值
    X = X.fillna(0)
    
    selected_features = feature_cols.copy()
    
    # 方差阈值
    if method in ['variance', 'all']:
        print("使用方差阈值选择特征...")
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        var_features = [feature_cols[i] for i, mask in enumerate(selector.get_support()) if mask]
        print(f"方差阈值选择后特征数: {len(var_features)}")
        selected_features = var_features
        X = X[selected_features]
    
    # 相关性分析
    if method in ['correlation', 'all'] and len(selected_features) > 1:
        print("使用相关性分析选择特征...")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        corr_features = [col for col in selected_features if col not in to_drop]
        print(f"相关性分析后特征数: {len(corr_features)}")
        selected_features = corr_features
        X = X[selected_features]
    
    # 特征重要性
    if method in ['importance', 'all'] and len(selected_features) > n_features:
        print("使用特征重要性选择特征...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        feature_importance = sorted(zip(selected_features, importances), key=lambda x: x[1], reverse=True)
        imp_features = [f[0] for f in feature_importance[:n_features]]
        print(f"特征重要性选择后特征数: {len(imp_features)}")
        selected_features = imp_features
    
    # 递归特征消除
    if method in ['rfe', 'all'] and len(selected_features) > n_features:
        print("使用递归特征消除选择特征...")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=10)
        rfe.fit(X, y)
        rfe_features = [selected_features[i] for i, mask in enumerate(rfe.get_support()) if mask]
        print(f"递归特征消除后特征数: {len(rfe_features)}")
        selected_features = rfe_features
    
    # 保留原始列
    result_df = df.copy()
    # 只保留选择的特征和必要的列
    keep_cols = ['canonical_smiles', 'molecule_chembl_id', target_column] + selected_features
    # 确保所有必要的列都存在
    for col in keep_cols:
        if col not in result_df.columns:
            keep_cols.remove(col)
    result_df = result_df[keep_cols]
    
    print(f"最终选择的特征数: {len(selected_features)}")
    return result_df


def preprocess_features(df, drop_constant: bool = True, drop_correlated: bool = True, corr_threshold: float = 0.95, feature_selection_method: str = None, n_features: int = 100) -> pd.DataFrame:
    """
    预处理特征数据
    
    参数:
        df: 包含特征的DataFrame
        drop_constant: 是否移除常数特征
        drop_correlated: 是否移除高相关特征
        corr_threshold: 相关系数阈值
        feature_selection_method: 特征选择方法，None表示不进行特征选择
        n_features: 选择的特征数量
    
    返回:
        pd.DataFrame: 预处理后的DataFrame
    """
    result_df = df.copy()
    
    # 获取所有数值特征列，但排除活性标签列
    exclude_cols = ['active', 'standard_value', 'standard_value_nM', 'pIC50', 'activity_class']
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
    
    # 特征选择
    if feature_selection_method is not None and 'active' in result_df.columns:
        result_df = feature_selection(result_df, method=feature_selection_method, n_features=n_features)
    
    return result_df


def calculate_features(input_data, output_path: str = None, 
                     smiles_column: str = 'canonical_smiles',
                     features_to_calculate: list = None,
                     feature_selection_method: str = None,
                     n_features: int = 100) -> pd.DataFrame:
    """
    生成分子特征的完整流程
    
    参数:
        input_data: 输入CSV文件路径或DataFrame对象
        output_path: 输出CSV文件路径
        smiles_column: SMILES字符串所在的列名
        features_to_calculate: 要计算的特征类型列表，默认为None（使用配置中的默认值）
        feature_selection_method: 特征选择方法，默认为None（不进行特征选择）
        n_features: 选择的特征数量，默认为100
    
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
    desc_types = [dt for dt in features_to_calculate if dt in ['rdkit_desc', 'padel_desc']]
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
    df = preprocess_features(df, feature_selection_method=feature_selection_method, n_features=n_features)
    
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
