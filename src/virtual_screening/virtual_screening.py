#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟筛选模块
用于筛选大型化合物库，预测潜在的抗登革病毒活性化合物
"""

import os
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm
# 为pandas添加进度条支持
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # 禁用SettingWithCopyWarning
from rdkit import Chem
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# 导入配置
from src.config import SCREENING_CONFIG, DATA_DIR, RESULTS_DIR
from src.feature_engineering.molecular_features import calculate_features

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_compound_library(library_path, format='smi', smiles_column=None):
    """
    加载化合物库
    
    参数:
        library_path: 化合物库文件路径
        format: 化合物库格式 ('smi' 或 'csv')
        smiles_column: SMILES列的名称（仅适用于CSV格式）
    
    返回:
        df: 包含SMILES的DataFrame
    """
    logger.info(f"加载化合物库: {library_path}")
    
    if format == 'smi':
        # 从SMILES文件加载
        df = pd.read_csv(library_path, header=None, names=['canonical_smiles'])
    elif format == 'csv':
        # 从CSV文件加载
        df = pd.read_csv(library_path)
        
        # 如果没有指定SMILES列名，尝试自动检测
        if smiles_column is None:
            # 常见的SMILES列名
            common_smiles_columns = ['canonical_smiles', 'smiles', 'SMILES', 'Canonical_SMILES', 'smi', 'SMI']
            found_columns = [col for col in df.columns if col.lower() in [c.lower() for c in common_smiles_columns]]
            
            if found_columns:
                smiles_column = found_columns[0]
                logger.info(f"自动检测到SMILES列: {smiles_column}")
            else:
                raise ValueError(f"CSV文件必须包含SMILES列。常见的SMILES列名包括: {', '.join(common_smiles_columns)}")
        
        # 确保指定的列存在
        if smiles_column not in df.columns:
            raise ValueError(f"CSV文件中不存在指定的SMILES列: {smiles_column}")
        
        # 将SMILES列重命名为'canonical_smiles'，以便后续处理
        if smiles_column != 'canonical_smiles':
            df = df.rename(columns={smiles_column: 'canonical_smiles'})
    else:
        raise ValueError(f"不支持的文件格式: {format}")
    
    logger.info(f"加载成功，共 {len(df)} 个化合物")
    
    # 如果没有化合物ID列，添加化合物ID
    if 'compound_id' not in df.columns:
        df['compound_id'] = [f'CMPD_{i+1}' for i in range(len(df))]
    
    return df

def batch_process_compounds(df, batch_size=100):
    """
    批量处理化合物，计算分子特征
    
    参数:
        df: 包含SMILES的DataFrame
        batch_size: 批处理大小
    
    返回:
        df_features: 包含特征的DataFrame
    """
    logger.info(f"批量处理化合物，批大小: {batch_size}")
    
    # 计算特征
    try:
        # 确保原始数据有compound_id列
        if 'compound_id' not in df.columns:
            df['compound_id'] = [f'CMPD_{i+1}' for i in range(len(df))]
        
        # 验证并清理SMILES列
        logger.info("验证并清理SMILES列...")
        
        # 检查SMILES列的数据类型
        logger.info(f"SMILES列数据类型: {df['canonical_smiles'].dtype}")
        
        # 将SMILES列转换为字符串类型
        df['canonical_smiles'] = df['canonical_smiles'].astype(str)
        
        # 移除空字符串或无效值
        initial_count = len(df)
        df = df[df['canonical_smiles'].str.strip() != '']
        df = df[df['canonical_smiles'] != 'nan']
        df = df[df['canonical_smiles'] != 'NaN']
        valid_count = len(df)
        
        logger.info(f"清理后有效SMILES数: {valid_count}/{initial_count}")
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 导入分子特征计算函数
        from src.feature_engineering.molecular_features import (
            calculate_morgan_fingerprint,
            calculate_maccs_fingerprint,
            calculate_rdkit_descriptors,
            preprocess_features
        )
        
        # 分批处理
        batch_results = []
        total_batches = (valid_count + batch_size - 1) // batch_size
        logger.info(f"共 {total_batches} 批")
        
        for i in tqdm(range(0, valid_count, batch_size), desc="批量处理"):
            end_idx = min(i + batch_size, valid_count)
            batch_df = df.iloc[i:end_idx].copy()
            
            logger.info(f"处理批次 {i//batch_size + 1}/{total_batches}, 化合物 {i+1}-{end_idx}")
            
            # 转换SMILES为分子对象
            mols = []
            for smiles in batch_df['canonical_smiles']:
                mols.append(Chem.MolFromSmiles(smiles))
            batch_df['mol'] = mols
            
            # 移除无效分子
            batch_df = batch_df.dropna(subset=['mol'])
            
            if len(batch_df) == 0:
                continue
            
            # 计算Morgan指纹
            morgan_fps = []
            for mol in batch_df['mol']:
                morgan_fps.append(calculate_morgan_fingerprint(mol, radius=2, n_bits=1024))
            batch_df['morgan_fp'] = morgan_fps
            
            # 计算MACCS指纹
            maccs_fps = []
            for mol in batch_df['mol']:
                maccs_fps.append(calculate_maccs_fingerprint(mol))
            batch_df['maccs_fp'] = maccs_fps
            
            # 计算RDKit描述符
            descriptors = []
            for mol in batch_df['mol']:
                descriptors.append(calculate_rdkit_descriptors(mol))
            batch_df['descriptors'] = descriptors
            
            # 展开Morgan指纹
            morgan_cols = [f'Morgan_{i}' for i in range(1024)]
            batch_df[morgan_cols] = pd.DataFrame(batch_df['morgan_fp'].tolist(), index=batch_df.index)
            
            # 展开MACCS指纹
            maccs_cols = [f'MACCS_{i}' for i in range(167)]
            batch_df[maccs_cols] = pd.DataFrame(batch_df['maccs_fp'].tolist(), index=batch_df.index)
            
            # 展开描述符
            df_descriptors = pd.DataFrame(batch_df['descriptors'].tolist(), index=batch_df.index)
            batch_df = pd.concat([batch_df, df_descriptors], axis=1)
            
            # 移除中间列
            batch_df = batch_df.drop(columns=['mol', 'morgan_fp', 'maccs_fp', 'descriptors'])
            
            # 先保留数值列和必要的标识符列
            keep_cols = ['compound_id', 'canonical_smiles']
            for col in batch_df.columns:
                try:
                    if batch_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        keep_cols.append(col)
                except AttributeError:
                    # 处理可能的DataFrame列（虽然不应该发生）
                    pass
            
            logger.info(f"保留的列数: {len(keep_cols)}")
            logger.info(f"数值特征列数: {len(keep_cols) - 2}")
            
            batch_df = batch_df[keep_cols]
            
            # 处理异常值
            logger.info("处理异常值...")
            # 替换无穷大和NaN值
            import numpy as np
            batch_df = batch_df.replace([np.inf, -np.inf], np.nan)
            # 用0填充NaN值
            batch_df = batch_df.fillna(0)
            
            # 限制数值范围，避免过大的值
            numeric_cols = [col for col in batch_df.columns if col not in ['compound_id', 'canonical_smiles']]
            for col in numeric_cols:
                # 限制最大值和最小值
                batch_df[col] = batch_df[col].clip(lower=-1e6, upper=1e6)
                # 确保所有值都是有限的
                batch_df[col] = batch_df[col].apply(lambda x: x if np.isfinite(x) else 0)
            
            logger.info("异常值处理完成")
            
            # 确保使用与训练时相同的特征
            logger.info("确保特征与训练时一致...")
            import joblib
            feature_list_path = os.path.join(DATA_DIR['processed'], 'feature_list.pkl')
            
            if os.path.exists(feature_list_path):
                try:
                    # 加载训练时使用的特征列表
                    train_features = joblib.load(feature_list_path)
                    logger.info(f"训练时使用的特征数: {len(train_features)}")
                    
                    # 确保所有训练特征都存在
                    for feature in train_features:
                        if feature not in batch_df.columns:
                            batch_df[feature] = 0
                    
                    # 只保留训练时使用的特征
                    batch_df = batch_df[['compound_id', 'canonical_smiles'] + train_features]
                    logger.info(f"调整后的特征数: {len(batch_df.columns) - 2}")
                except Exception as e:
                    logger.error(f"加载特征列表失败: {e}")
            else:
                logger.warning("未找到训练特征列表，使用默认特征")
            
            batch_results.append(batch_df)
        
        # 合并所有批次结果
        if batch_results:
            df_features = pd.concat(batch_results, ignore_index=True)
            logger.info(f"特征计算完成，共 {len(df_features)} 个化合物")
            return df_features
        else:
            logger.warning("没有有效化合物")
            return None
    except Exception as e:
        logger.error(f"特征计算失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def predict_activity(df_features, models, feature_columns=None):
    """
    预测化合物活性
    
    参数:
        df_features: 包含分子特征的DataFrame
        models: 训练好的模型字典，值可以是模型对象或(model, scaler)元组
        feature_columns: 用于预测的特征列
    
    返回:
        df_predictions: 包含预测结果的DataFrame
    """
    logger.info(f"使用 {len(models)} 个模型进行活性预测")
    
    # 尝试加载训练时使用的特征列表
    import joblib
    feature_list_path = os.path.join(DATA_DIR['processed'], 'feature_list.pkl')
    
    if os.path.exists(feature_list_path):
        try:
            feature_columns = joblib.load(feature_list_path)
            logger.info(f"加载训练时使用的特征列表，共 {len(feature_columns)} 个特征")
        except Exception as e:
            logger.error(f"加载特征列表失败: {e}")
            # 如果加载失败，使用默认方法
            if feature_columns is None:
                # 排除非特征列
                non_feature_columns = ['compound_id', 'molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'is_active']
                feature_columns = [col for col in df_features.columns if col not in non_feature_columns]
    else:
        # 如果没有特征列表，使用默认方法
        if feature_columns is None:
            # 排除非特征列
            non_feature_columns = ['compound_id', 'molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'is_active']
            feature_columns = [col for col in df_features.columns if col not in non_feature_columns]
    
    # 确保所有特征都存在
    missing_features = [col for col in feature_columns if col not in df_features.columns]
    if missing_features:
        logger.warning(f"缺少以下特征: {missing_features}")
        # 移除缺失的特征
        feature_columns = [col for col in feature_columns if col in df_features.columns]
        logger.info(f"使用 {len(feature_columns)} 个可用特征进行预测")
    
    logger.info(f"使用 {len(feature_columns)} 个特征进行预测")
    
    # 检查特征数量
    if len(feature_columns) == 0:
        logger.error("没有可用的特征进行预测！")
        # 返回包含基本信息的DataFrame
        df_predictions = df_features[['compound_id', 'canonical_smiles']].copy()
        df_predictions['prediction'] = 0
        df_predictions['average_probability'] = 0.0
        return df_predictions
    
    # 获取特征矩阵
    X = df_features[feature_columns].values
    
    # 为每个模型进行预测
    predictions = {}
    probabilities = {}
    
    for model_name, model_info in models.items():
        try:
            # 检查模型是否包含scaler
            if isinstance(model_info, tuple):
                model = model_info[0]
                scaler = model_info[1]
                # 标准化特征
                X_scaled = scaler.transform(X)
            else:
                model = model_info
                scaler = None
                X_scaled = X
            
            # 预测类别
            pred = model.predict(X_scaled)
            predictions[model_name] = pred
            
            # 预测概率
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)[:, 1]  # 正类概率
            else:
                # SVM等没有predict_proba方法的模型
                prob = model.decision_function(X_scaled)
                # 归一化到[0, 1]范围
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            
            probabilities[model_name] = prob
            
            logger.info(f"{model_name} 预测完成")
        except Exception as e:
            logger.error(f"{model_name} 预测失败: {e}")
    
    # 创建预测结果DataFrame
    df_predictions = df_features[['compound_id', 'canonical_smiles']].copy()
    
    # 添加预测结果
    for model_name, pred in predictions.items():
        df_predictions[f'{model_name}_prediction'] = pred
        df_predictions[f'{model_name}_probability'] = probabilities[model_name]
    
    # 计算平均预测结果
    if probabilities:
        prob_df = pd.DataFrame(probabilities)
        df_predictions['average_probability'] = prob_df.mean(axis=1)
        
        # 根据平均概率确定最终预测
        threshold = 0.4  # 降低预测阈值，增加活性化合物的检出率
        df_predictions['prediction'] = (df_predictions['average_probability'] >= threshold).astype(int)
        
        # 打印预测结果统计
        logger.info(f"使用阈值 {threshold} 进行预测")
        logger.info(f"预测活性化合物数: {df_predictions['prediction'].sum()}")
        logger.info(f"预测非活性化合物数: {len(df_predictions) - df_predictions['prediction'].sum()}")
    
    logger.info(f"预测完成，共预测 {len(df_predictions)} 个化合物")
    
    return df_predictions

def screen_compound_library(library_path, models, format='smi', batch_size=100, 
                           save_results=True, smiles_column=None):
    """
    虚拟筛选主函数
    
    参数:
        library_path: 化合物库文件路径
        models: 训练好的模型字典
        format: 化合物库格式 ('smi' 或 'csv')
        batch_size: 批处理大小
        save_results: 是否保存结果
        smiles_column: SMILES列的名称（仅适用于CSV格式）
    
    返回:
        screening_results: 筛选结果DataFrame
    """
    start_time = time.time()
    logger.info("开始虚拟筛选")
    
    # 1. 加载化合物库
    df = load_compound_library(library_path, format=format, smiles_column=smiles_column)
    
    # 2. 批量计算分子特征
    logger.info("批量计算分子特征...")
    df_features = batch_process_compounds(df, batch_size=batch_size)
    
    if df_features is None:
        logger.error("特征计算失败，无法进行虚拟筛选")
        return None
    
    # 3. 预测活性
    screening_results = predict_activity(df_features, models)
    
    # 4. 保存结果
    if save_results:
        # 创建虚拟筛选结果目录
        results_dir = os.path.join(RESULTS_DIR['models'], 'virtual_screening')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'screening_results_{timestamp}.csv')
        
        screening_results.to_csv(results_file, index=False)
        logger.info(f"筛选结果保存到: {results_file}")
    
    # 5. 统计结果
    total_count = len(screening_results)
    active_count = 0
    hit_rate = 0.0
    
    if 'prediction' in screening_results.columns:
        active_count = screening_results['prediction'].sum()
        hit_rate = active_count / total_count * 100 if total_count > 0 else 0.0
        logger.info(f"虚拟筛选完成，共筛选 {total_count} 个化合物，其中 {active_count} 个被预测为活性化合物，命中率: {hit_rate:.2f}%")
    else:
        logger.warning(f"虚拟筛选完成，但没有生成预测结果")
    
    logger.info(f"虚拟筛选完成")
    logger.info(f"总化合物数: {total_count}")
    logger.info(f"预测活性化合物数: {active_count}")
    logger.info(f"命中率: {hit_rate:.2f}%")
    
    end_time = time.time()
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    
    return screening_results

if __name__ == "__main__":
    # 示例用法
    import joblib
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='虚拟筛选模块，用于预测化合物的抗登革病毒活性')
    parser.add_argument('--library_path', '-l', type=str, 
                       default=os.path.join(DATA_DIR['raw'], 'example_library.smi'),
                       help='化合物库文件路径 (SMI或CSV格式)')
    parser.add_argument('--format', '-f', type=str, choices=['smi', 'csv'], default='smi',
                       help='化合物库文件格式 (smi 或 csv)')
    parser.add_argument('--batch_size', '-b', type=int, default=1000,
                       help='批处理大小 (默认: 1000)')
    parser.add_argument('--smiles_column', '-s', type=str, default=None,
                       help='SMILES列的名称 (仅适用于CSV格式，默认自动检测)')
    args = parser.parse_args()
    
    # 加载训练好的模型
    models_dir = RESULTS_DIR['models']
    logger.info(f"模型目录路径: {models_dir}")
    
    if os.path.exists(models_dir):
        # 列出目录中的所有文件
        all_files = os.listdir(models_dir)
        logger.info(f"目录中的文件: {all_files}")
        
        # 加载模型和对应的标准化器
        models = {}
        model_names = [f.split('_model.pkl')[0] for f in all_files if f.endswith('_model.pkl')]
        
        for model_name in model_names:
            model_file = f"{model_name}_model.pkl"
            scaler_file = f"{model_name}_scaler.pkl"
            
            model_path = os.path.join(models_dir, model_file)
            scaler_path = os.path.join(models_dir, scaler_file)
            
            try:
                # 加载模型
                model = joblib.load(model_path)
                
                # 加载标准化器（如果存在）
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    models[model_name] = (model, scaler)
                else:
                    models[model_name] = model
                
                logger.info(f"成功加载模型: {model_name}")
            except Exception as e:
                logger.error(f"加载模型 {model_name} 失败: {e}")
        
        logger.info(f"成功加载 {len(models)} 个模型: {list(models.keys())}")
        
        if models:
            # 进行虚拟筛选
            library_path = args.library_path
            logger.info(f"化合物库路径: {library_path}")
            
            if os.path.exists(library_path):
                screen_compound_library(
                    library_path=library_path, 
                    models=models, 
                    format=args.format,
                    batch_size=args.batch_size,
                    smiles_column=args.smiles_column
                )
            else:
                logger.error(f"化合物库文件不存在: {library_path}")
        else:
            logger.error(f"没有成功加载任何模型文件")
    else:
        logger.error(f"模型目录不存在: {models_dir}")
