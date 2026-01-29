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
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# 导入配置
from src.config import SCREENING_CONFIG, DATA_DIR, RESULTS_DIR
from src.feature_engineering.molecular_features import calculate_features

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_compound_library(library_path, format='smi'):
    """
    加载化合物库
    
    参数:
        library_path: 化合物库文件路径
        format: 化合物库格式 ('smi' 或 'csv')
    
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
        if 'canonical_smiles' not in df.columns:
            raise ValueError("CSV文件必须包含'canonical_smiles'列")
    else:
        raise ValueError(f"不支持的文件格式: {format}")
    
    logger.info(f"加载成功，共 {len(df)} 个化合物")
    
    # 添加化合物ID
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
        
        # 将DataFrame保存为临时文件，包含compound_id
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=True)
        
        # 调用calculate_features函数
        df_features = calculate_features(temp_path, smiles_column='canonical_smiles')
        
        # 确保df_features包含compound_id
        if 'compound_id' not in df_features.columns:
            # 如果特征数据没有compound_id，使用索引重新添加
            df_features['compound_id'] = df['compound_id'].values
            df_features['canonical_smiles'] = df['canonical_smiles'].values
        
        logger.info(f"特征计算完成，共 {len(df_features)} 个化合物")
        return df_features
    except Exception as e:
        logger.error(f"特征计算失败: {e}")
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
    
    # 如果没有指定特征列，则自动识别
    if feature_columns is None:
        # 排除非特征列
        non_feature_columns = ['compound_id', 'molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'is_active']
        feature_columns = [col for col in df_features.columns if col not in non_feature_columns]
    
    logger.info(f"使用 {len(feature_columns)} 个特征进行预测")
    
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
        threshold = 0.5  # 默认预测阈值
        df_predictions['prediction'] = (df_predictions['average_probability'] >= threshold).astype(int)
    
    logger.info(f"预测完成，共预测 {len(df_predictions)} 个化合物")
    
    return df_predictions

def screen_compound_library(library_path, models, format='smi', batch_size=100, 
                           save_results=True):
    """
    虚拟筛选主函数
    
    参数:
        library_path: 化合物库文件路径
        models: 训练好的模型字典
        format: 化合物库格式 ('smi' 或 'csv')
        batch_size: 批处理大小
        save_results: 是否保存结果
    
    返回:
        screening_results: 筛选结果DataFrame
    """
    start_time = time.time()
    logger.info("开始虚拟筛选")
    
    # 1. 加载化合物库
    df = load_compound_library(library_path, format=format)
    
    # 2. 批量计算分子特征
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
    
    # 加载训练好的模型
    models_dir = RESULTS_DIR['models']
    logger.info(f"模型目录路径: {models_dir}")
    
    if os.path.exists(models_dir):
        # 列出目录中的所有文件
        all_files = os.listdir(models_dir)
        logger.info(f"目录中的文件: {all_files}")
        
        # 只加载模型文件，不加载scaler或预测结果文件
        model_files = [f for f in all_files if f.endswith('_model.pkl')]
        logger.info(f"匹配的模型文件: {model_files}")
        
        if model_files:
            models = {}
            for model_file in model_files:
                # 从文件名中提取模型名称
                model_name = model_file.split('_model_')[0]
                model_path = os.path.join(models_dir, model_file)
                logger.info(f"加载模型: {model_path}")
                models[model_name] = joblib.load(model_path)
            logger.info(f"成功加载 {len(models)} 个模型: {list(models.keys())}")
            
            # 进行虚拟筛选
            library_path = os.path.join(DATA_DIR['raw'], 'example_library.smi')
            logger.info(f"化合物库路径: {library_path}")
            
            if os.path.exists(library_path):
                screen_compound_library(library_path, models)
            else:
                logger.error(f"化合物库文件不存在: {library_path}")
        else:
            logger.error(f"没有找到训练好的模型文件")
    else:
        logger.error(f"模型目录不存在: {models_dir}")
