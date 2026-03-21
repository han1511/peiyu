#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练和评估模块
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, average_precision_score, matthews_corrcoef,
                            classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import json
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置
from src.config import DATA_DIR, RESULTS_DIR, DATA_CONFIG, MODEL_CONFIG, ACTIVITY_CONFIG


def load_processed_data(data_path: str = None, target_column: str = 'active') -> tuple:
    """
    加载处理后的数据
    
    参数:
        data_path: 处理后的数据文件路径
        target_column: 目标列名
    
    返回:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR['processed'], DATA_CONFIG['processed_data_file'])
    
    print(f"正在加载处理后的数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 特征和目标变量
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col != target_column]
    
    X = df[feature_cols]
    y = df[target_column]
    
    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(df)}")
    print(f"活性化合物数: {sum(y)}")
    print(f"非活性化合物数: {sum(1 - y)}")
    print(f"类不平衡比例: {sum(y) / len(y):.2%} / {sum(1 - y) / len(y):.2%}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=MODEL_CONFIG['random_state']
    )
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集活性比例: {sum(y_train) / len(y_train):.2%}")
    print(f"测试集活性比例: {sum(y_test) / len(y_test):.2%}")
    
    return X_train, X_test, y_train, y_test


def handle_class_imbalance(X, y, method: str = 'SMOTE') -> tuple:
    """
    处理类不平衡问题
    
    参数:
        X: 特征数据
        y: 目标变量
        method: 处理方法 (SMOTE, ADASYN, SMOTEENN)
    
    返回:
        tuple: (X_resampled, y_resampled)
    """
    print(f"\n正在使用{method}处理类不平衡...")
    print(f"原始样本数: {len(y)}")
    print(f"原始活性化合物数: {sum(y)}")
    print(f"原始非活性化合物数: {sum(1 - y)}")
    
    if method == 'SMOTE':
        # 计算少数类样本数量
        active_count = sum(y)
        inactive_count = len(y) - active_count
        minority_count = min(active_count, inactive_count)
        
        if minority_count < 2:
            print("警告：少数类样本数量太少，无法使用SMOTE")
            # 如果样本数太少，直接返回原始数据
            return X, y
        else:
            # 调整n_neighbors参数，确保不超过少数类样本数量-1
            n_neighbors = int(min(5, minority_count - 1))
            sampler = SMOTE(random_state=MODEL_CONFIG['random_state'], k_neighbors=n_neighbors)
    elif method == 'ADASYN':
        # 计算少数类样本数量
        active_count = sum(y)
        inactive_count = len(y) - active_count
        minority_count = min(active_count, inactive_count)
        
        if minority_count < 2:
            print("警告：少数类样本数量太少，无法使用ADASYN")
            # 如果样本数太少，直接返回原始数据
            return X, y
        else:
            sampler = ADASYN(random_state=MODEL_CONFIG['random_state'])
    elif method == 'SMOTEENN':
        # 计算少数类样本数量
        active_count = sum(y)
        inactive_count = len(y) - active_count
        minority_count = min(active_count, inactive_count)
        
        if minority_count < 2:
            print("警告：少数类样本数量太少，无法使用SMOTEENN")
            # 如果样本数太少，直接返回原始数据
            return X, y
        else:
            # 调整SMOTE部分的n_neighbors参数
             n_neighbors = int(min(5, minority_count - 1))
             sampler = SMOTEENN(random_state=MODEL_CONFIG['random_state'], 
                               smote=SMOTE(random_state=MODEL_CONFIG['random_state'], k_neighbors=n_neighbors))
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    print(f"重采样后样本数: {len(y_resampled)}")
    print(f"重采样后活性化合物数: {sum(y_resampled)}")
    print(f"重采样后非活性化合物数: {sum(1 - y_resampled)}")
    
    return X_resampled, y_resampled


def train_model(X_train, y_train, model_name: str, handle_imbalance: bool = True) -> tuple:
    """
    训练单个模型
    
    参数:
        X_train: 训练特征
        y_train: 训练目标
        model_name: 模型名称
        handle_imbalance: 是否处理类不平衡
    
    返回:
        tuple: (model, scaler)
    """
    print(f"\n正在训练{model_name}模型...")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 处理类不平衡
    if handle_imbalance:
        X_train_scaled, y_train = handle_class_imbalance(X_train_scaled, y_train)
    
    # 获取模型配置
    model_config = MODEL_CONFIG['models'][model_name]
    
    # 初始化模型
    if model_name == 'RandomForest':
        model = RandomForestClassifier(**model_config)
    elif model_name == 'XGBoost':
        model = XGBClassifier(**model_config)
    elif model_name == 'SVM':
        model = SVC(**model_config, probability=True)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    print(f"{model_name}模型训练完成！")
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, model_name: str) -> dict:
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        scaler: 标准化器
        X_test: 测试特征
        y_test: 测试目标
        model_name: 模型名称
    
    返回:
        dict: 评估指标
    """
    print(f"\n正在评估{model_name}模型...")
    
    # 标准化
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算性能指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_prob),
        'pr_auc': average_precision_score(y_test, y_pred_prob),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    
    print(f"{model_name}模型性能:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 分类报告
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_pred_prob


def train_ensemble(X_train, y_train, models: list, model_names: list, handle_imbalance: bool = True) -> tuple:
    """
    训练集成模型
    
    参数:
        X_train: 训练特征
        y_train: 训练目标
        models: 基础模型列表
        model_names: 模型名称列表
        handle_imbalance: 是否处理类不平衡
    
    返回:
        tuple: (ensemble_model, scaler)
    """
    print(f"\n正在训练集成模型...")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 处理类不平衡
    if handle_imbalance:
        X_train_scaled, y_train = handle_class_imbalance(X_train_scaled, y_train)
    
    # 创建投票分类器
    estimators = list(zip(model_names, models))
    ensemble_model = VotingClassifier(
        estimators=estimators,
        voting=MODEL_CONFIG['ensemble']['voting'],
        weights=MODEL_CONFIG['ensemble']['weights']
    )
    
    # 训练集成模型
    ensemble_model.fit(X_train_scaled, y_train)
    
    print("集成模型训练完成！")
    
    return ensemble_model, scaler


def cross_validation(X, y, model_name: str, cv: int = None, handle_imbalance: bool = True) -> dict:
    """
    交叉验证
    
    参数:
        X: 特征数据
        y: 目标变量
        model_name: 模型名称
        cv: 交叉验证折数
        handle_imbalance: 是否处理类不平衡
    
    返回:
        dict: 交叉验证结果
    """
    if cv is None:
        cv = MODEL_CONFIG['cv']
    
    print(f"\n正在进行{cv}折交叉验证...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=MODEL_CONFIG['random_state'])
    
    cv_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': [],
        'pr_auc': [],
        'mcc': []
    }
    
    fold = 1
    for train_idx, val_idx in tqdm(skf.split(X, y), total=cv, desc="交叉验证"):
        print(f"\nFold {fold}:")
        fold += 1
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model, scaler = train_model(X_train_fold, y_train_fold, model_name, handle_imbalance)
        
        # 评估模型
        metrics, _, _ = evaluate_model(model, scaler, X_val_fold, y_val_fold, model_name)
        
        # 保存指标
        for metric_name, value in metrics.items():
            cv_metrics[metric_name].append(value)
    
    # 计算平均指标
    print(f"\n{cv}折交叉验证平均结果:")
    avg_metrics = {}
    for metric_name, values in cv_metrics.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        avg_metrics[f'{metric_name}_mean'] = avg_value
        avg_metrics[f'{metric_name}_std'] = std_value
        print(f"{metric_name}: {avg_value:.4f} ± {std_value:.4f}")
    
    return avg_metrics


def save_model(model, scaler, model_name: str) -> str:
    """
    保存模型和标准化器
    
    参数:
        model: 训练好的模型
        scaler: 标准化器
        model_name: 模型名称
    
    返回:
        str: 模型保存路径
    """
    model_path = os.path.join(RESULTS_DIR['models'], f'{model_name}_model.pkl')
    scaler_path = os.path.join(RESULTS_DIR['models'], f'{model_name}_scaler.pkl')
    
    print(f"\n正在保存{model_name}模型到: {model_path}")
    print(f"正在保存{model_name}标准化器到: {scaler_path}")
    
    # 保存模型和标准化器
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model_path


def save_results(results: dict, results_name: str) -> str:
    """
    保存评估结果
    
    参数:
        results: 评估结果
        results_name: 结果名称
    
    返回:
        str: 结果保存路径
    """
    results_path = os.path.join(RESULTS_DIR['models'], f'{results_name}_results.json')
    
    print(f"正在保存结果到: {results_path}")
    
    # 保存结果
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)
    
    return results_path


def train_classification_model(input_df=None, models_to_train=None, balance_data: bool = True, cross_val: bool = False) -> dict:
    """
    运行完整的模型训练和评估流程
    
    参数:
        input_df: 输入的DataFrame，包含特征和目标变量
        models_to_train: 要训练的模型名称列表，默认为None（训练所有模型）
        balance_data: 是否处理类不平衡
        cross_val: 是否进行交叉验证
    
    返回:
        tuple: (models_dict, all_results, df_features)，包含训练好的模型、结果和特征DataFrame
    """
    print("=" * 80)
    print("开始模型训练和评估流程")
    print("=" * 80)
    
    # 加载或使用输入数据
    if input_df is not None:
        df_features = input_df.copy()
        target_column = 'active'
        
        # 检查是否有目标列
        if target_column not in df_features.columns:
            raise ValueError(f"输入DataFrame中没有找到目标列 '{target_column}'")
            
        # 特征和目标变量
        feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != target_column]
        
        # 检查并移除包含NaN值的行
        df_features = df_features.dropna(subset=[target_column] + feature_cols)
        print(f"移除NaN值后的数据数量: {len(df_features)}")
        
        X = df_features[feature_cols]
        y = df_features[target_column]
        
        # 再次检查是否还有NaN值
        print(f"X中的NaN值数量: {X.isna().sum().sum()}")
        print(f"y中的NaN值数量: {y.isna().sum()}")
        
        print(f"特征数: {len(feature_cols)}")
        print(f"样本数: {len(df_features)}")
        print(f"活性化合物数: {sum(y)}")
        print(f"非活性化合物数: {sum(1 - y)}")
        print(f"类不平衡比例: {sum(y) / len(y):.2%} / {sum(1 - y) / len(y):.2%}")
        
        # 保存特征列表
        feature_list_path = os.path.join(DATA_DIR['processed'], 'feature_list.pkl')
        print(f"保存特征列表到: {feature_list_path}")
        joblib.dump(feature_cols, feature_list_path)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=MODEL_CONFIG['random_state']
        )
        
        print(f"\n训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"训练集活性比例: {sum(y_train) / len(y_train):.2%}")
        print(f"测试集活性比例: {sum(y_test) / len(y_test):.2%}")
    else:
        # 从文件加载数据
        X_train, X_test, y_train, y_test = load_processed_data()
        df_features = pd.read_csv(os.path.join(DATA_DIR['processed'], DATA_CONFIG['processed_data_file']))
        
        # 获取特征列表
        feature_cols = X_train.columns.tolist()
        
        # 保存特征列表
        feature_list_path = os.path.join(DATA_DIR['processed'], 'feature_list.pkl')
        print(f"保存特征列表到: {feature_list_path}")
        joblib.dump(feature_cols, feature_list_path)
    
    # 所有模型的结果
    all_results = {}
    
    # 训练和评估每个模型
    models = []
    model_names = []
    trained_models = {}
    
    # 确定要训练的模型
    if models_to_train is None:
        models_to_train = MODEL_CONFIG['models'].keys()
    
    for model_name in MODEL_CONFIG['models'].keys():
        # 只训练指定的模型（不区分大小写）
        if model_name.upper() not in [m.upper() for m in models_to_train]:
            continue
        print("\n" + "-" * 60)
        print(f"模型: {model_name}")
        print("-" * 60)
        
        # 训练模型
        model, scaler = train_model(X_train, y_train, model_name, balance_data)
        
        # 交叉验证
        if cross_val:
            cv_results = cross_validation(X_train, y_train, model_name, handle_imbalance=balance_data)
        else:
            cv_results = {}
        
        # 评估模型
        test_metrics, y_pred, y_pred_prob = evaluate_model(model, scaler, X_test, y_test, model_name)
        
        # 保存模型和标准化器
        save_model(model, scaler, model_name)
        
        # 保存测试集预测结果
        test_results = {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_prob': y_pred_prob.tolist()
        }
        joblib.dump(test_results, os.path.join(RESULTS_DIR['models'], f'{model_name}_test_predictions.pkl'))
        
        # 保存结果
        model_results = {
            'test_metrics': test_metrics,
            'cv_results': cv_results
        }
        save_results(model_results, model_name)
        
        all_results[model_name] = model_results
        
        # 添加到集成模型
        models.append(model)
        model_names.append(model_name)
        
        # 保存训练好的模型
        trained_models[model_name] = (model, scaler)
    
    # 比较所有模型
    print("\n" + "=" * 80)
    print("模型性能比较")
    print("=" * 80)
    
    comparison_df = pd.DataFrame()
    for model_name, results in all_results.items():
        comparison_df[model_name] = pd.Series(results['test_metrics'])
    
    print(comparison_df.round(4))
    
    # 保存比较结果
    comparison_df.to_csv(os.path.join(RESULTS_DIR['models'], 'model_comparison.csv'))
    
    # 如果有多个模型被训练且'ENSEMBLE'在models_to_train中，训练集成模型
    if len(models) > 1 and 'ENSEMBLE' in models_to_train:
        print("\n" + "-" * 60)
        print("模型: Ensemble")
        print("-" * 60)
        
        # 训练集成模型
        ensemble_model, ensemble_scaler = train_ensemble(X_train, y_train, models, model_names, balance_data)
        
        # 评估集成模型
        ensemble_metrics, ensemble_pred, ensemble_pred_prob = evaluate_model(
            ensemble_model, ensemble_scaler, X_test, y_test, "Ensemble"
        )
        
        # 保存集成模型
        save_model(ensemble_model, ensemble_scaler, "Ensemble")
        
        # 保存测试集预测结果
        ensemble_test_results = {
            'y_true': y_test.tolist(),
            'y_pred': ensemble_pred.tolist(),
            'y_pred_prob': ensemble_pred_prob.tolist()
        }
        joblib.dump(ensemble_test_results, os.path.join(RESULTS_DIR['models'], 'Ensemble_test_predictions.pkl'))
        
        # 保存集成模型结果
        ensemble_results = {
            'test_metrics': ensemble_metrics
        }
        save_results(ensemble_results, "Ensemble")
        all_results["Ensemble"] = ensemble_results
        trained_models["Ensemble"] = (ensemble_model, ensemble_scaler)
    
    print("\n" + "=" * 80)
    print("模型训练和评估流程完成！")
    print("=" * 80)
    
    return trained_models, all_results, df_features


if __name__ == "__main__":
    # 运行模型训练和评估流程
    trained_models, all_results, df_features = train_classification_model(balance_data=True, cross_val=True)
