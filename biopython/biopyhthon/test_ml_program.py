#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高通量数据机器学习预测程序的脚本
"""

from ml_high_throughput import HighThroughputML
import pandas as pd
import numpy as np

def test_classification():
    """测试分类任务"""
    print("="*50)
    print("测试分类任务")
    print("="*50)
    
    # 创建ML对象
    ml = HighThroughputML()
    
    # 加载分类数据
    ml.load_data("example_classification_data.txt", target_column="class", sep="\t")
    
    # 预处理数据
    ml.preprocess_data(scale_method="standard", select_features=False)
    
    # 划分数据
    ml.split_data(test_size=0.3, random_state=42)
    
    # 训练随机森林分类器
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    ml.train_model(model_type="random_forest", param_grid=param_grid)
    
    # 评估模型
    ml.evaluate_model()
    
    # 测试预测功能
    print("\n测试预测功能:")
    new_data = pd.DataFrame({
        'feature1': [0.60, 1.30, -0.60],
        'feature2': [1.20, -0.50, 0.80],
        'feature3': [0.90, 1.00, -0.40],
        'feature4': [-0.50, 1.60, 1.00],
        'feature5': [2.00, -0.80, 1.30],
        'feature6': [-1.30, 2.20, -0.80],
        'feature7': [0.80, -0.40, 1.60],
        'feature8': [1.60, 0.60, -0.60],
        'feature9': [-1.00, 1.90, 0.90],
        'feature10': [2.30, -1.00, 1.40]
    })
    
    predictions = ml.predict(new_data)
    print(f"预测结果: {predictions}")
    
    # 保存模型
    ml.save_model("classification_model.pkl")
    print("\n分类任务测试完成！")

def test_regression():
    """测试回归任务"""
    print("\n" + "="*50)
    print("测试回归任务")
    print("="*50)
    
    # 创建ML对象
    ml = HighThroughputML()
    
    # 加载回归数据
    ml.load_data("example_regression_data.txt", target_column="target_value", sep="\t")
    
    # 预处理数据
    ml.preprocess_data(scale_method="minmax", select_features=False)
    
    # 划分数据
    ml.split_data(test_size=0.3, random_state=42)
    
    # 训练支持向量回归器
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.1, 1.0]
    }
    ml.train_model(model_type="svm", param_grid=param_grid)
    
    # 评估模型
    ml.evaluate_model()
    
    # 测试预测功能
    print("\n测试预测功能:")
    new_data = pd.DataFrame({
        'feature1': [0.60, 1.30, -0.60],
        'feature2': [1.20, -0.50, 0.80],
        'feature3': [0.90, 1.00, -0.40],
        'feature4': [-0.50, 1.60, 1.00],
        'feature5': [2.00, -0.80, 1.30],
        'feature6': [-1.30, 2.20, -0.80],
        'feature7': [0.80, -0.40, 1.60],
        'feature8': [1.60, 0.60, -0.60],
        'feature9': [-1.00, 1.90, 0.90],
        'feature10': [2.30, -1.00, 1.40]
    })
    
    predictions = ml.predict(new_data)
    print(f"预测结果: {predictions}")
    
    # 保存模型
    ml.save_model("regression_model.pkl")
    print("\n回归任务测试完成！")

if __name__ == "__main__":
    # 测试分类任务
    test_classification()
    
    # 测试回归任务
    test_regression()
    
    print("\n" + "="*50)
    print("所有测试完成！")
    print("="*50)