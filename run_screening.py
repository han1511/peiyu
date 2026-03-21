#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟筛选脚本
使用训练好的模型对化合物库进行筛选
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入配置
from src.config import DATA_DIR, RESULTS_DIR

# 导入功能模块
from src.feature_engineering.molecular_features import calculate_features
from src.virtual_screening.virtual_screening import screen_compound_library


def main():
    print("=" * 80)
    print("虚拟筛选 PubChem 化合物库")
    print("=" * 80)
    
    # 化合物库路径
    library_path = os.path.join(DATA_DIR['raw'], 'pubchem_100k_compounds.csv')
    
    # 检查文件是否存在
    if not os.path.exists(library_path):
        print(f"错误：化合物库文件不存在：{library_path}")
        return
    
    print(f"化合物库路径：{library_path}")
    
    # 加载训练好的模型
    print("\n加载训练好的模型...")
    models = {}
    
    model_names = ['RandomForest', 'XGBoost', 'SVM']
    
    for model_name in model_names:
        model_path = os.path.join(RESULTS_DIR['models'], f'{model_name}_model.pkl')
        scaler_path = os.path.join(RESULTS_DIR['models'], f'{model_name}_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            models[model_name] = (model, scaler)
            print(f"  ✓ {model_name}")
        else:
            print(f"  ✗ {model_name} (模型文件不存在)")
    
    if not models:
        print("\n错误：没有找到任何训练好的模型！")
        print("请先运行 research_pipeline.py 训练模型。")
        return
    
    print(f"\n成功加载 {len(models)} 个模型")
    
    # 运行虚拟筛选
    print("\n开始虚拟筛选...")
    print("=" * 80)
    
    try:
        results = screen_compound_library(
            library_path=library_path,
            models=models,
            format='csv',
            smiles_column='SMILES',
            batch_size=1000
        )
        
        print("\n" + "=" * 80)
        print("虚拟筛选完成！")
        print("=" * 80)
        
        # 显示筛选结果统计
        if results is not None and len(results) > 0:
            print(f"\n总筛选化合物数：{len(results)}")
            
            if 'prediction' in results.columns:
                active_count = results['prediction'].sum()
                inactive_count = len(results) - active_count
                hit_rate = active_count / len(results) * 100
                
                print(f"预测活性化合物数：{active_count}")
                print(f"预测非活性化合物数：{inactive_count}")
                print(f"命中率：{hit_rate:.2f}%")
            
            # 显示 Top 10 预测活性化合物
            if 'average_probability' in results.columns:
                print("\nTop 10 预测活性化合物:")
                print("-" * 80)
                
                top_compounds = results.nlargest(10, 'average_probability')
                
                for idx, row in top_compounds.iterrows():
                    cid = row.get('CID', 'N/A')
                    smiles = row.get('SMILES', 'N/A')
                    prob = row.get('average_probability', 0.0)
                    prediction = '活性' if row.get('prediction', 0) == 1 else '非活性'
                    
                    print(f"ID: {cid:10s} | 概率: {prob:.4f} | {prediction}")
                    print(f"SMILES: {smiles}")
                    print("-" * 80)
            
            # 保存筛选结果
            output_dir = os.path.join(RESULTS_DIR['models'], 'virtual_screening')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'screening_results_{timestamp}.csv')
            
            results.to_csv(output_path, index=False)
            print(f"\n筛选结果已保存：{output_path}")
            
            # 保存活性化合物
            if 'prediction' in results.columns:
                active_compounds = results[results['prediction'] == 1]
                if len(active_compounds) > 0:
                    active_path = os.path.join(output_dir, f'active_compounds_{timestamp}.csv')
                    active_compounds.to_csv(active_path, index=False)
                    print(f"活性化合物已保存：{active_path}")
        
        else:
            print("筛选结果为空！")
    
    except Exception as e:
        print(f"\n虚拟筛选失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("虚拟筛选流程完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
