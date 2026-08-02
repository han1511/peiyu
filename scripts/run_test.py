#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：抗登革病毒药物筛选流程

这个脚本用于测试整个抗登革病毒药物筛选流程，包括：
1. 数据加载
2. 特征工程（计算分子描述符和指纹）
3. 模型训练（训练机器学习模型）
4. 虚拟筛选（筛选化合物库）
5. 结果分析
"""

import os
import sys
import time
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 添加virtual_screening_pipeline到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'virtual_screening_pipeline')))

# 导入配置
from configs.config import PROJECT_ROOT, RESULTS_DIR, DATA_DIR

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查依赖包"""
    logger.info("=== 检查依赖包 ===")
    
    deps_ok = True
    
    # 核心依赖
    try:
        import numpy
        logger.info("  OK: numpy")
    except ImportError as e:
        logger.error(f"  MISSING: numpy - {e}")
        deps_ok = False
    
    try:
        import pandas
        logger.info("  OK: pandas")
    except ImportError as e:
        logger.error(f"  MISSING: pandas - {e}")
        deps_ok = False
    
    try:
        import sklearn
        logger.info("  OK: scikit-learn")
    except ImportError as e:
        logger.error(f"  MISSING: scikit-learn - {e}")
        deps_ok = False
    
    try:
        from rdkit import Chem
        logger.info("  OK: rdkit")
    except ImportError as e:
        logger.error(f"  MISSING: rdkit - {e}")
        deps_ok = False
    
    # 可选依赖
    try:
        import xgboost
        logger.info("  OK: xgboost (optional)")
    except ImportError:
        logger.info("  OPTIONAL: xgboost not available")
    
    try:
        import torch
        logger.info("  OK: torch (optional)")
    except ImportError:
        logger.info("  OPTIONAL: torch not available")
    
    return deps_ok


def test_compound_library():
    """测试化合物库模块"""
    logger.info("=== 测试化合物库模块 ===")
    try:
        from src.compound_library import CompoundLibrary
        from rdkit import Chem
        
        library = CompoundLibrary("test_library")
        logger.info(f"化合物库模块加载成功")
        
        # 测试添加化合物
        mol = Chem.MolFromSmiles("CCO")  # 乙醇
        if mol:
            library.compounds.append({
                'name': 'ethanol',
                'mol': mol,
                'smiles': 'CCO',
                'standardized_smiles': 'CCO'
            })
            logger.info(f"添加化合物成功，当前库中化合物数: {len(library.compounds)}")
        
        return True
    except Exception as e:
        logger.error(f"化合物库模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_molecular_features():
    """测试分子特征模块"""
    logger.info("=== 测试分子特征模块 ===")
    try:
        from src.molecular_features import FeatureEngineering
        from rdkit import Chem
        
        fe = FeatureEngineering()
        logger.info("特征工程模块加载成功")
        
        # 测试特征计算
        mol = Chem.MolFromSmiles("CCO")  # 乙醇
        if mol:
            features, names, _ = fe.calculate_all_features([mol])
            logger.info(f"特征计算成功，特征维度: {features.shape}")
        
        return True
    except Exception as e:
        logger.error(f"分子特征模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_screening():
    """测试机器学习筛选模块"""
    logger.info("=== 测试机器学习筛选模块 ===")
    try:
        from src.ml_screening import VirtualScreening, ModelTrainer
        
        screening = VirtualScreening()
        logger.info("机器学习筛选模块加载成功")
        
        # 测试模型创建
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # 创建简单测试数据
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # 训练测试模型
        trainer = ModelTrainer("RandomForest")
        trainer.train(X, y)
        logger.info("模型训练成功")
        
        # 测试预测
        X_test = np.random.rand(10, 10)
        predictions = trainer.predict(X_test)
        probabilities = trainer.predict_proba(X_test)
        logger.info(f"模型预测成功，预测结果: {len(predictions)} 个")
        
        return True
    except Exception as e:
        logger.error(f"机器学习筛选模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_admet_evaluation():
    """测试ADMET评估模块"""
    logger.info("=== 测试ADMET评估模块 ===")
    try:
        from src.admet_evaluation import ADMETCalculator
        from rdkit import Chem
        
        calculator = ADMETCalculator()
        logger.info("ADMET评估模块加载成功")
        
        # 测试ADMET计算
        mol = Chem.MolFromSmiles("CCO")  # 乙醇
        if mol:
            result = calculator.calculate_all_admet(mol)
            logger.info(f"ADMET计算成功，包含 {len(result)} 个性质")
        
        return True
    except Exception as e:
        logger.error(f"ADMET评估模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行完整测试流程"""
    logger.info("=" * 60)
    logger.info("开始抗登革病毒药物筛选流程测试")
    logger.info("=" * 60)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    
    start_time = time.time()
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，请安装缺失的依赖包")
        return 1
    
    # 测试各个模块
    results = {
        'compound_library': test_compound_library(),
        'molecular_features': test_molecular_features(),
        'ml_screening': test_ml_screening(),
        'admet_evaluation': test_admet_evaluation(),
    }
    
    end_time = time.time()
    
    # 打印测试结果摘要
    logger.info("=" * 60)
    logger.info("测试摘要")
    logger.info("=" * 60)
    for module, success in results.items():
        status = "通过" if success else "失败"
        logger.info(f"  {module}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n所有模块测试通过！")
    else:
        logger.warning("\n部分模块测试失败，请检查日志")
    
    logger.info(f"测试完成，总耗时: {end_time - start_time:.2f} 秒")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
