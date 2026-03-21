#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 数据目录
DATA_DIR = {
    'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'processed': os.path.join(PROJECT_ROOT, 'data', 'processed')
}

# 结果目录
RESULTS_DIR = {
    'models': os.path.join(PROJECT_ROOT, 'results', 'models'),
    'figures': os.path.join(PROJECT_ROOT, 'results', 'figures'),
    'tables': os.path.join(PROJECT_ROOT, 'results', 'tables'),
    'reports': os.path.join(PROJECT_ROOT, 'results', 'reports')
}

# 源代码目录
SRC_DIR = {
    'data_acquisition': os.path.join(PROJECT_ROOT, 'src', 'data_acquisition'),
    'feature_engineering': os.path.join(PROJECT_ROOT, 'src', 'feature_engineering'),
    'modeling': os.path.join(PROJECT_ROOT, 'src', 'modeling'),
    'virtual_screening': os.path.join(PROJECT_ROOT, 'src', 'virtual_screening'),
    'analysis': os.path.join(PROJECT_ROOT, 'src', 'analysis')
}

# 脚本目录
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

# 文档目录
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

# 数据配置
DATA_CONFIG = {
    'chembl_data_file': 'dengue_antiviral_data.csv',
    'processed_data_file': 'processed_dengue_data.csv',
    'test_data_file': 'test_compounds.csv'
}

# 特征工程配置
FEATURE_CONFIG = {
    'fingerprint_types': ['Morgan', 'MACCS'],
    'morgan_radius': 2,
    'morgan_bits': 1024,
    'desc_types': ['rdkit_desc']
}

# 模型配置
MODEL_CONFIG = {
    'models': {
        'RandomForest': {
            'n_estimators': 1000,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'XGBoost': {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'SVM': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42
        }
    },
    'ensemble': {
        'voting': 'soft',
        'weights': [1, 1, 1]
    },
    'cv': 5,
    'random_state': 42
}

# 虚拟筛选配置
SCREENING_CONFIG = {
    'library_file': 'chembl_library.smi',
    'top_n_compounds': 100,
    'batch_size': 1000
}

# PubChem配置
PUBCHEM_CONFIG = {
    'default_batch_size': 100,
    'default_delay': 0.2,
    'max_retries': 3,
    'timeout': 30,
    'max_compounds_per_search': 1000
}

# 活性阈值配置
ACTIVITY_CONFIG = {
    'pic50_threshold': 6.0,  # pIC50 >= 6.0 视为活性
    'ic50_threshold': 1000.0  # IC50 <= 1000 nM 视为活性
}

# 创建所有目录
for dir_path in list(DATA_DIR.values()) + list(RESULTS_DIR.values()) + list(SRC_DIR.values()):
    os.makedirs(dir_path, exist_ok=True)
