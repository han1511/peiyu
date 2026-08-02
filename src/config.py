#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件

配置项目路径、数据参数、特征参数等
"""

import os
from pathlib import Path

# ============================================================================
# 项目路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 数据目录
DATA_DIR = {
    'raw': PROJECT_ROOT / 'data' / 'raw',
    'processed': PROJECT_ROOT / 'data' / 'processed',
    'external': PROJECT_ROOT / 'data' / 'external',
}

# 结果目录
RESULTS_DIR = {
    'models': PROJECT_ROOT / 'results' / 'models',
    'figures': PROJECT_ROOT / 'results' / 'figures',
    'tables': PROJECT_ROOT / 'results' / 'tables',
    'reports': PROJECT_ROOT / 'results' / 'reports',
}

# 创建必要的目录
for d in list(DATA_DIR.values()) + list(RESULTS_DIR.values()):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据配置
# ============================================================================

DATA_CONFIG = {
    'chembl_data_file': 'dengue_chembl_data.csv',
    'pubchem_data_file': 'pubchem_compounds.csv',
    'example_library_file': 'example_library.smi',
    'activity_threshold_nm': 1000,  # IC50 <= 1000 nM 视为活性
}

# ============================================================================
# PubChem配置
# ============================================================================

PUBCHEM_CONFIG = {
    'batch_size': 100,
    'delay': 0.5,
    'max_retries': 3,
    'timeout': 30,
    'checkpoint_interval': 1000,
}

# ============================================================================
# 特征配置
# ============================================================================

FEATURE_CONFIG = {
    'morgan': {
        'radius': 2,
        'n_bits': 1024,
        'use_features': False,
    },
    'maccs': {
        'enabled': True,
    },
    'rdkit_desc': {
        'enabled': True,
    },
}

# ============================================================================
# 活性配置
# ============================================================================

ACTIVITY_CONFIG = {
    'active_threshold': 1000,  # nM
    'intermediate_threshold': 10000,  # nM
    'units': 'nM',
}

# ============================================================================
# 模型配置
# ============================================================================

MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'models_to_train': ['RandomForest', 'XGBoost'],
}
