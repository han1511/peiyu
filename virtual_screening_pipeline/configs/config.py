#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟筛选管道配置文件
用于登革病毒抑制剂研发的完整虚拟筛选流程

作者：研究团队
版本：1.0.0
"""

import os
from pathlib import Path

# ============================================================================
# 项目路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 子目录
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
COMPOUND_LIBRARY_DIR = DATA_DIR / "compound_libraries"
TARGET_STRUCTURES_DIR = DATA_DIR / "target_structures"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, COMPOUND_LIBRARY_DIR, TARGET_STRUCTURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 登革病毒靶点配置
# ============================================================================

DENGUE_TARGETS = {
    "NS2A": {
        "pdb_id": None,
        "uniprot_id": "Q9Y8C8",
        "name": "Dengue virus NS2A protein",
        "function": "Viral replication complex component",
        "binding_site": None,
        "notes": "Membrane-associated protein involved in viral replication"
    },
    "NS3": {
        "pdb_id": None,
        "uniprot_id": "Q9Y8C9",
        "name": "Dengue virus NS3 protease/helicase",
        "function": "Serine protease and RNA helicase",
        "binding_site": None,
        "notes": "Essential for viral polyprotein processing and RNA replication"
    },
    "NS5": {
        "pdb_id": None,
        "uniprot_id": "Q9Y8D0",
        "name": "Dengue virus NS5 methyltransferase/RdRp",
        "function": "RNA-dependent RNA polymerase and methyltransferase",
        "binding_site": None,
        "notes": "Core enzyme for viral RNA synthesis"
    },
    "Envelope": {
        "pdb_id": None,
        "uniprot_id": "Q9Y8C6",
        "name": "Dengue virus envelope glycoprotein",
        "function": "Membrane fusion and cell entry",
        "binding_site": None,
        "notes": "Target for neutralizing antibodies and entry inhibitors"
    }
}

# ============================================================================
# 化合物库配置
# ============================================================================

COMPOUND_LIBRARIES = {
    "pubchem_100k": {
        "source": "PubChem",
        "path": RAW_DATA_DIR / "pubchem_100k_compounds.csv",
        "description": "100K compounds from PubChem",
        "size": 100000
    },
    "zinc_natural": {
        "source": "ZINC15",
        "path": COMPOUND_LIBRARY_DIR / "zinc_natural_products.smi",
        "description": "Natural product library from ZINC15",
        "size": None
    },
    "chembl_natural": {
        "source": "ChEMBL",
        "path": COMPOUND_LIBRARY_DIR / "chembl_natural_products.smi",
        "description": "Natural products from ChEMBL database",
        "size": None
    },
    "fda_drugs": {
        "source": "FDA",
        "path": COMPOUND_LIBRARY_DIR / "fda_approved_drugs.smi",
        "description": "FDA-approved drugs for drug repurposing",
        "size": None
    }
}

# ============================================================================
# 分子对接配置
# ============================================================================

DOCKING_CONFIG = {
    "software": "AutoDock Vina",
    "exhaustiveness": 32,
    "num_poses": 20,
    "search_space": {
        "center_x": None,
        "center_y": None,
        "center_z": None,
        "size_x": 22.5,
        "size_y": 22.5,
        "size_z": 22.5
    },
    "energy_range": 3.0,
    "cpu_cores": -1,
    "docking_timeout": 600
}

# 对接结合能阈值 (kcal/mol)
BINDING_AFFINITY_THRESHOLD = {
    "strong": -9.0,
    "moderate": -7.0,
    "weak": -5.0
}

# ============================================================================
# 机器学习模型配置
# ============================================================================

ML_MODELS = {
    "XGBoost": {
        "n_estimators": 1000,
        "max_depth": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"
    },
    "RandomForest": {
        "n_estimators": 1000,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    },
    "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "random_state": 42
    },
    "LogisticRegression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    }
}

# 模型性能评估阈值
MODEL_PERFORMANCE_THRESHOLDS = {
    "auc": 0.80,
    "sensitivity": 0.80,
    "specificity": 0.80,
    "f1_score": 0.75
}

# ============================================================================
# 分子特征配置
# ============================================================================

FEATURE_CONFIG = {
    "fingerprints": {
        "Morgan": {
            "radius": 2,
            "nBits": 2048,
            "use_features": True
        },
        "MACCS": {
            "nBits": 167
        },
        "Topological": {
            "nBits": 2048
        }
    },
    "descriptors": {
        "molecular_weight": {"enabled": True},
        "logp": {"enabled": True},
        "tpsa": {"enabled": True},
        "num_h_donors": {"enabled": True},
        "num_h_acceptors": {"enabled": True},
        "num_rotatable_bonds": {"enabled": True},
        "num_aromatic_rings": {"enabled": True},
        "fraction_csp3": {"enabled": True},
        "num_heavy_atoms": {"enabled": True},
        "balaban_j": {"enabled": True},
        "bertz_ct": {"enabled": True},
        "hall_kier_alpha": {"enabled": True},
        "kappa1": {"enabled": True},
        "kappa2": {"enabled": True},
        "kappa3": {"enabled": True},
        "chi0": {"enabled": True},
        "chi1": {"enabled": True}
    }
}

# ============================================================================
# Lipinski类药规则配置
# ============================================================================

LIPINSKI_RULES = {
    "molecular_weight": {"min": 0, "max": 500},
    "logp": {"min": -2, "max": 5},
    "h_donors": {"max": 5},
    "h_acceptors": {"max": 10},
    "tpsa": {"min": 0, "max": 140},
    "num_rotatable_bonds": {"max": 10}
}

# 扩展类药规则
DRUG_LIKENESS_FILTERS = {
    "PAINS": {"enabled": True, "max_hits": 0},
    "BRENK": {"enabled": True, "max_hits": 0},
    "zinc": {"enabled": False}
}

# ============================================================================
# ADMET配置
# ============================================================================

ADMET_THRESHOLDS = {
    "absorption": {
        "human_intestinal_absorption": {"low": 0, "high": 30},
        "caco2_permeability": {"low": 0, "high": 8}
    },
    "metabolism": {
        "cyp3a4_inhibition": {"threshold": 0.5},
        "cyp2c9_inhibition": {"threshold": 0.5}
    },
    "toxicity": {
        "ames_toxicity": {"threshold": 0.5},
        "hERG_inhibition": {"threshold": 0.5},
        "LD50": {"min": 500}
    }
}

# ============================================================================
# 数据集划分配置
# ============================================================================

DATA_SPLIT_CONFIG = {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,
    "stratify": True,
    "random_state": 42
}

# ============================================================================
# 交叉验证配置
# ============================================================================

CROSS_VALIDATION_CONFIG = {
    "n_folds": 5,
    "stratified": True,
    "shuffle": True,
    "random_state": 42
}

# ============================================================================
# 输出报告配置
# ============================================================================

REPORT_CONFIG = {
    "format": ["markdown", "html"],
    "include_figures": True,
    "include_tables": True,
    "precision": 4,
    "max_compounds_to_display": 100
}

# ============================================================================
# 日志配置
# ============================================================================

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "virtual_screening.log",
    "console_output": True
}

# ============================================================================
# 并行计算配置
# ============================================================================

PARALLEL_CONFIG = {
    "n_jobs": -1,
    "batch_size": 1000,
    "verbose": 10
}

# ============================================================================
# 文件格式配置
# ============================================================================

FILE_FORMATS = {
    "sdf": ".sdf",
    "mol": ".mol",
    "smiles": ".smi",
    "pdbqt": ".pdbqt",
    "csv": ".csv",
    "json": ".json",
    "pkl": ".pkl"
}

# ============================================================================
# 常量定义
# ============================================================================

# 活性阈值
ACTIVITY_THRESHOLDS = {
    "ic50_nM": 1000,  # IC50 <= 1 μM 为活性
    "ec50_nM": 1000,  # EC50 <= 1 μM 为活性
    "ki_nM": 100       # Ki <= 100 nM 为高活性
}

# pIC50计算
def calculate_pIC50(ic50_nM):
    """计算pIC50值"""
    if ic50_nM <= 0:
        return 0
    return -np.log10(ic50_nM * 1e-9)

# 导入numpy供calculate_pIC50使用
import numpy as np
