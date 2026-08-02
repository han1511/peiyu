#!/usr/bin/env python
# scripts/build_model.py
import sqlite3
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# ========== 配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DB_PATH = os.path.join(DATA_DIR, "DenvInD.db")
ACTIVITY_THRESHOLD_nM = 1000    # IC50 < 1000 nM 视为活性

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ========== 辅助函数 ==========
def smiles_to_fp(smiles, radius=2, nBits=1024):
    """将 SMILES 转为 Morgan 指纹（二进制向量）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

def load_denvind(db_path):
    """从 DenvInD.db 加载数据，假设表名为 DenvInD，含有 SMILES 和 IC50_nM 列"""
    conn = sqlite3.connect(db_path)
    # 先查看有哪些表和列（调试用，正式可注释）
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)
    
    # 我们假设表名是 DenvInD（最常见），如果不是请修改这里
    table_name = "DenvInD"
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    conn.close()
    
    # 自动寻找包含 SMILES 和 IC50 的列（不区分大小写）
    smiles_col = None
    ic50_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
        if 'ic50' in col.lower():
            ic50_col = col
    if smiles_col is None or ic50_col is None:
        raise ValueError("找不到 SMILES 或 IC50 列，请手动检查数据库列名")
    
    df = df[[smiles_col, ic50_col]].dropna()
    df.columns = ['SMILES', 'IC50_nM']
    # 如果 IC50 值非常大（可能是 pIC50 或单位不是 nM），需要用户自行调整，这里简单假定单位已经是 nM
    df['label'] = (df['IC50_nM'] < ACTIVITY_THRESHOLD_nM).astype(int)
    print(f"Loaded {len(df)} compounds, active ratio: {df['label'].mean():.2f}")
    return df

# ========== 主流程 ==========
def main():
    # 1. 加载数据
    print("Loading DenvInD data...")
    df = load_denvind(DB_PATH)
    
    # 2. 计算指纹特征
    print("Computing fingerprints (this may take a few minutes)...")
    df['features'] = df['SMILES'].apply(smiles_to_fp)
    initial_len = len(df)
    df = df[df['features'].notna()]
    print(f"Dropped {initial_len - len(df)} invalid SMILES")
    
    X = np.vstack(df['features'].values)
    y = df['label'].values
    print(f"Feature matrix shape: {X.shape}")
    
    # 3. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. 训练随机森林（利用所有 CPU 核心）
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # 5. 评估
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.3f}")
    
    # 6. 保存模型
    model_path = os.path.join(OUTPUT_DIR, "anti_denv_small_mol_model.pkl")
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")
    
    # 7. （可选）保存特征提取函数，以便后续对新化合物预测
    import pickle
    with open(os.path.join(OUTPUT_DIR, "smiles_to_fp.pkl"), "wb") as f:
        pickle.dump(smiles_to_fp, f)
    print("All done.")

if __name__ == "__main__":
    main()