#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从ChEMBL下载登革病毒NS5抑制剂的活性数据用于模型训练

直接使用ChEMBL REST API，不依赖requests_cache
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    from rdkit import Chem
    print("RDKit和Pandas已安装")
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

import requests

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get_chembl_data(endpoint, params=None):
    """直接调用ChEMBL REST API"""
    url = f"{CHEMBL_API_BASE}/{endpoint}"
    params = params or {}
    params['format'] = 'json'

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()

def download_training_data(output_file="DENV_NS5_training_data.csv", target_chembl_id="CHEMBL3130"):
    """
    从ChEMBL下载训练数据
    """
    print("=" * 60)
    print("从ChEMBL下载登革病毒NS5训练数据")
    print("靶点: CHEMBL3130 (Dengue virus NS5)")
    print("=" * 60)

    print("\n[1/4] 下载活性化合物 (Label=1, IC50/EC50/Ki ≤ 10 μM)...")

    df_active = pd.DataFrame()
    try:
        params = {
            'target_chembl_id': target_chembl_id,
            'standard_type__in': 'IC50,EC50,Ki',
            'standard_value__lte': 10000,
            'limit': 10000
        }

        data = get_chembl_data('activity', params)

        if 'activities' in data:
            df_active = pd.DataFrame(data['activities'])
            print(f"  原始活性数据: {len(df_active)} 条")

    except Exception as e:
        print(f"  下载活性数据出错: {e}")
        import traceback
        traceback.print_exc()

    if len(df_active) > 0 and 'canonical_smiles' in df_active.columns:
        df_active = df_active.dropna(subset=["canonical_smiles"])
        df_active = df_active.drop_duplicates(subset=["canonical_smiles"])
        df_active["Label"] = 1
        print(f"  去重后活性数据: {len(df_active)} 条")

    print(f"\n[2/4] 下载非活性化合物 (Label=0, IC50 > 10 μM)...")

    df_inactive = pd.DataFrame()
    try:
        params = {
            'target_chembl_id': target_chembl_id,
            'standard_type__in': 'IC50,EC50,Ki',
            'standard_value__gt': 10000,
            'limit': 10000
        }

        data = get_chembl_data('activity', params)

        if 'activities' in data:
            df_inactive = pd.DataFrame(data['activities'])
            print(f"  原始非活性数据: {len(df_inactive)} 条")

    except Exception as e:
        print(f"  下载非活性数据出错: {e}")
        import traceback
        traceback.print_exc()

    if len(df_inactive) > 0 and 'canonical_smiles' in df_inactive.columns:
        df_inactive = df_inactive.dropna(subset=["canonical_smiles"])
        df_inactive = df_inactive.drop_duplicates(subset=["canonical_smiles"])
        df_inactive["Label"] = 0
        print(f"  去重后非活性数据: {len(df_inactive)} 条")

    print("\n[3/4] 合并并验证数据...")

    if len(df_active) > 0 and len(df_inactive) > 0:
        df = pd.concat([df_active, df_inactive], ignore_index=True)
    elif len(df_active) > 0:
        print("  警告: 没有非活性数据，使用活性数据")
        df = df_active.copy()
    else:
        print("  错误: 没有获取到任何数据")
        return None

    def validate_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    if 'canonical_smiles' in df.columns:
        df["valid"] = df["canonical_smiles"].apply(validate_smiles)
        df_valid = df[df["valid"] == True].copy()
        df_valid = df_valid.drop(columns=["valid"])

        print(f"  有效SMILES: {len(df_valid)} / {len(df)}")

        print(f"\n[4/4] 保存数据到 {output_file}...")

        df_result = df_valid[["canonical_smiles", "Label"]].copy()
        df_result.columns = ["SMILES", "Label"]

        df_result.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("数据集统计:")
        print(f"  总化合物数: {len(df_result)}")
        print(f"  活性化合物 (Label=1): {sum(df_result['Label'])}")
        print(f"  非活性化合物 (Label=0): {len(df_result) - sum(df_result['Label'])}")
        print(f"  保存至: {output_file}")
        print("=" * 60)

        return df_result
    else:
        print("  错误: 数据中没有canonical_smiles列")
        return None

if __name__ == "__main__":
    print("开始下载训练数据...")
    print("注意: 如果数据量较大，下载可能需要几分钟...")
    print("ChEMBL API可能需要较长时间响应，请耐心等待...")

    output_path = os.path.join(os.path.dirname(__file__), "DENV_NS5_training_data.csv")

    result = download_training_data(
        output_file=output_path,
        target_chembl_id="CHEMBL3130"
    )

    if result is not None and len(result) > 0:
        print("\n数据集下载成功!")
    else:
        print("\n数据集下载失败或数据为空")
        print("可能原因:")
        print("  1. 网络连接问题")
        print("  2. ChEMBL服务暂时不可用")
        print("  3. CHEMBL3130靶点没有相关数据")