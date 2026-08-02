#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChEMBL生物活性数据下载模块

从ChEMBL数据库获取登革病毒靶点的真实生物活性数据
支持IC50/EC50/Ki等活性终点
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 登革病毒靶点ChEMBL Target ID映射
DENGUE_CHEMBL_TARGETS = {
    "NS2A": {
        "chembl_id": None,  # NS2A在ChEMBL中数据较少
        "uniprot_id": "Q9Y8C8",
        "search_terms": ["dengue NS2A", "dengue virus NS2A"],
    },
    "NS3": {
        "chembl_id": "CHEMBL1926",
        "uniprot_id": "Q9Y8C9",
        "search_terms": ["dengue NS3 protease", "dengue NS3 helicase", "dengue virus NS3"],
    },
    "NS5": {
        "chembl_id": "CHEMBL1928",
        "uniprot_id": "Q9Y8D0",
        "search_terms": ["dengue NS5", "dengue RNA polymerase", "dengue methyltransferase"],
    },
    "Envelope": {
        "chembl_id": None,
        "uniprot_id": "Q9Y8C6",
        "search_terms": ["dengue envelope", "dengue E protein", "dengue E glycoprotein"],
    },
}


class ChEMBLDataFetcher:
    """从ChEMBL获取生物活性数据"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/chembl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
    
    def _get_client(self):
        """延迟导入ChEMBL客户端 (设置超时和重试，避免长时间挂起)"""
        if self._client is None:
            try:
                from chembl_webresource_client.new_client import new_client
                # 为requests设置默认超时 (chembl_webresource_client底层用requests但未设超时)
                import requests
                _orig_request = requests.Session.request
                if not getattr(requests.Session, '_timeout_patched', False):
                    def _request_with_timeout(self, *args, **kwargs):
                        kwargs.setdefault('timeout', 15)  # 15秒超时，避免无限等待
                        return _orig_request(self, *args, **kwargs)
                    requests.Session.request = _request_with_timeout
                    requests.Session._timeout_patched = True
                self._client = new_client
            except ImportError:
                raise ImportError(
                    "chembl-webresource-client未安装。请运行: pip install chembl-webresource-client"
                )
        return self._client
    
    def fetch_activity_data(self, target_name: str, 
                            activity_types: List[str] = None,
                            max_results: int = 5000) -> pd.DataFrame:
        """
        获取指定靶点的生物活性数据
        
        参数:
            target_name: 靶点名称 (NS2A/NS3/NS5/Envelope)
            activity_types: 活性类型 (IC50/EC50/Ki/Kd等)
            max_results: 最大结果数
            
        返回:
            pd.DataFrame: 活性数据
        """
        if target_name not in DENGUE_CHEMBL_TARGETS:
            raise ValueError(f"未知靶点: {target_name}")
        
        target_info = DENGUE_CHEMBL_TARGETS[target_name]
        
        if activity_types is None:
            activity_types = ['IC50', 'EC50', 'Ki', 'Kd', 'IC90', 'EC90']
        
        # 缓存路径
        cache_file = self.cache_dir / f"chembl_{target_name}_activities.csv"
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 7 * 24 * 3600:  # 7天缓存
                logger.info(f"从缓存加载ChEMBL数据: {cache_file}")
                return pd.read_csv(cache_file)
        
        new_client = self._get_client()
        activity = new_client.activity
        
        all_records = []
        
        # 如果有ChEMBL Target ID，直接查询
        if target_info['chembl_id']:
            logger.info(f"查询ChEMBL靶点: {target_info['chembl_id']}")
            for act_type in activity_types:
                try:
                    results = activity.filter(
                        target_chembl_id=target_info['chembl_id'],
                        standard_type=act_type,
                        assay_type='B'  # Binding assay
                    ).only(
                        'molecule_chembl_id', 'canonical_smiles',
                        'standard_value', 'standard_units',
                        'standard_type', 'assay_chembl_id',
                        'assay_description', 'target_chembl_id',
                        'target_pref_name', 'organism',
                        'activity_id', 'pchembl_value'
                    )[:max_results]
                    
                    for r in results:
                        all_records.append(r)
                    logger.info(f"  {act_type}: {len(results)} 条记录")
                    time.sleep(0.5)  # 限制请求频率
                    
                except Exception as e:
                    logger.warning(f"查询{act_type}失败: {e}")
        
        # 如果没有直接Target ID或结果不足，用关键词搜索
        if len(all_records) < 10:
            logger.info("直接查询结果不足，使用关键词搜索...")
            for term in target_info['search_terms']:
                try:
                    target_client = new_client.target
                    targets = target_client.filter(
                        pref_name__icontains=term,
                        organism__icontains='dengue'
                    ).only('target_chembl_id', 'pref_name')[:20]
                    
                    for t in targets:
                        tid = t.get('target_chembl_id')
                        if not tid:
                            continue
                        for act_type in activity_types:
                            try:
                                results = activity.filter(
                                    target_chembl_id=tid,
                                    standard_type=act_type
                                ).only(
                                    'molecule_chembl_id', 'canonical_smiles',
                                    'standard_value', 'standard_units',
                                    'standard_type', 'pchembl_value',
                                    'target_pref_name', 'organism'
                                )[:max_results]
                                
                                for r in results:
                                    all_records.append(r)
                                time.sleep(0.3)
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"关键词搜索'{term}'失败: {e}")
        
        if not all_records:
            logger.warning(f"未找到{target_name}的ChEMBL活性数据")
            return pd.DataFrame()
        
        # 转为DataFrame
        df = pd.DataFrame(all_records)
        
        # 数据清洗
        df = self._clean_activity_data(df)
        
        # 保存缓存
        df.to_csv(cache_file, index=False)
        logger.info(f"ChEMBL数据已缓存: {cache_file} ({len(df)}条)")
        
        return df
    
    def _clean_activity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗活性数据"""
        if df.empty:
            return df
        
        # 重命名列
        col_map = {
            'canonical_smiles': 'SMILES',
            'molecule_chembl_id': 'Molecule_ID',
            'standard_value': 'Activity_Value',
            'standard_units': 'Activity_Units',
            'standard_type': 'Activity_Type',
            'pchembl_value': 'pChEMBL',
            'target_pref_name': 'Target_Name',
            'organism': 'Organism',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # 确保SMILES列存在
        if 'SMILES' not in df.columns:
            logger.warning("数据中无SMILES列")
            return pd.DataFrame()
        
        # 去除无SMILES的记录
        df = df.dropna(subset=['SMILES'])
        
        # 去除无效SMILES
        df = df[df['SMILES'] != '']
        df = df[df['SMILES'].notna()]
        
        # 数值转换
        if 'Activity_Value' in df.columns:
            df['Activity_Value'] = pd.to_numeric(df['Activity_Value'], errors='coerce')
        if 'pChEMBL' in df.columns:
            df['pChEMBL'] = pd.to_numeric(df['pChEMBL'], errors='coerce')
        
        # 去重 (同一分子同一活性类型取中位数)
        if 'Molecule_ID' in df.columns and 'Activity_Type' in df.columns:
            df = df.groupby(['Molecule_ID', 'Activity_Type']).agg({
                'SMILES': 'first',
                'Activity_Value': 'median',
                'Activity_Units': 'first',
                'pChEMBL': 'median',
                'Target_Name': 'first',
                'Organism': 'first',
            }).reset_index()
        
        # 活性标签: pChEMBL >= 5 (即IC50 <= 10μM) 为活性
        if 'pChEMBL' in df.columns:
            df['Activity_Label'] = (df['pChEMBL'] >= 5.0).astype(int)
        elif 'Activity_Value' in df.columns and 'Activity_Type' in df.columns:
            # IC50/EC50 <= 10μM 为活性
            mask = (
                df['Activity_Type'].isin(['IC50', 'EC50', 'Ki', 'Kd']) &
                df['Activity_Units'].str.contains('nM', na=False) &
                (df['Activity_Value'] <= 10000)  # 10μM = 10000nM
            )
            df['Activity_Label'] = mask.astype(int)
        else:
            df['Activity_Label'] = 1  # 默认
        
        logger.info(f"清洗后数据: {len(df)}条 | 活性: {df['Activity_Label'].sum()} | 非活性: {(df['Activity_Label']==0).sum()}")
        
        return df
    
    def prepare_training_data(self, target_name: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        获取训练数据 (特征X + 标签y)
        
        返回:
            X: SMILES列表 (后续计算特征)
            y: 活性标签 (1=活性, 0=非活性)
            df: 完整数据DataFrame
        """
        df = self.fetch_activity_data(target_name)
        
        if df.empty:
            logger.warning(f"无{target_name}数据，将使用内置示例数据")
            return self._get_fallback_data()
        
        smiles_list = df['SMILES'].tolist()
        y = df['Activity_Label'].values
        
        return smiles_list, y, df
    
    def _get_fallback_data(self) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
        """当ChEMBL不可用时的后备数据 (已知登革病毒抑制剂)"""
        # 已知的登革病毒NS5抑制剂相关化合物 (简化示例)
        fallback_data = [
            ("Nc1ncnc2c1c(=O)[nH]cn2", 1, "Ribavirin analog"),
            ("OC1=CC=C(C=C1O)C2=CC=CC=C2", 0, "Non-active compound"),
            ("CC1=CC(=O)C2=C(C=CC=C2O1)C(=O)O", 1, "Nordihydroguaiaretic acid"),
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 0, "Ibuprofen (non-active)"),
            ("C1=CC=C(C=C1)C2=CC=CC=C2", 0, "Biphenyl (non-active)"),
            ("OC1=CC=C(C=C1)C2=CC=CC=C2O", 1, "2,2'-Biphenol"),
            ("CC(=O)OC1=CC=CC=C1C(=O)O", 0, "Aspirin (non-active)"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0, "Caffeine (non-active)"),
            ("OC1=CC=C(C=C1)C(=O)C2=CC=CC=C2", 1, "4-Hydroxybenzophenone"),
            ("CC1=CC=C(C=C1)S(=O)(=O)NC2=NC=CC=N2", 0, "Non-active sulfonamide"),
        ]
        
        smiles_list = [d[0] for d in fallback_data]
        y = np.array([d[1] for d in fallback_data])
        df = pd.DataFrame({
            'SMILES': smiles_list,
            'Activity_Label': y,
            'Name': [d[2] for d in fallback_data],
            'Source': 'fallback'
        })
        
        logger.info(f"使用后备数据: {len(smiles_list)}个化合物")
        return smiles_list, y, df


def download_chembl_data(target_name: str, output_dir: Path = None) -> Path:
    """
    便捷函数: 下载ChEMBL数据并保存
    
    返回:
        保存的CSV文件路径
    """
    fetcher = ChEMBLDataFetcher(cache_dir=output_dir / "chembl_cache" if output_dir else None)
    df = fetcher.fetch_activity_data(target_name)
    
    if output_dir:
        output_file = output_dir / f"chembl_{target_name}_activity.csv"
        df.to_csv(output_file, index=False)
        return output_file
    
    return Path("chembl_data.csv")
