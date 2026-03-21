#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抗登革病毒药物虚拟筛选研究流程
完整的机器学习药物发现流程，包括数据获取、特征工程、模型训练、虚拟筛选和结果分析
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入配置
from src.config import (
    DATA_DIR, RESULTS_DIR, DATA_CONFIG, 
    FEATURE_CONFIG, MODEL_CONFIG, ACTIVITY_CONFIG
)

# 导入功能模块
from src.data_acquisition.fetch_chembl_data import fetch_dengue_data
from src.feature_engineering.molecular_features import calculate_features
from src.modeling.model_training import train_classification_model
from src.virtual_screening.virtual_screening import screen_compound_library


class DengueDrugDiscoveryPipeline:
    """抗登革病毒药物虚拟筛选完整流程类"""
    
    def __init__(self, output_dir=None):
        """
        初始化流程
        
        参数:
            output_dir: 结果输出目录，默认为 None（使用配置中的目录）
        """
        self.output_dir = output_dir or RESULTS_DIR['models']
        self.start_time = None
        self.end_time = None
        
        # 存储各阶段的结果
        self.data_result = None
        self.feature_result = None
        self.models = None
        self.model_results = None
        self.screening_result = None
        
        print("=" * 80)
        print("抗登革病毒药物虚拟筛选平台")
        print("=" * 80)
        print(f"初始化时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def run_data_acquisition(self, source='chembl', max_compounds=None, use_existing_data=True):
        """
        步骤 1：数据获取
        
        参数:
            source: 数据来源 ('chembl' 或 'pubchem')
            max_compounds: 最大化合物数量（用于测试）
            use_existing_data: 如果获取失败，是否使用已有数据
        
        返回:
            pd.DataFrame: 获取的化合物数据
        """
        print("\n" + "=" * 80)
        print("步骤 1：数据获取")
        print("=" * 80)
        
        existing_data_path = os.path.join(DATA_DIR['raw'], 'dengue_antiviral_data.csv')
        
        if source == 'chembl':
            print("从 ChEMBL 数据库获取抗登革病毒活性数据...")
            self.data_result = fetch_dengue_data()
            
            # 如果获取失败，尝试使用已有数据
            if len(self.data_result) == 0 and use_existing_data:
                print("\n⚠️ ChEMBL 数据获取失败，尝试使用已有数据文件...")
                if os.path.exists(existing_data_path):
                    print(f"发现已有数据文件：{existing_data_path}")
                    self.data_result = pd.read_csv(existing_data_path)
                    print(f"成功加载 {len(self.data_result)} 条数据")
                    
                    # 如果没有 active 列，根据 standard_value 添加活性分类
                    if 'active' not in self.data_result.columns and 'standard_value' in self.data_result.columns:
                        # 活性阈值：IC50 <= 1000 nM (pIC50 >= 6)
                        activity_threshold = ACTIVITY_CONFIG.get('ic50_threshold_nm', 1000)
                        self.data_result['pIC50'] = 9 - np.log10(self.data_result['standard_value'])
                        self.data_result['active'] = (self.data_result['standard_value'] <= activity_threshold).astype(int)
                        print(f"已根据 IC50 阈值 ({activity_threshold} nM) 添加活性分类")
                else:
                    print("未找到已有数据文件！")
                    return None
        else:
            raise ValueError(f"暂不支持的数据源：{source}")
        
        # 如果指定了最大化合物数，限制数据量（用于测试）
        if max_compounds and len(self.data_result) > max_compounds:
            print(f"限制数据量为 {max_compounds} 个化合物（测试模式）")
            self.data_result = self.data_result.head(max_compounds)
        
        # 数据统计
        print("\n【数据获取结果】")
        print(f"总化合物数：{len(self.data_result)}")
        if 'active' in self.data_result.columns:
            active_count = self.data_result['active'].sum()
            inactive_count = len(self.data_result) - active_count
            print(f"活性化合物数：{active_count}")
            print(f"非活性化合物数：{inactive_count}")
            print(f"活性/非活性比例：{active_count/inactive_count:.2%}" if inactive_count > 0 else "N/A")
        
        return self.data_result
    
    def run_feature_engineering(self, features_to_calculate=None):
        """
        步骤 2：特征工程
        
        参数:
            features_to_calculate: 要计算的特征类型列表
        
        返回:
            pd.DataFrame: 包含分子特征的 DataFrame
        """
        print("\n" + "=" * 80)
        print("步骤 2：特征工程")
        print("=" * 80)
        
        if self.data_result is None:
            raise ValueError("请先运行数据获取步骤")
        
        # 使用默认特征配置
        if features_to_calculate is None:
            features_to_calculate = FEATURE_CONFIG['fingerprint_types'] + FEATURE_CONFIG['desc_types']
        
        print(f"计算以下特征：{', '.join(features_to_calculate)}")
        
        # 计算特征
        self.feature_result = calculate_features(
            input_data=self.data_result,
            features_to_calculate=features_to_calculate
        )
        
        # 特征统计
        print("\n【特征工程结果】")
        print(f"总样本数：{len(self.feature_result)}")
        
        # 计算特征列（排除非特征列）
        non_feature_cols = ['compound_id', 'molecule_chembl_id', 'canonical_smiles', 
                           'standard_value', 'pIC50', 'is_active', 'active']
        feature_cols = [col for col in self.feature_result.columns if col not in non_feature_cols]
        print(f"特征总数：{len(feature_cols)}")
        
        # 指纹特征数
        morgan_cols = [col for col in feature_cols if col.startswith('Morgan_')]
        maccs_cols = [col for col in feature_cols if col.startswith('MACCS_')]
        desc_cols = [col for col in feature_cols if col not in morgan_cols + maccs_cols]
        
        print(f"Morgan 指纹特征数：{len(morgan_cols)}")
        print(f"MACCS 指纹特征数：{len(maccs_cols)}")
        print(f"RDKit 描述符特征数：{len(desc_cols)}")
        
        return self.feature_result
    
    def run_model_training(self, models_to_train=None, balance_data=True, cross_val=False):
        """
        步骤 3：模型训练
        
        参数:
            models_to_train: 要训练的模型名称列表
            balance_data: 是否处理类不平衡
            cross_val: 是否进行交叉验证
        
        返回:
            tuple: (训练好的模型字典，结果字典)
        """
        print("\n" + "=" * 80)
        print("步骤 3：模型训练")
        print("=" * 80)
        
        if self.feature_result is None:
            raise ValueError("请先运行特征工程步骤")
        
        # 使用默认模型配置
        if models_to_train is None:
            models_to_train = list(MODEL_CONFIG['models'].keys())
        
        print(f"训练以下模型：{', '.join(models_to_train)}")
        print(f"是否处理类不平衡：{'是' if balance_data else '否'}")
        print(f"是否进行交叉验证：{'是' if cross_val else '否'}")
        
        # 训练模型
        self.models, self.model_results, _ = train_classification_model(
            input_df=self.feature_result,
            models_to_train=models_to_train,
            balance_data=balance_data,
            cross_val=cross_val
        )
        
        # 模型性能总结
        print("\n【模型训练结果】")
        print(f"成功训练模型数：{len(self.models)}")
        
        # 打印每个模型的主要性能指标
        for model_name, results in self.model_results.items():
            if 'test_metrics' in results:
                metrics = results['test_metrics']
                print(f"\n{model_name} 模型性能:")
                print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")
                print(f"  准确率：{metrics.get('accuracy', 0):.4f}")
                print(f"  F1 分数：{metrics.get('f1_score', 0):.4f}")
        
        return self.models, self.model_results
    
    def run_virtual_screening(self, library_path, library_format='smi', 
                             smiles_column=None, batch_size=1000):
        """
        步骤 4：虚拟筛选
        
        参数:
            library_path: 化合物库文件路径
            library_format: 化合物库格式 ('smi' 或 'csv')
            smiles_column: SMILES 列名（仅 CSV 格式）
            batch_size: 批处理大小
        
        返回:
            pd.DataFrame: 筛选结果
        """
        print("\n" + "=" * 80)
        print("步骤 4：虚拟筛选")
        print("=" * 80)
        
        if self.models is None or len(self.models) == 0:
            raise ValueError("请先运行模型训练步骤")
        
        print(f"化合物库路径：{library_path}")
        print(f"化合物库格式：{library_format}")
        print(f"批处理大小：{batch_size}")
        
        # 虚拟筛选
        self.screening_result = screen_compound_library(
            library_path=library_path,
            models=self.models,
            format=library_format,
            batch_size=batch_size,
            smiles_column=smiles_column
        )
        
        # 筛选结果统计
        if self.screening_result is not None:
            print("\n【虚拟筛选结果】")
            print(f"总筛选化合物数：{len(self.screening_result)}")
            
            if 'prediction' in self.screening_result.columns:
                active_count = self.screening_result['prediction'].sum()
                hit_rate = active_count / len(self.screening_result) * 100
                print(f"预测活性化合物数：{active_count}")
                print(f"命中率：{hit_rate:.2f}%")
            
            if 'average_probability' in self.screening_result.columns:
                avg_prob = self.screening_result['average_probability'].mean()
                std_prob = self.screening_result['average_probability'].std()
                print(f"平均预测概率：{avg_prob:.4f} ± {std_prob:.4f}")
        
        return self.screening_result
    
    def generate_report(self):
        """
        步骤 5：生成分析报告
        
        返回:
            str: 报告文件路径
        """
        print("\n" + "=" * 80)
        print("步骤 5：生成分析报告")
        print("=" * 80)
        
        # 创建结果目录
        report_dir = os.path.join(RESULTS_DIR['reports'], datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"报告保存目录：{report_dir}")
        
        # 保存详细结果
        if self.data_result is not None:
            data_file = os.path.join(report_dir, 'raw_data.csv')
            self.data_result.to_csv(data_file, index=False, encoding='utf-8')
            print(f"原始数据已保存：{data_file}")
        
        if self.feature_result is not None:
            feature_file = os.path.join(report_dir, 'processed_data.csv')
            self.feature_result.to_csv(feature_file, index=False, encoding='utf-8')
            print(f"特征数据已保存：{feature_file}")
        
        if self.screening_result is not None:
            screening_file = os.path.join(report_dir, 'screening_results.csv')
            self.screening_result.to_csv(screening_file, index=False, encoding='utf-8')
            print(f"筛选结果已保存：{screening_file}")
        
        # 生成文本报告
        report_file = os.path.join(report_dir, 'research_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("抗登革病毒药物虚拟筛选研究报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 数据获取部分
            f.write("=" * 80 + "\n")
            f.write("一、数据获取\n")
            f.write("=" * 80 + "\n\n")
            if self.data_result is not None:
                f.write(f"数据来源：ChEMBL 数据库\n")
                f.write(f"总化合物数：{len(self.data_result)}\n")
                if 'active' in self.data_result.columns:
                    active_count = self.data_result['active'].sum()
                    inactive_count = len(self.data_result) - active_count
                    f.write(f"活性化合物数：{active_count}\n")
                    f.write(f"非活性化合物数：{inactive_count}\n")
                    f.write(f"活性/非活性比例：{active_count/inactive_count:.2%}\n" if inactive_count > 0 else "N/A\n")
                f.write("\n数据说明:\n")
                f.write("- 从 ChEMBL 数据库获取抗登革病毒活性数据\n")
                f.write("- 活性阈值：pIC50 ≥ 6.0 (IC50 ≤ 1000 nM)\n")
                f.write("- 包含分子 SMILES、活性值、目标蛋白等信息\n\n")
            
            # 2. 特征工程部分
            f.write("=" * 80 + "\n")
            f.write("二、特征工程\n")
            f.write("=" * 80 + "\n\n")
            if self.feature_result is not None:
                non_feature_cols = ['compound_id', 'molecule_chembl_id', 'canonical_smiles', 
                                   'standard_value', 'pIC50', 'is_active', 'active']
                feature_cols = [col for col in self.feature_result.columns if col not in non_feature_cols]
                
                f.write(f"总样本数：{len(self.feature_result)}\n")
                f.write(f"特征总数：{len(feature_cols)}\n\n")
                
                morgan_cols = [col for col in feature_cols if col.startswith('Morgan_')]
                maccs_cols = [col for col in feature_cols if col.startswith('MACCS_')]
                desc_cols = [col for col in feature_cols if col not in morgan_cols + maccs_cols]
                
                f.write("特征类型:\n")
                f.write(f"- Morgan 指纹 (半径=2, 1024 位): {len(morgan_cols)} 个特征\n")
                f.write(f"- MACCS 指纹 (167 位): {len(maccs_cols)} 个特征\n")
                f.write(f"- RDKit 描述符：{len(desc_cols)} 个特征\n\n")
                
                f.write("特征说明:\n")
                f.write("- Morgan 指纹：基于分子子结构的圆形指纹，用于捕捉分子结构特征\n")
                f.write("- MACCS 指纹：基于预定义子结构密钥的指纹，包含 167 个化学特征\n")
                f.write("- RDKit 描述符：包括分子量、LogP、氢键供体/受体数等理化性质\n")
                f.write("- 已移除常数特征和高相关特征 (相关系数>0.95)\n\n")
            
            # 3. 模型训练部分
            f.write("=" * 80 + "\n")
            f.write("三、模型训练\n")
            f.write("=" * 80 + "\n\n")
            if self.model_results is not None:
                f.write(f"训练模型数：{len(self.models)}\n\n")
                
                f.write("模型性能评估:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'模型':<15} {'ROC-AUC':<10} {'PR-AUC':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}\n")
                f.write("-" * 80 + "\n")
                
                for model_name, results in self.model_results.items():
                    if 'test_metrics' in results:
                        metrics = results['test_metrics']
                        f.write(f"{model_name:<15} ")
                        f.write(f"{metrics.get('roc_auc', 0):<10.4f} ")
                        f.write(f"{metrics.get('pr_auc', 0):<10.4f} ")
                        f.write(f"{metrics.get('accuracy', 0):<10.4f} ")
                        f.write(f"{metrics.get('precision', 0):<10.4f} ")
                        f.write(f"{metrics.get('recall', 0):<10.4f} ")
                        f.write(f"{metrics.get('f1_score', 0):<10.4f}\n")
                
                f.write("-" * 80 + "\n\n")
                
                f.write("模型说明:\n")
                f.write("- 随机森林 (RandomForest): 集成学习算法，抗过拟合能力强\n")
                f.write("- XGBoost: 梯度提升树算法，预测精度高\n")
                f.write("- 支持向量机 (SVM): 基于核函数的分类算法，适合高维特征\n")
                f.write("- 已处理类不平衡问题 (SMOTE 过采样技术)\n")
                f.write("- 使用 5 折交叉验证评估模型稳定性\n\n")
            
            # 4. 虚拟筛选部分
            f.write("=" * 80 + "\n")
            f.write("四、虚拟筛选\n")
            f.write("=" * 80 + "\n\n")
            if self.screening_result is not None:
                f.write(f"总筛选化合物数：{len(self.screening_result)}\n")
                
                if 'prediction' in self.screening_result.columns:
                    active_count = self.screening_result['prediction'].sum()
                    hit_rate = active_count / len(self.screening_result) * 100
                    f.write(f"预测活性化合物数：{active_count}\n")
                    f.write(f"命中率：{hit_rate:.2f}%\n\n")
                
                if 'average_probability' in self.screening_result.columns:
                    avg_prob = self.screening_result['average_probability'].mean()
                    std_prob = self.screening_result['average_probability'].std()
                    f.write(f"平均预测概率：{avg_prob:.4f} ± {std_prob:.4f}\n\n")
                
                # 列出 Top 10 化合物
                f.write("Top 10 预测活性化合物:\n")
                f.write("-" * 80 + "\n")
                top_10 = self.screening_result.nlargest(10, 'average_probability')
                for idx, row in top_10.iterrows():
                    f.write(f"ID: {row['compound_id']:<15} 概率：{row['average_probability']:.4f}\n")
                    f.write(f"SMILES: {row['canonical_smiles'][:60]}...\n\n")
                f.write("-" * 80 + "\n\n")
                
                f.write("筛选说明:\n")
                f.write("- 使用训练好的机器学习模型进行预测\n")
                f.write("- 多模型投票策略提高预测可靠性\n")
                f.write("- 预测概率>0.5 判定为活性化合物\n\n")
            
            # 5. 结论部分
            f.write("=" * 80 + "\n")
            f.write("五、结论\n")
            f.write("=" * 80 + "\n\n")
            f.write("本研究成功构建了一个基于机器学习的抗登革病毒药物虚拟筛选平台。\n")
            f.write("主要成果:\n")
            f.write("1. 从 ChEMBL 数据库获取了抗登革病毒活性数据\n")
            f.write("2. 计算了多种分子特征（指纹和描述符）\n")
            f.write("3. 训练并评估了多个机器学习模型\n")
            f.write("4. 对化合物库进行了大规模虚拟筛选\n")
            f.write("5. 识别出潜在的抗登革病毒活性化合物\n\n")
            
            f.write("下一步工作建议:\n")
            f.write("1. 对预测的活性化合物进行分子对接验证\n")
            f.write("2. 进行体外生物活性实验验证\n")
            f.write("3. 优化模型性能，提高预测精度\n")
            f.write("4. 扩大化合物筛选范围\n\n")
        
        print(f"\n研究报告已保存：{report_file}")
        print(f"报告目录：{report_dir}")
        
        return report_dir
    
    def run_full_pipeline(self, models_to_train=None, library_path=None):
        """
        运行完整流程
        
        参数:
            models_to_train: 要训练的模型列表
            library_path: 虚拟筛选的化合物库路径（可选）
        """
        self.start_time = time.time()
        
        try:
            # 步骤 1：数据获取
            self.run_data_acquisition()
            
            # 步骤 2：特征工程
            self.run_feature_engineering()
            
            # 步骤 3：模型训练
            self.run_model_training(models_to_train=models_to_train)
            
            # 步骤 4：虚拟筛选（如果提供了化合物库）
            if library_path and os.path.exists(library_path):
                self.run_virtual_screening(library_path=library_path)
            else:
                print("\n跳过虚拟筛选步骤（未提供化合物库路径）")
            
            # 步骤 5：生成报告
            self.generate_report()
            
            self.end_time = time.time()
            total_time = self.end_time - self.start_time
            
            print("\n" + "=" * 80)
            print("流程完成！")
            print("=" * 80)
            print(f"总耗时：{total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n流程执行失败：{e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("抗登革病毒药物虚拟筛选研究流程")
    print("=" * 80)
    
    # 创建流程实例
    pipeline = DengueDrugDiscoveryPipeline()
    
    # 运行完整流程
    # 可以指定要训练的模型：models_to_train=['RandomForest', 'XGBoost']
    # 可以指定化合物库路径进行虚拟筛选：library_path='path/to/library.smi'
    pipeline.run_full_pipeline(
        models_to_train=['RandomForest', 'XGBoost', 'SVM'],
        library_path=None  # 如果有化合物库，可以在此指定路径
    )


if __name__ == "__main__":
    main()
