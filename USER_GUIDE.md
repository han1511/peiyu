# 抗登革病毒药物虚拟筛选平台 - 使用指南

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [完整使用流程](#完整使用流程)
- [各模块详细说明](#各模块详细说明)
- [结果解释](#结果解释)
- [常见问题](#常见问题)

---

## 项目概述

本项目是一个**基于机器学习的抗登革病毒药物虚拟筛选平台**，用于：

✅ 从公共数据库获取抗登革病毒活性数据  
✅ 计算分子特征（指纹和理化性质）  
✅ 训练机器学习模型  
✅ 大规模虚拟筛选化合物库  
✅ 预测潜在的抗登革病毒活性化合物  
✅ 生成详细的分析报告和可视化结果

### 技术特点

- **数据来源**: ChEMBL、PubChem 等公共数据库
- **分子特征**: Morgan 指纹、MACCS 指纹、RDKit 描述符
- **机器学习模型**: 随机森林、XGBoost、支持向量机
- **类不平衡处理**: SMOTE、ADASYN 过采样技术
- **模型评估**: 交叉验证、多种性能指标
- **可视化**: 混淆矩阵、ROC 曲线、PR 曲线、特征重要性

---

## 快速开始

### 1. 环境安装

```bash
# 创建 conda 环境
conda create -n dengue_drug python=3.8
conda activate dengue_drug

# 进入项目目录
cd e:\Python\dengue_drug_discovery

# 安装依赖
pip install -r requirements.txt
```

### 2. 一键运行完整流程

```bash
# 运行完整研究流程
python research_pipeline.py
```

这将自动执行：
1. 数据获取 → 2. 特征工程 → 3. 模型训练 → 4. 虚拟筛选 → 5. 生成报告

### 3. 查看详细分析

```bash
# 运行详细分析模块
python src/analysis/detailed_analysis.py
```

---

## 完整使用流程

### 步骤 1: 数据获取

从 ChEMBL 数据库获取抗登革病毒活性数据：

```python
from src.data_acquisition.fetch_chembl_data import fetch_dengue_data

# 获取数据
df = fetch_dengue_data()

# 数据将保存到：data/raw/dengue_antiviral_data.csv
```

**说明**：
- 数据来源：ChEMBL 数据库中的登革病毒相关活性数据
- 活性阈值：pIC50 ≥ 6.0 (IC50 ≤ 1000 nM) 判定为活性化合物
- 包含信息：SMILES、活性值、目标蛋白、化合物 ID 等

### 步骤 2: 特征工程

计算分子特征：

```python
from src.feature_engineering.molecular_features import calculate_features

# 计算特征
df_features = calculate_features(
    input_data='data/raw/dengue_antiviral_data.csv',
    features_to_calculate=['Morgan', 'MACCS', 'rdkit_desc']
)

# 特征数据保存到：data/processed/processed_dengue_data.csv
```

**特征类型**：
- **Morgan 指纹** (1024 位): 基于分子子结构的圆形指纹
- **MACCS 指纹** (167 位): 预定义的化学特征密钥
- **RDKit 描述符**: 分子量、LogP、氢键供体/受体数等

**预处理**：
- 移除常数特征
- 移除高相关特征（相关系数 > 0.95）
- 处理缺失值

### 步骤 3: 模型训练

训练机器学习模型：

```python
from src.modeling.model_training import train_classification_model

# 训练模型
models, results, df_features = train_classification_model(
    input_df=df_features,              # 特征数据
    models_to_train=['RandomForest', 'XGBoost', 'SVM'],  # 要训练的模型
    balance_data=True,                 # 处理类不平衡
    cross_val=True                     # 进行交叉验证
)
```

**模型说明**：

| 模型 | 优点 | 适用场景 |
|------|------|----------|
| **随机森林** | 抗过拟合、可解释性强 | 通用场景，首选模型 |
| **XGBoost** | 预测精度高、处理不平衡数据 | 数据量较大时 |
| **SVM** | 适合高维特征、小样本 | 特征维度高时 |

**评估指标**：
- **ROC-AUC**: 区分活性/非活性的能力（0.5-1.0，越大越好）
- **PR-AUC**: 在不平衡数据上的性能（更适合本场景）
- **准确率**: 预测正确的比例
- **精确率**: 预测为活性的化合物中真正活性的比例
- **召回率**: 真正活性的化合物被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均

### 步骤 4: 虚拟筛选

使用训练好的模型筛选化合物库：

```python
from src.virtual_screening.virtual_screening import screen_compound_library
import joblib

# 加载训练好的模型
models = {}
for model_name in ['RandomForest', 'XGBoost', 'SVM']:
    model = joblib.load(f'results/models/{model_name}_model.pkl')
    scaler = joblib.load(f'results/models/{model_name}_scaler.pkl')
    models[model_name] = (model, scaler)

# 筛选化合物库
results = screen_compound_library(
    library_path='data/raw/compound_library.smi',  # 化合物库路径
    models=models,                                  # 模型字典
    format='smi',                                   # 文件格式
    batch_size=1000                                 # 批处理大小
)
```

**支持的文件格式**：
- **SMI 格式**: 每行一个 SMILES 字符串
- **CSV 格式**: 包含 SMILES 列的 CSV 文件

**输出结果**：
- 每个化合物的预测结果（活性/非活性）
- 预测概率（0-1 之间）
- 多模型投票结果

### 步骤 5: 结果分析

生成详细分析报告：

```python
# 方法 1: 使用研究流程自动生成报告
python research_pipeline.py

# 方法 2: 单独运行分析模块
python src/analysis/detailed_analysis.py

# 方法 3: 在 Python 中调用
from src.analysis.detailed_analysis import generate_detailed_report

generate_detailed_report(
    data_path='data/processed/processed_dengue_data.csv',
    model_results_path='results/models/',
    models_path='results/models/',
    save_dir='results/reports/my_analysis'
)
```

**报告内容**：
- 数据分布分析图表
- 模型性能对比图表
- 特征重要性分析
- 详细文本报告（Markdown 格式）

---

## 各模块详细说明

### 1. 研究流程主脚本 (`research_pipeline.py`)

**功能**: 一键运行完整的药物发现流程

**使用方法**：

```python
from research_pipeline import DengueDrugDiscoveryPipeline

# 创建流程实例
pipeline = DengueDrugDiscoveryPipeline()

# 运行完整流程
pipeline.run_full_pipeline(
    models_to_train=['RandomForest', 'XGBoost', 'SVM'],
    library_path='path/to/compound_library.smi'  # 可选
)

# 或分步运行
pipeline.run_data_acquisition(source='chembl')
pipeline.run_feature_engineering()
pipeline.run_model_training(balance_data=True, cross_val=True)
pipeline.run_virtual_screening(library_path='path/to/library.smi')
pipeline.generate_report()
```

**输出**：
- 各阶段数据文件（CSV 格式）
- 训练好的模型（PKL 格式）
- 筛选结果（CSV 格式）
- 研究报告（TXT 格式）

### 2. 详细分析模块 (`src/analysis/detailed_analysis.py`)

**功能**: 深入分析数据和模型结果

**分析方法**：
- 数据分布分析
- 模型性能对比
- 特征重要性分析
- 可视化图表生成

**使用方法**：

```bash
python src/analysis/detailed_analysis.py
```

**输出**：
- 数据分布图表
- 模型性能图表
- 特征重要性图表
- 详细分析报告（Markdown）

### 3. 结果分析模块 (`src/analysis/result_analysis.py`)

**功能**: 生成模型评估图表和 HTML 报告

**可视化类型**：
- 混淆矩阵
- ROC 曲线
- Precision-Recall 曲线
- 模型性能对比图
- 筛选结果分布图

**使用方法**：

```python
from src.analysis.result_analysis import generate_analysis_report

models = ['RandomForest', 'XGBoost', 'SVM']
generate_analysis_report(
    models_list=models,
    screening_results_path='results/models/virtual_screening/screening_results.csv'
)
```

---

## 结果解释

### 1. 数据获取结果

**关键指标**：
- **总化合物数**: 获取的化合物总数
- **活性化合物数**: pIC50 ≥ 6.0 的化合物数
- **非活性化合物数**: pIC50 < 6.0 的化合物数
- **活性/非活性比例**: 反映数据不平衡程度

**解释示例**：
```
总化合物数：1500
活性化合物数：300 (20.0%)
非活性化合物数：1200 (80.0%)
活性/非活性比例：25.0%
```
说明：数据存在类不平衡问题，需要使用 SMOTE 等技术处理。

### 2. 特征工程结果

**关键指标**：
- **特征总数**: 用于训练的分子特征数量
- **Morgan 指纹特征数**: 1024 位
- **MACCS 指纹特征数**: 167 位
- **RDKit 描述符数**: 约 200 个理化性质

**解释示例**：
```
总样本数：1450
特征总数：1391
Morgan 指纹特征数：1024
MACCS 指纹特征数：167
RDKit 描述符数：200
```
说明：高维特征空间，适合机器学习模型训练。

### 3. 模型训练结果

**性能指标解读**：

| 指标 | 优秀 | 良好 | 一般 | 说明 |
|------|------|------|------|------|
| **ROC-AUC** | >0.9 | 0.8-0.9 | 0.7-0.8 | 区分能力 |
| **PR-AUC** | >0.8 | 0.6-0.8 | 0.4-0.6 | 不平衡数据性能 |
| **准确率** | >0.85 | 0.75-0.85 | 0.65-0.75 | 整体预测准确度 |
| **精确率** | >0.8 | 0.6-0.8 | 0.4-0.6 | 预测活性的可靠性 |
| **召回率** | >0.8 | 0.6-0.8 | 0.4-0.6 | 发现活性化合物的能力 |
| **F1 分数** | >0.8 | 0.6-0.8 | 0.4-0.6 | 综合指标 |

**解释示例**：
```
RandomForest 模型性能:
  ROC-AUC: 0.9234  ← 优秀的区分能力
  PR-AUC: 0.8567   ← 在不平衡数据上表现良好
  准确率：0.8750   ← 87.5% 的预测是正确的
  精确率：0.8200   ← 预测为活性的化合物中 82% 真正活性
  召回率：0.7800   ← 78% 的真正活性化合物被找到
  F1 分数：0.7995  ← 精确率和召回率的平衡
```

**模型选择建议**：
- **ROC-AUC 最高**: 整体区分能力最强
- **PR-AUC 最高**: 在不平衡数据上表现最好
- **F1 分数最高**: 精确率和召回率平衡最好
- **召回率最高**: 最适合虚拟筛选（宁可错杀，不可放过）

### 4. 虚拟筛选结果

**关键指标**：
- **总筛选化合物数**: 筛选的化合物总数
- **预测活性化合物数**: 被预测为活性的化合物数
- **命中率**: 预测活性化合物的比例
- **平均预测概率**: 所有化合物的平均预测概率

**解释示例**：
```
总筛选化合物数：10000
预测活性化合物数：850
命中率：8.50%
平均预测概率：0.3245 ± 0.2156
```
说明：从 10000 个化合物中筛选出 850 个潜在活性化合物（8.5%）。

**Top 化合物分析**：
```
Top 10 预测活性化合物:
ID: CMPD_3521      概率：0.9876
SMILES: CC(=O)Oc1ccccc1C(=O)O...

ID: CMPD_7845      概率：0.9654
SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)...
```
说明：这些化合物最有可能是抗登革病毒活性化合物，建议优先进行实验验证。

### 5. 特征重要性分析

**重要特征类型**：
- **Morgan 子结构**: 特定的分子子结构模式
- **理化性质**: 分子量、LogP、极性表面积等
- **药效团特征**: 氢键供体/受体、芳香环等

**解释示例**：
```
Top 重要特征:
1. MolWt (分子量): 0.0523
2. MolLogP (脂水分配系数): 0.0487
3. NumHDonors (氢键供体数): 0.0456
4. Morgan_125 (特定子结构): 0.0432
```
说明：分子量、脂溶性和氢键相互作用对活性影响最大。

---

## 常见问题

### Q1: 安装依赖时出错

**问题**: `pip install -r requirements.txt` 失败

**解决方案**：
```bash
# 升级 pip
python -m pip install --upgrade pip

# 逐个安装依赖
pip install numpy pandas scikit-learn rdkit
pip install xgboost imbalanced-learn
pip install matplotlib seaborn plotly
```

### Q2: 数据获取失败

**问题**: 无法从 ChEMBL 数据库获取数据

**解决方案**：
1. 检查网络连接
2. 使用代理（如果需要）
3. 手动下载数据并放到 `data/raw/` 目录

### Q3: 模型训练时间过长

**问题**: 模型训练耗时超过 1 小时

**解决方案**：
- 减少训练样本数（测试模式）
- 减少特征数量
- 降低模型复杂度（减少树的数量）
- 使用更强大的 CPU 或 GPU

### Q4: 类不平衡问题

**问题**: 模型总是预测为非活性

**解决方案**：
```python
# 确保启用类不平衡处理
train_classification_model(
    balance_data=True,  # 使用 SMOTE 过采样
    models_to_train=['RandomForest', 'XGBoost']
)
```

### Q5: 虚拟筛选结果为空

**问题**: 筛选结果为空或命中率极低

**解决方案**：
1. 检查化合物库文件格式是否正确
2. 检查 SMILES 字符串是否有效
3. 降低预测阈值（默认 0.5）
4. 使用更多模型进行投票

### Q6: 如何保存和加载模型

**保存模型**：
```python
import joblib

# 保存模型
joblib.dump(model, 'my_model.pkl')
joblib.dump(scaler, 'my_scaler.pkl')
```

**加载模型**：
```python
import joblib

# 加载模型
model = joblib.load('my_model.pkl')
scaler = joblib.load('my_scaler.pkl')
```

### Q7: 如何自定义模型参数

**修改配置文件** (`src/config.py`)：
```python
MODEL_CONFIG = {
    'models': {
        'RandomForest': {
            'n_estimators': 500,  # 树的数量
            'max_depth': 20,      # 最大深度
            'min_samples_split': 5,
            'random_state': 42
        },
        # ... 其他模型
    }
}
```

### Q8: 如何分析自己的化合物库

**步骤**：
1. 准备化合物库文件（SMI 或 CSV 格式）
2. 确保包含 SMILES 列
3. 调用虚拟筛选函数：

```python
from src.virtual_screening.virtual_screening import screen_compound_library

# 加载训练好的模型
models = {...}  # 见步骤 4

# 筛选
results = screen_compound_library(
    library_path='my_compounds.csv',
    models=models,
    format='csv',
    smiles_column='canonical_smiles'  # SMILES 列名
)
```

---

## 技术支持

如遇到问题，请检查：
1. Python 版本是否为 3.8+
2. 所有依赖是否正确安装
3. 文件路径是否正确
4. 数据格式是否符合要求

---

## 参考文献

1. ChEMBL 数据库：https://www.ebi.ac.uk/chembl/
2. PubChem 数据库：https://pubchem.ncbi.nlm.nih.gov/
3. RDKit 文档：https://www.rdkit.org/docs/
4. scikit-learn: https://scikit-learn.org/
5. XGBoost: https://xgboost.readthedocs.io/

---

**祝研究顺利！** 🎉
