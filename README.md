# 抗登革病毒药物虚拟筛选平台

## 项目概述

本项目是一个基于机器学习的抗登革病毒药物虚拟筛选平台，旨在利用网络公开的化合物库对潜在的抗登革病毒药物进行高效筛选。该平台集成了分子数据获取、特征工程、模型训练、虚拟筛选和结果分析等功能模块，为药物发现研究提供完整的解决方案。

### 核心功能

- **数据获取**：从ChEMBL数据库获取抗登革病毒活性数据
- **特征工程**：计算分子描述符（RDKit）和指纹（Morgan、MACCS）
- **模型训练**：支持多种分类模型（随机森林、XGBoost、SVM）
- **虚拟筛选**：高效筛选大型化合物库
- **结果分析**：生成论文可用的可视化图表

## 项目结构

```
dengue_drug_discovery/
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的数据
│   └── external/             # 外部数据
├── src/                      # 源代码目录
│   ├── data_acquisition/     # 数据获取模块
│   ├── feature_engineering/  # 特征工程模块
│   ├── modeling/             # 模型训练模块
│   ├── virtual_screening/    # 虚拟筛选模块
│   ├── analysis/             # 结果分析模块
│   └── config.py             # 项目配置文件
├── results/                  # 结果目录
│   ├── models/               # 训练好的模型
│   ├── figures/              # 可视化图表
│   └── reports/              # 分析报告
├── scripts/                  # 脚本目录
├── docs/                     # 文档目录
├── requirements.txt          # 依赖包
└── README.md                 # 项目说明文档
```

## 安装

### 环境要求

- Python 3.8+
- RDKit 2021.09+
- ChemBL WebResource Client 0.10+
- 其他依赖包详见requirements.txt

### 安装步骤

1. 克隆项目到本地
```bash
git clone <repository_url>
cd dengue_drug_discovery
```

2. 创建虚拟环境
```bash
conda create -n dengue_drug python=3.8
conda activate dengue_drug
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

## 配置

项目配置文件位于 `src/config.py`，可以根据需要修改以下配置参数：

- **目录配置**：项目根目录、数据目录、结果目录等
- **数据配置**：活性类型、单位、阈值等
- **特征工程配置**：要计算的特征类型、参数等
- **模型配置**：模型类型、参数、评估指标等
- **虚拟筛选配置**：批处理大小、并行处理等

## 使用方法

### 1. 数据获取

从ChEMBL数据库获取抗登革病毒活性数据：

```python
from src.data_acquisition.fetch_chembl_data import fetch_dengue_data

# 获取IC50值，单位为nM
df = fetch_dengue_data()
print(f"数据保存到: {os.path.join(DATA_DIR['raw'], DATA_CONFIG['chembl_data_file'])}")
```

从PubChem数据库获取化合物数据：

```python
from src.data_acquisition.fetch_pubchem_data import fetch_pubchem_compounds, fetch_pubchem_substance_list

# 下载特定CID范围的化合物
cid_list = list(range(1, 1001))
df = fetch_pubchem_compounds(cid_list, identifier_type='cid')

# 按物质名称搜索并下载化合物
df = fetch_pubchem_substance_list("Dengue inhibitor", max_compounds=500)

# 使用交互式脚本下载
# python scripts/download_pubchem.py
```

### 2. 特征工程

计算分子描述符和指纹：

```python
from src.feature_engineering.molecular_features import calculate_features

# 计算Morgan指纹、MACCS指纹和RDKit描述符
df_features = calculate_features(df, features_to_calculate=['morgan', 'maccs', 'rdkit_desc'])
```

### 3. 模型训练

训练机器学习模型：

```python
from src.modeling.model_training import train_classification_model

# 训练随机森林和XGBoost模型
models, results, df_features = train_classification_model(df_features, 
                                                         models_to_train=['random_forest', 'xgboost'],
                                                         balance_data=True)
```

### 4. 虚拟筛选

筛选化合物库：

```python
from src.virtual_screening.virtual_screening import screen_compound_library

# 筛选化合物库
library_path = 'data/raw/example_library.smi'
screening_results = screen_compound_library(library_path, models, batch_size=100)
```

### 5. 结果分析

生成分析报告：

```python
from src.analysis.result_analysis import generate_analysis_report

# 生成分析报告
generate_analysis_report()
```

## 示例工作流程

运行测试脚本可以体验完整的工作流程：

```bash
python scripts/run_test.py
```

运行PubChem化合物下载脚本：

```bash
python scripts/download_pubchem.py
```

测试脚本会执行以下步骤：
1. 获取抗登革病毒活性数据
2. 计算分子特征
3. 训练机器学习模型
4. 筛选示例化合物库
5. 生成分析报告

## 结果

### 模型性能指标

- **分类模型**：ROC-AUC、PR-AUC、准确率、精确率、召回率、F1分数
- **回归模型**：RMSE、MAE、R²

### 可视化结果

- ROC曲线和PR曲线
- 模型性能比较图
- 混淆矩阵
- 特征重要性图
- 活性分布直方图
- 分子性质散点图

## 自定义

### 修改目标病毒

在 `src/data_acquisition/fetch_chembl_data.py` 中修改以下参数：

```python
# 修改目标为其他病毒
target_name = 'Zika virus'
```

### 添加新的特征

在 `src/feature_engineering/molecular_features.py` 中添加新的特征计算函数：

```python
def calculate_new_feature(mol):
    # 实现新特征计算逻辑
    return feature_value
```

### 使用自定义模型

在 `src/modeling/model_training.py` 中添加新的模型：

```python
from sklearn.ensemble import AdaBoostClassifier

# 添加AdaBoost分类器
model_configs['adaboost'] = {
    'class': AdaBoostClassifier,
    'params': {
        'n_estimators': 100,
        'random_state': 42
    }
}
```

## 数据来源

- **ChEMBL数据库**：提供抗登革病毒活性数据
- **PubChem**：可扩展的数据来源
- **ZINC数据库**：用于虚拟筛选的化合物库

## 学术考虑

### 可重复性

- 设置随机种子确保实验可重复
- 保存模型参数和特征集
- 使用标准化的数据预处理流程

### 模型评估

- 使用交叉验证避免过拟合
- 评估多种性能指标
- 进行Y-scrambling测试

### 适用范围

- 计算模型的适用范围
- 分子相似性阈值
- 活性预测的置信区间

## 常见问题

### 安装RDKit失败

```bash
# 使用conda安装RDKit
conda install -c conda-forge rdkit
```

### 数据获取超时

```python
# 增加超时时间
from chembl_webresource_client.settings import Settings
Settings.Instance().TIMEOUT = 60  # 设置为60秒
```

### 内存不足

```python
# 减少特征维度
features_to_calculate = ['morgan']  # 只计算Morgan指纹
```

## 引用

如果您使用本项目的代码或结果发表论文，请引用：

```
@article{dengue_drug_discovery,
  title={基于机器学习的抗登革病毒药物虚拟筛选平台},
  author={您的名字},
  journal={药物发现杂志},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱：your_email@example.com
- GitHub：https://github.com/your_username/dengue_drug_discovery

## 版本历史

- v1.0.0 (2024-01-01)：初始版本
- v1.1.0 (2024-03-23)：优化版本
  - 改进数据预处理流程，包括异常值移除、pIC50计算和活性标签化
  - 增强特征工程，添加PADEL指纹计算和特征选择功能
  - 优化模型训练和评估，添加LazyPredict支持和多种模型比较
  - 改进数据可视化，添加更多图表来分析数据分布和模型性能
  - 支持使用现有的化合物数据文件进行筛选

## 优化说明

本项目基于 `chEMBL_Alzheimer_Glycogen_synthase_kinase_3_beta.ipynb` 中的QSAR最佳实践进行了优化，包括：

1. **数据预处理**：
   - 改进了pIC50计算，添加了IC50值限制，避免负的pIC50值
   - 增强了异常值处理，移除不合理的IC50值
   - 添加了活性分类功能，将化合物分为active、intermediate和inactive三个类别
   - 优化了SMILES字符串处理，获取最长分子组件

2. **特征工程**：
   - 改进了PADEL描述符计算，添加了索引列处理和更详细的参数设置
   - 添加了特征选择功能，支持多种特征选择方法（方差阈值、相关性分析、特征重要性、递归特征消除）
   - 优化了特征预处理流程，包括常数特征移除和高相关特征移除

3. **模型训练**：
   - 改进了LazyPredict评估功能，添加了NaN值处理，增加了前15个模型的性能展示
   - 保存了前10个最佳模型的详细信息
   - 优化了模型评估流程，添加了更多性能指标

4. **数据可视化**：
   - 在数据获取模块中添加了IC50 vs pIC50散点图和活性化合物的pIC50分布图
   - 在模型训练模块中添加了混淆矩阵热图、ROC曲线、精确率-召回率曲线和预测概率分布图

## 致谢

感谢ChEMBL数据库提供的药物活性数据，以及RDKit开发团队提供的分子计算工具。
