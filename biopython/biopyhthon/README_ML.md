# 高通量数据机器学习预测工具

## 程序功能

这是一个用于高通量数据（如基因组学、蛋白质组学等）的机器学习预测工具，支持：

- 自动识别分类和回归任务
- 数据预处理（标准化、特征选择）
- 多种机器学习模型（随机森林、支持向量机）
- 网格搜索参数调优
- 模型评估和可视化
- 新数据预测

## 依赖安装

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 使用方法

### 命令行使用

```bash
# 使用默认参数运行
python ml_high_throughput.py input_data.txt target_column

# 自定义参数
python ml_high_throughput.py input_data.txt target_column --model-type random_forest --test-size 0.2 --save-model my_model.pkl
```

### 参数说明

- `input_data.txt`: 输入数据文件路径（制表符分隔）
- `target_column`: 目标变量列名
- `--model-type`: 模型类型（random_forest 或 svm）
- `--test-size`: 测试集比例（0.0-1.0）
- `--save-model`: 模型保存路径

### Python脚本使用

```python
from ml_high_throughput import HighThroughputML

# 创建ML对象
ml = HighThroughputML()

# 加载数据
ml.load_data("example_data.txt", target_column="class", sep="\t")

# 预处理数据
ml.preprocess_data(scale_method="standard", select_features=False)

# 划分数据
ml.split_data(test_size=0.3, random_state=42)

# 训练模型
ml.train_model(model_type="random_forest")

# 评估模型
ml.evaluate_model()

# 预测新数据
predictions = ml.predict(new_data)

# 保存模型
ml.save_model("model.pkl")
```

## 数据格式要求

输入数据应为制表符分隔的文本文件，包含特征列和目标变量列：

```
sample_id	feature1	feature2	...	featureN	target
sample1	0.1	0.2	...	0.9	class1
sample2	0.3	0.4	...	0.7	class2
...
```

## 示例数据

程序包含两个示例数据文件：

- `example_classification_data.txt`: 分类任务示例数据
- `example_regression_data.txt`: 回归任务示例数据

## 测试程序

运行测试脚本以验证程序功能：

```bash
python test_ml_program.py
```

## 输出文件

程序运行后会生成：

- 模型文件（默认：model.pkl）
- 混淆矩阵（分类任务）：confusion_matrix.png
- 真实值vs预测值图（回归任务）：true_vs_predicted.png

## 注意事项

1. 程序会自动检测任务类型（分类或回归）
2. 建议先查看示例数据了解格式要求
3. 对于大型数据集，可通过特征选择减少计算量