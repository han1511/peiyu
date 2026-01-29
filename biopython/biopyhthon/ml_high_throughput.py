import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, mean_squared_error, r2_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
import os

class HighThroughputML:
    def __init__(self):
        """初始化高通量数据机器学习预测类"""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_selector = None
        self.model = None
        self.task_type = None  # 'classification' or 'regression'
    
    def load_data(self, file_path, target_column, sep='\t'):
        """
        加载高通量数据
        
        Parameters:
        file_path: str, 数据文件路径
        target_column: str, 目标变量列名
        sep: str, 数据分隔符，默认为制表符
        
        Returns:
        None
        """
        try:
            self.data = pd.read_csv(file_path, sep=sep)
            print(f"数据加载成功，共 {self.data.shape[0]} 样本，{self.data.shape[1]} 特征")
            
            # 分离特征和目标变量
            self.X = self.data.drop(columns=[target_column])
            self.y = self.data[target_column]
            
            # 自动判断任务类型
            if len(self.y.unique()) <= 10:  # 假设类别数小于等于10为分类任务
                self.task_type = 'classification'
                print(f"自动识别为分类任务，共 {len(self.y.unique())} 个类别")
            else:
                self.task_type = 'regression'
                print("自动识别为回归任务")
                
        except Exception as e:
            print(f"数据加载失败: {e}")
    
    def preprocess_data(self, scale_method='standard', select_features=False, k=10):
        """
        预处理数据：标准化和特征选择
        
        Parameters:
        scale_method: str, 标准化方法 ('standard' 或 'minmax')
        select_features: bool, 是否进行特征选择
        k: int, 选择的特征数量
        
        Returns:
        None
        """
        try:
            # 数据标准化
            if scale_method == 'standard':
                self.scaler = StandardScaler()
            elif scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("scale_method 必须是 'standard' 或 'minmax'")
            
            X_scaled = self.scaler.fit_transform(self.X)
            
            # 特征选择
            if select_features:
                if self.task_type == 'classification':
                    self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                else:
                    self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
                
                X_selected = self.feature_selector.fit_transform(X_scaled, self.y)
                print(f"特征选择完成，保留 {k} 个特征")
                return X_selected
            else:
                return X_scaled
                
        except Exception as e:
            print(f"数据预处理失败: {e}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集
        
        Parameters:
        test_size: float, 测试集比例
        random_state: int, 随机种子
        
        Returns:
        None
        """
        try:
            processed_X = self.preprocess_data()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                processed_X, self.y, test_size=test_size, random_state=random_state
            )
            print(f"数据划分完成，训练集: {self.X_train.shape[0]} 样本，测试集: {self.X_test.shape[0]} 样本")
            
        except Exception as e:
            print(f"数据划分失败: {e}")
    
    def train_model(self, model_type='random_forest', param_grid=None):
        """
        训练机器学习模型
        
        Parameters:
        model_type: str, 模型类型 ('random_forest' 或 'svm')
        param_grid: dict, 网格搜索参数
        
        Returns:
        None
        """
        try:
            # 确保数据已划分
            if self.X_train is None:
                self.split_data()
            
            # 选择模型
            if model_type == 'random_forest':
                if self.task_type == 'classification':
                    base_model = RandomForestClassifier()
                else:
                    base_model = RandomForestRegressor()
            elif model_type == 'svm':
                if self.task_type == 'classification':
                    base_model = SVC(probability=True)
                else:
                    base_model = SVR()
            else:
                raise ValueError("model_type 必须是 'random_forest' 或 'svm'")
            
            # 网格搜索调参
            if param_grid:
                print("正在进行网格搜索调参...")
                grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy' if self.task_type == 'classification' else 'r2')
                grid_search.fit(self.X_train, self.y_train)
                self.model = grid_search.best_estimator_
                print(f"最佳参数: {grid_search.best_params_}")
                print(f"交叉验证最佳得分: {grid_search.best_score_:.4f}")
            else:
                # 使用默认参数训练
                self.model = base_model
                self.model.fit(self.X_train, self.y_train)
                print("模型训练完成")
                
        except Exception as e:
            print(f"模型训练失败: {e}")
    
    def evaluate_model(self):
        """
        评估模型性能
        
        Returns:
        dict, 评估指标结果
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            y_pred = self.model.predict(self.X_test)
            results = {}
            
            if self.task_type == 'classification':
                # 分类任务评估指标
                y_prob = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
                
                results['accuracy'] = accuracy_score(self.y_test, y_pred)
                results['precision'] = precision_score(self.y_test, y_pred, average='weighted')
                results['recall'] = recall_score(self.y_test, y_pred, average='weighted')
                results['f1'] = f1_score(self.y_test, y_pred, average='weighted')
                
                if y_prob is not None and len(self.y.unique()) == 2:  # 二分类计算AUC
                    results['auc'] = roc_auc_score(self.y_test, y_prob)
                
                # 混淆矩阵
                cm = confusion_matrix(self.y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.y.unique(), yticklabels=self.y.unique())
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                plt.savefig('confusion_matrix.png')
                print("混淆矩阵已保存为 confusion_matrix.png")
                
            else:
                # 回归任务评估指标
                results['mse'] = mean_squared_error(self.y_test, y_pred)
                results['rmse'] = np.sqrt(results['mse'])
                results['r2'] = r2_score(self.y_test, y_pred)
                
                # 真实值 vs 预测值散点图
                plt.figure(figsize=(8, 6))
                plt.scatter(self.y_test, y_pred, alpha=0.5)
                plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title('True vs Predicted Values')
                plt.savefig('true_vs_predicted.png')
                print("真实值 vs 预测值图已保存为 true_vs_predicted.png")
            
            # 打印评估结果
            print("\n模型评估结果:")
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
                
            return results
            
        except Exception as e:
            print(f"模型评估失败: {e}")
            return None
    
    def predict(self, new_data):
        """
        使用训练好的模型进行预测
        
        Parameters:
        new_data: pandas.DataFrame or numpy.array, 新数据
        
        Returns:
        numpy.array, 预测结果
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            # 预处理新数据
            if isinstance(new_data, pd.DataFrame):
                new_data = new_data.values
            
            new_data_scaled = self.scaler.transform(new_data)
            if self.feature_selector:
                new_data_scaled = self.feature_selector.transform(new_data_scaled)
            
            predictions = self.model.predict(new_data_scaled)
            return predictions
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def save_model(self, file_path='model.pkl'):
        """
        保存训练好的模型
        
        Parameters:
        file_path: str, 模型保存路径
        
        Returns:
        None
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            # 保存模型和相关组件
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'task_type': self.task_type
            }
            
            joblib.dump(model_data, file_path)
            print(f"模型已保存到 {file_path}")
            
        except Exception as e:
            print(f"模型保存失败: {e}")
    
    def load_model(self, file_path='model.pkl'):
        """
        加载已保存的模型
        
        Parameters:
        file_path: str, 模型文件路径
        
        Returns:
        None
        """
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data.get('feature_selector')
            self.task_type = model_data['task_type']
            print(f"模型已从 {file_path} 加载")
            
        except Exception as e:
            print(f"模型加载失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高通量数据机器学习预测工具")
    
    # 添加命令行参数
    parser.add_argument('input_file', help='输入数据文件路径')
    parser.add_argument('target_column', help='目标变量列名')
    parser.add_argument('--model-type', default='random_forest', choices=['random_forest', 'svm'], help='模型类型')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--save-model', default='model.pkl', help='模型保存路径')
    parser.add_argument('--sep', default='\t', help='数据分隔符')
    
    args = parser.parse_args()
    
    # 创建并使用ML对象
    ml = HighThroughputML()
    ml.load_data(args.input_file, args.target_column, sep=args.sep)
    ml.split_data(test_size=args.test_size)
    ml.train_model(model_type=args.model_type)
    ml.evaluate_model()
    ml.save_model(args.save_model)

if __name__ == "__main__":
    main()