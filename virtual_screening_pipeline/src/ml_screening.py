#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习初筛模块

功能：
1. 多种机器学习模型训练（XGBoost, RandomForest, SVM, LogisticRegression）
2. 模型交叉验证和性能评估
3. 模型集成和投票
4. 化合物活性预测和排序
5. 特征重要性分析

作者：研究团队
版本：1.0.0
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict,
    train_test_split, learning_curve, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available")

from configs.config import (
    PROJECT_ROOT, RESULTS_DIR, LOG_CONFIG,
    ML_MODELS, MODEL_PERFORMANCE_THRESHOLDS,
    CROSS_VALIDATION_CONFIG, DATA_SPLIT_CONFIG
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    机器学习模型训练类

    支持多种分类模型的训练和评估
    """

    def __init__(self, model_name: str, model_config: Optional[Dict] = None):
        """
        初始化模型训练器

        参数:
            model_name: 模型名称
            model_config: 模型配置字典
        """
        self.model_name = model_name
        self.model_config = model_config or ML_MODELS.get(model_name, {})
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None

    def _create_model(self):
        """根据模型名称创建模型实例"""
        if self.model_name == "XGBoost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost is not available")
            return xgb.XGBClassifier(**self.model_config)

        elif self.model_name == "RandomForest":
            return RandomForestClassifier(**self.model_config)

        elif self.model_name == "SVM":
            return SVC(**self.model_config)

        elif self.model_name == "LogisticRegression":
            return LogisticRegression(**self.model_config)

        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        训练模型

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        返回:
            dict: 训练结果
        """
        try:
            logger.info(f"Training {self.model_name} model...")

            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model = self._create_model()

            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_val_scaled, y_val)]

            if self.model_name == "XGBoost" and eval_set is not None:
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            elif self.model_name == "SVM" or self.model_name == "LogisticRegression":
                self.model.fit(X_train_scaled, y_train)
            else:
                self.model.fit(X_train_scaled, y_train)

            self.is_trained = True

            train_results = {"model": self.model_name, "trained": True}

            if eval_set is not None:
                y_pred = self.model.predict(X_val_scaled)
                y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]

                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                train_results["validation_metrics"] = metrics

            if hasattr(self.model, "feature_importances_"):
                self.feature_importance = self.model.feature_importances_

            logger.info(f"{self.model_name} training completed")

            return train_results

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            return {"model": self.model_name, "trained": False, "error": str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        参数:
            X: 特征矩阵

        返回:
            np.ndarray: 预测类别
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        参数:
            X: 特征矩阵

        return: np.ndarray: 预测概率矩阵
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            logger.warning(f"Model {self.model_name} does not support predict_proba")
            return None

    def cross_validate(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     cv: int = 5) -> Dict[str, Any]:
        """
        交叉验证

        参数:
            X: 特征矩阵
            y: 标签数组
            cv: 交叉验证折数

        返回:
            dict: 交叉验证结果
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        try:
            X_scaled = self.scaler.transform(X)

            skf = StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=CROSS_VALIDATION_CONFIG["random_state"]
            )

            cv_results = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "auc": []
            }

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_val_fold = y[val_idx]

                model = self._create_model()
                model.fit(X_train_fold, y_train_fold)

                y_pred = model.predict(X_val_fold)
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

                cv_results["accuracy"].append(accuracy_score(y_val_fold, y_pred))
                cv_results["precision"].append(precision_score(y_val_fold, y_pred, zero_division=0))
                cv_results["recall"].append(recall_score(y_val_fold, y_pred, zero_division=0))
                cv_results["f1"].append(f1_score(y_val_fold, y_pred, zero_division=0))
                cv_results["auc"].append(roc_auc_score(y_val_fold, y_pred_proba))

            summary = {}
            for metric, values in cv_results.items():
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": [float(v) for v in values]
                }

            logger.info(f"Cross-validation completed for {self.model_name}")
            return summary

        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {}

    def _calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        计算分类指标

        参数:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率

        返回:
            dict: 指标字典
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_pred_proba)),
            "auprc": float(average_precision_score(y_true, y_pred_proba))
        }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        return metrics

    def get_feature_importance(self,
                              feature_names: Optional[List[str]] = None,
                              top_k: Optional[int] = None) -> Dict[str, float]:
        """
        获取特征重要性

        参数:
            feature_names: 特征名称列表
            top_k: 返回前k个最重要的特征

        return: dict: 特征重要性字典
        """
        if self.feature_importance is None:
            return {}

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]

        importance_dict = dict(zip(feature_names, self.feature_importance))

        if top_k is not None:
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )

        return importance_dict


class EnsembleClassifier:
    """
    集成分类器类

    通过投票机制集成多个模型
    """

    def __init__(self, models: List[Tuple[str, Dict]], voting: str = "soft"):
        """
        初始化集成分类器

        参数:
            models: 模型列表，每个元素为(模型名称, 配置字典)的元组
            voting: 投票方式 ('soft' 或 'hard')
        """
        self.models = models
        self.voting = voting
        self.classifiers = []
        self.scalers = []
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        训练集成模型

        参数:
            X_train: 训练特征
            y_train: 训练标签

        返回:
            dict: 训练结果
        """
        try:
            logger.info(f"Training ensemble with {len(self.models)} models...")

            self.classifiers = []
            self.scalers = []

            for model_name, model_config in self.models:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                trainer = ModelTrainer(model_name, model_config)
                trainer.scaler = scaler

                result = trainer.train(X_train, y_train)

                if result.get("trained", False):
                    self.classifiers.append((model_name, trainer.model, scaler))
                    self.scalers.append(scaler)

            self.is_trained = True

            logger.info(f"Ensemble training completed with {len(self.classifiers)} models")

            return {
                "success": True,
                "num_models": len(self.classifiers),
                "models": [name for name, _, _ in self.classifiers]
            }

        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            return {"success": False, "error": str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别（投票）

        参数:
            X: 特征矩阵

        返回:
            np.ndarray: 预测类别
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained yet")

        predictions = []
        probas = []

        for model_name, model, scaler in self.classifiers:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions.append(pred)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 1]
                probas.append(proba)

        if self.voting == "soft" and probas:
            avg_proba = np.mean(probas, axis=0)
            return (avg_proba >= 0.5).astype(int)
        else:
            predictions_array = np.array(predictions)
            from scipy.stats import mode
            return mode(predictions_array, axis=0, keepdims=False)[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（平均概率）

        参数:
            X: 特征矩阵

        返回:
            np.ndarray: 预测概率矩阵
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained yet")

        probas = []

        for model_name, model, scaler in self.classifiers:
            X_scaled = scaler.transform(X)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 1]
                probas.append(proba)

        if probas:
            return np.mean(probas, axis=0)
        else:
            return None


class VirtualScreening:
    """
    虚拟筛选类

    使用机器学习模型对化合物库进行筛选
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        初始化虚拟筛选器

        参数:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir or RESULTS_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models = {}
        self.ensemble_model = None

        logger.info("Initialized VirtualScreening")

    def train_models(self,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练多个机器学习模型

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            model_names: 要训练的模型名称列表

        返回:
            dict: 训练结果
        """
        if model_names is None:
            model_names = list(ML_MODELS.keys())

        results = {
            "models": {},
            "best_model": None,
            "best_auc": 0.0
        }

        for model_name in model_names:
            if model_name not in ML_MODELS:
                logger.warning(f"Model {model_name} not in configuration, skipping")
                continue

            try:
                trainer = ModelTrainer(model_name, ML_MODELS[model_name])
                train_result = trainer.train(X_train, y_train, X_val, y_val)

                if train_result.get("trained", False):
                    self.trained_models[model_name] = trainer

                    if "validation_metrics" in train_result:
                        val_auc = train_result["validation_metrics"]["auc"]
                        results["models"][model_name] = {
                            "trained": True,
                            "validation_auc": val_auc
                        }

                        if val_auc > results["best_auc"]:
                            results["best_auc"] = val_auc
                            results["best_model"] = model_name

                    model_path = self.model_dir / f"{model_name}_model.pkl"
                    self._save_model(trainer, model_path)

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results["models"][model_name] = {"trained": False, "error": str(e)}

        logger.info(f"Training completed. Best model: {results['best_model']} (AUC: {results['best_auc']:.4f})")

        return results

    def train_ensemble(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练集成模型

        参数:
            X_train: 训练特征
            y_train: 训练标签
            model_names: 模型名称列表

        返回:
            dict: 训练结果
        """
        if model_names is None:
            model_names = list(ML_MODELS.keys())

        models_to_ensemble = [
            (name, ML_MODELS[name]) for name in model_names if name in ML_MODELS
        ]

        if not models_to_ensemble:
            return {"success": False, "error": "No valid models to ensemble"}

        self.ensemble_model = EnsembleClassifier(models_to_ensemble)
        result = self.ensemble_model.fit(X_train, y_train)

        if result.get("success", False):
            ensemble_path = self.model_dir / "ensemble_model.pkl"
            self._save_model(self.ensemble_model, ensemble_path)

        return result

    def screen_compounds(self,
                        X_compounds: np.ndarray,
                        model_name: Optional[str] = None,
                        use_ensemble: bool = False,
                        probability_threshold: float = 0.5) -> Dict[str, Any]:
        """
        筛选化合物

        参数:
            X_compounds: 化合物特征矩阵
            model_name: 使用的模型名称（如果不用集成）
            use_ensemble: 是否使用集成模型
            probability_threshold: 概率阈值

        返回:
            dict: 筛选结果
        """
        try:
            if use_ensemble and self.ensemble_model is not None:
                predictions = self.ensemble_model.predict(X_compounds)
                probabilities = self.ensemble_model.predict_proba(X_compounds)
                model_used = "Ensemble"
            elif model_name and model_name in self.trained_models:
                trainer = self.trained_models[model_name]
                predictions = trainer.predict(X_compounds)
                probabilities = trainer.predict_proba(X_compounds)
                model_used = model_name
            else:
                logger.error("No trained model available")
                return {"success": False, "error": "No trained model"}

            results = {
                "success": True,
                "model_used": model_used,
                "total_compounds": len(predictions),
                "predicted_active": int(np.sum(predictions == 1)),
                "predicted_inactive": int(np.sum(predictions == 0)),
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "probability_threshold": probability_threshold
            }

            if probabilities is not None:
                results["mean_probability"] = float(np.mean(probabilities))
                results["std_probability"] = float(np.std(probabilities))
                results["max_probability"] = float(np.max(probabilities))
                results["min_probability"] = float(np.min(probabilities))

            logger.info(f"Screening completed: {results['predicted_active']}/{results['total_compounds']} predicted as active")

            return results

        except Exception as e:
            logger.error(f"Error screening compounds: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_top_compounds(self,
                         X_compounds: np.ndarray,
                         probabilities: np.ndarray,
                         top_n: int = 100,
                         threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取排名最高的化合物

        参数:
            X_compounds: 化合物特征矩阵
            probabilities: 活性概率
            top_n: 返回前n个化合物
            threshold: 最小概率阈值

        返回:
            tuple: (排序后的特征, 排序后的概率)
        """
        if threshold is not None:
            mask = probabilities >= threshold
            X_filtered = X_compounds[mask]
            prob_filtered = probabilities[mask]
        else:
            X_filtered = X_compounds
            prob_filtered = probabilities

        sorted_indices = np.argsort(prob_filtered)[::-1]

        top_indices = sorted_indices[:top_n]

        return X_filtered[top_indices], prob_filtered[top_indices]

    def evaluate_models(self,
                       X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, Any]:
        """
        评估所有训练好的模型

        参数:
            X_test: 测试特征
            y_test: 测试标签

        返回:
            dict: 评估结果
        """
        results = {
            "models": {},
            "best_model": None,
            "best_auc": 0.0
        }

        for model_name, trainer in self.trained_models.items():
            try:
                y_pred = trainer.predict(X_test)
                y_pred_proba = trainer.predict_proba(X_test)

                metrics = trainer._calculate_metrics(y_test, y_pred, y_pred_proba)
                results["models"][model_name] = metrics

                if metrics["auc"] > results["best_auc"]:
                    results["best_auc"] = metrics["auc"]
                    results["best_model"] = model_name

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results["models"][model_name] = {"error": str(e)}

        if self.ensemble_model is not None:
            try:
                y_pred = self.ensemble_model.predict(X_test)
                y_pred_proba = self.ensemble_model.predict_proba(X_test)

                from sklearn.metrics import roc_auc_score, average_precision_score
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "auc": float(roc_auc_score(y_test, y_pred_proba)),
                    "auprc": float(average_precision_score(y_test, y_pred_proba))
                }

                results["models"]["Ensemble"] = metrics

                if metrics["auc"] > results["best_auc"]:
                    results["best_auc"] = metrics["auc"]
                    results["best_model"] = "Ensemble"

            except Exception as e:
                logger.error(f"Error evaluating Ensemble: {str(e)}")

        logger.info(f"Model evaluation completed. Best model: {results['best_model']}")

        return results

    def _save_model(self, model: Any, model_path: Path) -> bool:
        """保存模型到文件"""
        try:
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def _load_model(self, model_path: Path) -> Any:
        """从文件加载模型"""
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


def run_virtual_screening_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_screening: np.ndarray,
    model_names: Optional[List[str]] = None,
    use_ensemble: bool = True,
    top_n: int = 1000
) -> Dict[str, Any]:
    """
    运行完整的虚拟筛选流程

    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        X_screening: 要筛选的化合物特征
        model_names: 模型名称列表
        use_ensemble: 是否使用集成模型
        top_n: 筛选前n个化合物

    返回:
        dict: 完整流程结果
    """
    results = {
        "training": {},
        "evaluation": {},
        "screening": {},
        "success": False
    }

    try:
        screener = VirtualScreening()

        logger.info("Step 1: Training models...")
        train_results = screener.train_models(X_train, y_train, X_test, y_test, model_names)
        results["training"] = train_results

        if use_ensemble:
            logger.info("Step 2: Training ensemble model...")
            ensemble_results = screener.train_ensemble(X_train, y_train, model_names)
            results["training"]["ensemble"] = ensemble_results

        logger.info("Step 3: Evaluating models...")
        eval_results = screener.evaluate_models(X_test, y_test)
        results["evaluation"] = eval_results

        logger.info("Step 4: Screening compounds...")
        if use_ensemble:
            screen_results = screener.screen_compounds(X_screening, use_ensemble=True)
        else:
            best_model = train_results.get("best_model")
            screen_results = screener.screen_compounds(X_screening, model_name=best_model)
        results["screening"] = screen_results

        results["success"] = True

        logger.info("Virtual screening pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in virtual screening pipeline: {str(e)}")
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    logger.info("Testing ML Screening module")

    np.random.seed(42)
    X_dummy = np.random.rand(100, 50)
    y_dummy = np.random.randint(0, 2, 100)

    screener = VirtualScreening()
    logger.info(f"Created VirtualScreening instance")
