#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型持久化与断点续传模块

功能:
1. 模型保存/加载 (.pkl格式)
2. 流程检查点 (每步结果落盘)
3. 断点恢复 (从失败步骤继续)
4. 统计验证 (Y-scrambling, bootstrap置信区间)
"""

import os
import json
import pickle
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# 模型持久化
# ============================================================================

class ModelPersistence:
    """模型保存与加载"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model, model_name: str, 
                   metadata: Dict = None) -> Path:
        """
        保存模型到pkl文件
        
        参数:
            model: 训练好的模型对象
            model_name: 模型名称
            metadata: 元数据 (训练参数、性能指标等)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = self.model_dir / filename
        
        save_data = {
            'model': model,
            'model_name': model_name,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat(),
            'sklearn_version': self._get_sklearn_version(),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 同时保存一个"最新"指针
        latest_path = self.model_dir / f"{model_name}_latest.pkl"
        shutil_copy = filepath
        with open(latest_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"模型已保存: {filepath}")
        return filepath
    
    def load_model(self, model_name: str, 
                   use_latest: bool = True) -> Tuple[Any, Dict]:
        """
        加载模型
        
        参数:
            model_name: 模型名称
            use_latest: 是否加载最新版本
            
        返回:
            (model, metadata)
        """
        if use_latest:
            filepath = self.model_dir / f"{model_name}_latest.pkl"
            if not filepath.exists():
                # 查找最新文件
                files = sorted(self.model_dir.glob(f"{model_name}_*.pkl"))
                files = [f for f in files if 'latest' not in f.name]
                if files:
                    filepath = files[-1]
                else:
                    raise FileNotFoundError(f"未找到模型: {model_name}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        logger.info(f"模型已加载: {filepath}")
        return save_data['model'], save_data.get('metadata', {})
    
    def list_models(self) -> List[Dict]:
        """列出所有已保存的模型"""
        models = []
        for f in self.model_dir.glob("*.pkl"):
            if 'latest' in f.name:
                continue
            try:
                with open(f, 'rb') as fh:
                    data = pickle.load(fh)
                models.append({
                    'file': str(f),
                    'name': data.get('model_name', f.stem),
                    'saved_at': data.get('saved_at', ''),
                    'metadata': data.get('metadata', {}),
                })
            except:
                models.append({'file': str(f), 'name': f.stem, 'saved_at': '', 'metadata': {}})
        return models
    
    def _get_sklearn_version(self):
        try:
            import sklearn
            return sklearn.__version__
        except:
            return 'unknown'


# ============================================================================
# 断点续传
# ============================================================================

class CheckpointManager:
    """流程检查点管理器"""
    
    # 步骤定义
    STEPS = [
        "target_preparation",
        "compound_loading",
        "compound_preprocessing",
        "feature_computation",
        "model_training",
        "virtual_screening",
        "molecular_docking",
        "admet_evaluation",
        "result_analysis"
    ]
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "pipeline_state.json"
    
    def save_checkpoint(self, step_idx: int, step_name: str,
                        data: Dict = None, extra: Dict = None):
        """
        保存检查点
        
        参数:
            step_idx: 步骤索引
            step_name: 步骤名称
            data: 要保存的数据 (会序列化为pkl)
            extra: 额外元数据 (会保存到json)
        """
        # 保存数据
        if data is not None:
            data_file = self.checkpoint_dir / f"step_{step_idx}_{step_name}.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 更新状态
        state = self.load_state()
        state['steps'][step_name] = {
            'index': step_idx,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data_file) if data else None,
        }
        state['last_completed_step'] = step_idx
        state['last_step_name'] = step_name
        state['extra'] = {**state.get('extra', {}), **(extra or {})}
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查点已保存: Step {step_idx+1} - {step_name}")
    
    def load_state(self) -> Dict:
        """加载流程状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'steps': {}, 'last_completed_step': -1, 'extra': {}}
    
    def load_checkpoint_data(self, step_name: str) -> Any:
        """加载某个步骤的数据"""
        state = self.load_state()
        step_info = state.get('steps', {}).get(step_name)
        if step_info and step_info.get('data_file'):
            data_file = Path(step_info['data_file'])
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def get_resume_point(self) -> Tuple[int, str]:
        """
        获取恢复点
        
        返回:
            (next_step_idx, next_step_name) - 下一个要执行的步骤
        """
        state = self.load_state()
        last_step = state.get('last_completed_step', -1)
        next_step = last_step + 1
        
        if next_step >= len(self.STEPS):
            return -1, 'completed'  # 所有步骤已完成
        
        return next_step, self.STEPS[next_step]
    
    def can_resume(self) -> bool:
        """是否可以恢复"""
        state = self.load_state()
        return state.get('last_completed_step', -1) >= 0 and \
               state.get('last_completed_step', -1) < len(self.STEPS) - 1
    
    def clear(self):
        """清除所有检查点"""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        logger.info("检查点已清除")


# ============================================================================
# 统计验证
# ============================================================================

class StatisticalValidator:
    """
    统计验证模块
    
    1. Y-scrambling: 打乱标签验证模型非随机性
    2. Bootstrap置信区间: 评估指标稳定性
    3. 配对t检验: 比较模型显著性
    """
    
    def __init__(self, n_scramble: int = 20, n_bootstrap: int = 1000,
                 confidence_level: float = 0.95, random_state: int = 42):
        self.n_scramble = n_scramble
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
    
    def y_scrambling(self, model, X_train, y_train, X_test, y_test,
                     metric_func=None) -> Dict[str, Any]:
        """
        Y-scrambling验证
        
        打乱训练标签重新训练模型，如果真实模型的性能显著高于打乱模型，
        说明模型学到了真实的构效关系而非偶然相关性
        
        返回:
            dict: {
                'true_score': float,       # 真实模型得分
                'scrambled_scores': list,   # 打乱模型得分列表
                'scrambled_mean': float,    # 打乱模型平均分
                'scrambled_std': float,     # 打乱模型标准差
                'p_value': float,           # p值
                'is_valid': bool,           # 是否通过验证
            }
        """
        from sklearn.metrics import roc_auc_score
        from scipy import stats
        
        if metric_func is None:
            metric_func = lambda yt, yp: roc_auc_score(yt, yp)
        
        rng = np.random.RandomState(self.random_state)
        
        # 真实模型得分
        y_proba_true = model.predict_proba(X_test)[:, 1]
        true_score = metric_func(y_test, y_proba_true)
        
        # 打乱标签训练
        scrambled_scores = []
        for i in range(self.n_scramble):
            y_shuffled = rng.permutation(y_train)
            
            # 克隆模型
            from sklearn.base import clone
            scrambled_model = clone(model)
            
            try:
                scrambled_model.fit(X_train, y_shuffled)
                y_proba_scrambled = scrambled_model.predict_proba(X_test)[:, 1]
                score = metric_func(y_test, y_proba_scrambled)
                scrambled_scores.append(score)
            except Exception as e:
                logger.warning(f"Y-scramble第{i+1}次失败: {e}")
                continue
        
        if not scrambled_scores:
            return {'true_score': true_score, 'scrambled_scores': [],
                    'is_valid': False, 'p_value': 1.0}
        
        scrambled_mean = np.mean(scrambled_scores)
        scrambled_std = np.std(scrambled_scores)
        
        # 单样本t检验: 真实得分是否显著高于打乱得分
        if scrambled_std > 0:
            t_stat, p_value = stats.ttest_1samp(scrambled_scores, true_score)
            # 单侧检验: true_score > scrambled_mean
            p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        else:
            p_value = 0.0 if true_score > scrambled_mean else 1.0
        
        # 验证标准: 真实得分 > 打乱均值 + 2*标准差
        threshold = scrambled_mean + 2 * scrambled_std
        is_valid = true_score > threshold
        
        return {
            'true_score': float(true_score),
            'scrambled_scores': [float(s) for s in scrambled_scores],
            'scrambled_mean': float(scrambled_mean),
            'scrambled_std': float(scrambled_std),
            'threshold': float(threshold),
            'p_value': float(p_value),
            'is_valid': bool(is_valid),
            'n_scramble': len(scrambled_scores),
        }
    
    def bootstrap_confidence_interval(self, y_true, y_pred, y_proba=None,
                                       metrics=None) -> Dict[str, Tuple[float, float, float]]:
        """
        Bootstrap置信区间
        
        返回:
            dict: {metric_name: (mean, lower_ci, upper_ci)}
        """
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, f1_score,
            precision_score, recall_score
        )
        
        if metrics is None:
            metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_proba is not None:
            y_proba = np.array(y_proba)
        
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)
        
        boot_results = {m: [] for m in metrics}
        
        for _ in range(self.n_bootstrap):
            # 有放回采样
            indices = rng.randint(0, n, size=n)
            yt = y_true[indices]
            yp = y_pred[indices]
            
            # 确保有正负样本
            if len(np.unique(yt)) < 2:
                continue
            
            if 'auc' in metrics and y_proba is not None:
                yp_proba = y_proba[indices]
                try:
                    boot_results['auc'].append(roc_auc_score(yt, yp_proba))
                except:
                    pass
            
            if 'accuracy' in metrics:
                boot_results['accuracy'].append(accuracy_score(yt, yp))
            if 'f1' in metrics:
                boot_results['f1'].append(f1_score(yt, yp, zero_division=0))
            if 'precision' in metrics:
                boot_results['precision'].append(precision_score(yt, yp, zero_division=0))
            if 'recall' in metrics:
                boot_results['recall'].append(recall_score(yt, yp, zero_division=0))
        
        # 计算置信区间
        alpha = 1 - self.confidence_level
        results = {}
        for metric, scores in boot_results.items():
            if scores:
                mean = np.mean(scores)
                lower = np.percentile(scores, alpha/2 * 100)
                upper = np.percentile(scores, (1 - alpha/2) * 100)
                results[metric] = (float(mean), float(lower), float(upper))
        
        return results
    
    def paired_model_comparison(self, scores_a: List[float], 
                                 scores_b: List[float],
                                 model_a_name: str = "Model A",
                                 model_b_name: str = "Model B") -> Dict[str, Any]:
        """
        配对t检验比较两个模型
        
        参数:
            scores_a: 模型A的交叉验证得分
            scores_b: 模型B的交叉验证得分
        """
        from scipy import stats
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Wilcoxon符号秩检验 (非参数)
        try:
            w_stat, w_pvalue = stats.wilcoxon(scores_a, scores_b)
        except:
            w_stat, w_pvalue = 0, 1.0
        
        return {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'mean_a': float(np.mean(scores_a)),
            'mean_b': float(np.mean(scores_b)),
            'std_a': float(np.std(scores_a)),
            'std_b': float(np.std(scores_b)),
            't_statistic': float(t_stat),
            't_p_value': float(p_value),
            'wilcoxon_statistic': float(w_stat),
            'wilcoxon_p_value': float(w_pvalue),
            'significant': p_value < 0.05,
            'better_model': model_a_name if np.mean(scores_a) > np.mean(scores_b) else model_b_name,
        }
    
    def generate_validation_report(self, y_true, y_pred, y_proba,
                                    model, X_train, y_train, X_test, y_test,
                                    cv_scores: List[float] = None,
                                    model_name: str = "Model") -> Dict[str, Any]:
        """生成完整统计验证报告"""
        report = {'model_name': model_name}
        
        # Y-scrambling
        try:
            logger.info(f"执行Y-scrambling验证 (n={self.n_scramble})...")
            y_scramble = self.y_scrambling(model, X_train, y_train, X_test, y_test)
            report['y_scrambling'] = y_scramble
        except Exception as e:
            logger.warning(f"Y-scrambling失败: {e}")
            report['y_scrambling'] = {'error': str(e)}
        
        # Bootstrap CI
        try:
            logger.info(f"执行Bootstrap验证 (n={self.n_bootstrap})...")
            boot_ci = self.bootstrap_confidence_interval(y_true, y_pred, y_proba)
            report['bootstrap_ci'] = boot_ci
        except Exception as e:
            logger.warning(f"Bootstrap失败: {e}")
            report['bootstrap_ci'] = {'error': str(e)}
        
        # CV得分统计
        if cv_scores:
            report['cv_stats'] = {
                'mean': float(np.mean(cv_scores)),
                'std': float(np.std(cv_scores)),
                'min': float(np.min(cv_scores)),
                'max': float(np.max(cv_scores)),
                'ci_lower': float(np.percentile(cv_scores, 2.5)),
                'ci_upper': float(np.percentile(cv_scores, 97.5)),
            }
        
        return report
    
    def plot_y_scrambling(self, y_scramble_result: Dict, 
                          output_path: Path = None) -> str:
        """绘制Y-scrambling图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            true_score = y_scramble_result['true_score']
            scrambled = y_scramble_result['scrambled_scores']
            scrambled_mean = y_scramble_result['scrambled_mean']
            scrambled_std = y_scramble_result['scrambled_std']
            
            # 绘制打乱得分
            x = range(1, len(scrambled) + 1)
            ax.scatter(x, scrambled, color='#E67E22', s=80, alpha=0.7, 
                      label=f'Scrambled (mean={scrambled_mean:.3f}±{scrambled_std:.3f})', zorder=2)
            
            # 绘制真实得分
            ax.axhline(y=true_score, color='#2E5C8A', linewidth=2.5, 
                      label=f'True Model ({true_score:.3f})', zorder=3)
            
            # 绘制阈值线
            threshold = y_scramble_result.get('threshold', scrambled_mean + 2*scrambled_std)
            ax.axhline(y=threshold, color='#C0392B', linestyle='--', linewidth=1.5,
                      label=f'Threshold ({threshold:.3f})', zorder=1)
            
            ax.set_xlabel('Scrambling Iteration', fontweight='bold')
            ax.set_ylabel('AUC Score', fontweight='bold')
            ax.set_title(f'Y-Scrambling Validation (n={len(scrambled)})', fontweight='bold')
            ax.legend(loc='lower right', frameon=True)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.4, 1.0])
            
            # 添加验证结果文本
            is_valid = y_scramble_result['is_valid']
            p_val = y_scramble_result.get('p_value', 1.0)
            status = '✓ PASSED' if is_valid else '✗ FAILED'
            ax.text(0.02, 0.98, f'Status: {status}\np-value: {p_val:.4f}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path) if output_path else ''
            
        except Exception as e:
            logger.error(f"Y-scrambling绘图失败: {e}")
            return ''
