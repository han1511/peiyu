#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器模块

支持从YAML配置文件加载配置，并提供环境变量覆盖功能
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class PathConfig:
    project_root: str = "auto"
    data_dir: str = "data"
    results_dir: str = "results"
    logs_dir: str = "results/logs"
    temp_dir: str = "temp"
    cache_dir: str = ".cache"


@dataclass
class DockingConfig:
    software: str = "AutoDock Vina"
    exhaustiveness: int = 32
    num_poses: int = 20
    search_space: Dict[str, Any] = field(default_factory=dict)
    energy_range: float = 3.0
    cpu_cores: int = -1


@dataclass
class MLConfig:
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_split: Dict[str, Any] = field(default_factory=dict)
    cross_validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfig:
    formats: list = field(default_factory=lambda: ["markdown", "html"])
    include_figures: bool = True
    include_tables: bool = True
    precision: int = 4
    max_compounds_to_display: int = 100
    template: str = "default"
    language: str = "zh"


@dataclass
class QualityControlConfig:
    enabled: bool = True
    max_missing_ratio: float = 0.1
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    applicability_domain: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    project_name: str = "DrugScreen AI"
    version: str = "2.0.0"
    paths: PathConfig = field(default_factory=PathConfig)
    targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compound_libraries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ml: MLConfig = field(default_factory=MLConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    lipinski_rules: Dict[str, Any] = field(default_factory=dict)
    drug_likeness_filters: Dict[str, Any] = field(default_factory=dict)
    admet_thresholds: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    parallel: Dict[str, Any] = field(default_factory=dict)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    visualization: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    配置管理器
    
    支持从YAML文件加载配置，并通过环境变量覆盖
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._config is not None:
            return
            
        self.config_path = config_path or self._find_config_file()
        self._config = self._load_config()
        self._apply_env_overrides()
        self._resolve_paths()
    
    def _find_config_file(self) -> str:
        """自动查找配置文件"""
        possible_paths = [
            "settings.yaml",
            "configs/settings.yaml",
            "config.yaml",
            "configs/config.yaml",
        ]
        
        current_dir = Path(__file__).parent.parent
        
        for path in possible_paths:
            full_path = current_dir / path
            if full_path.exists():
                return str(full_path)
        
        raise FileNotFoundError("找不到配置文件 settings.yaml")
    
    def _load_config(self) -> AppConfig:
        """从YAML加载配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return self._dict_to_config(data)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """将字典转换为配置对象"""
        paths = PathConfig(**data.get('paths', {}))
        
        ml_data = data.get('ml_models', {})
        ml_config = MLConfig(
            models=ml_data,
            data_split=data.get('data_split', {}),
            cross_validation=data.get('cross_validation', {})
        )
        
        docking_data = data.get('docking', {})
        docking_config = DockingConfig(**docking_data)
        
        qc_data = data.get('quality_control', {})
        qc_config = QualityControlConfig(**qc_data)
        
        report_data = data.get('report', {})
        report_config = ReportConfig(**report_data)
        
        return AppConfig(
            project_name=data.get('project', {}).get('name', 'DrugScreen AI'),
            version=data.get('project', {}).get('version', '2.0.0'),
            paths=paths,
            targets=data.get('targets', {}),
            compound_libraries=data.get('compound_libraries', {}),
            ml=ml_config,
            docking=docking_config,
            lipinski_rules=data.get('lipinski_rules', {}),
            drug_likeness_filters=data.get('drug_likeness_filters', {}),
            admet_thresholds=data.get('admet_thresholds', {}),
            logging=data.get('logging', {}),
            parallel=data.get('parallel', {}),
            quality_control=qc_config,
            report=report_config,
            visualization=data.get('visualization', {})
        )
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_mappings = {
            'DRUGSCREEN_DATA_DIR': ['paths', 'data_dir'],
            'DRUGSCREEN_RESULTS_DIR': ['paths', 'results_dir'],
            'DRUGSCREEN_LOG_LEVEL': ['logging', 'level'],
            'DRUGSCREEN_N_JOBS': ['parallel', 'n_jobs'],
            'DRUGSCREEN_DOCKING_EXHAUSTIVENESS': ['docking', 'exhaustiveness'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_attr(self._config, config_path, value)
    
    def _set_nested_attr(self, obj, path: list, value: Any):
        """设置嵌套属性"""
        for key in path[:-1]:
            obj = getattr(obj, key, None)
            if obj is None:
                return
        
        final_key = path[-1]
        if hasattr(obj, final_key):
            current = getattr(obj, final_key)
            # 类型转换
            if isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            elif isinstance(current, bool):
                value = value.lower() in ('true', '1', 'yes', 'on')
            setattr(obj, final_key, value)
    
    def _resolve_paths(self):
        """解析路径为绝对路径"""
        if self._config.paths.project_root == "auto":
            project_root = Path(__file__).parent.parent.resolve()
        else:
            project_root = Path(self._config.paths.project_root).resolve()
        
        self._config.paths.project_root = str(project_root)
        self._config.paths.data_dir = str(project_root / self._config.paths.data_dir)
        self._config.paths.results_dir = str(project_root / self._config.paths.results_dir)
        self._config.paths.logs_dir = str(project_root / self._config.paths.logs_dir)
        self._config.paths.temp_dir = str(project_root / self._config.paths.temp_dir)
        self._config.paths.cache_dir = str(project_root / self._config.paths.cache_dir)
        
        # 创建目录
        for path_attr in ['data_dir', 'results_dir', 'logs_dir', 'temp_dir', 'cache_dir']:
            path = Path(getattr(self._config.paths, path_attr))
            path.mkdir(parents=True, exist_ok=True)
    
    @property
    def config(self) -> AppConfig:
        """获取配置对象"""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self):
        """重新加载配置"""
        self._config = self._load_config()
        self._apply_env_overrides()
        self._resolve_paths()
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return asdict(self._config)


# 全局配置管理器实例
def get_config(config_path: Optional[str] = None) -> AppConfig:
    """获取配置实例"""
    manager = ConfigManager(config_path)
    return manager.config


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取配置管理器实例"""
    return ConfigManager(config_path)
