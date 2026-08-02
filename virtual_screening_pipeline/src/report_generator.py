#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文级HTML报告生成模块

功能：
1. 生成符合学术论文标准的HTML报告
2. 包含所有可视化图表和统计表格
3. 支持中英文双语
4. 响应式设计，支持打印为PDF
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """
    HTML报告生成器
    
    生成专业级的虚拟筛选报告，适用于论文发表和工业汇报
    """
    
    def __init__(self, output_dir: Optional[Path] = None, 
                 title: str = "虚拟筛选报告",
                 language: str = "zh"):
        """
        初始化报告生成器
        
        参数:
            output_dir: 输出目录
            title: 报告标题
            language: 语言 ('zh' 或 'en')
        """
        self.output_dir = output_dir or Path("results/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.language = language
        self.sections = []
        self.data = {}
        
        # CSS样式
        self.css_style = self._get_css_style()
        
    def _get_css_style(self) -> str:
        """获取CSS样式"""
        return """
        <style>
            :root {
                --primary-color: #2E5C8A;
                --secondary-color: #E67E22;
                --success-color: #27AE60;
                --danger-color: #C0392B;
                --bg-color: #f8f9fa;
                --text-color: #333;
                --border-color: #dee2e6;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: #fff;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            /* 封面 */
            .cover {
                text-align: center;
                padding: 80px 20px;
                background: linear-gradient(135deg, var(--primary-color) 0%, #1a3a5c 100%);
                color: white;
                margin-bottom: 40px;
                border-radius: 10px;
            }
            
            .cover h1 {
                font-size: 2.5em;
                margin-bottom: 20px;
                font-weight: 300;
            }
            
            .cover .subtitle {
                font-size: 1.2em;
                opacity: 0.9;
                margin-bottom: 30px;
            }
            
            .cover .meta {
                font-size: 0.9em;
                opacity: 0.8;
            }
            
            /* 摘要 */
            .abstract {
                background: var(--bg-color);
                padding: 30px;
                border-radius: 8px;
                margin-bottom: 30px;
                border-left: 4px solid var(--primary-color);
            }
            
            .abstract h2 {
                color: var(--primary-color);
                margin-bottom: 15px;
            }
            
            /* 章节 */
            .section {
                margin-bottom: 40px;
            }
            
            .section h2 {
                color: var(--primary-color);
                border-bottom: 2px solid var(--primary-color);
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            
            .section h3 {
                color: #444;
                margin: 20px 0 10px 0;
            }
            
            /* 表格 */
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .data-table th {
                background: var(--primary-color);
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }
            
            .data-table td {
                padding: 10px 12px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .data-table tr:hover {
                background: #f8f9fa;
            }
            
            .data-table .highlight {
                background: #fff3cd !important;
                font-weight: 600;
            }
            
            /* 指标卡片 */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid var(--primary-color);
            }
            
            .metric-card .label {
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }
            
            .metric-card .value {
                font-size: 1.8em;
                font-weight: bold;
                color: var(--primary-color);
            }
            
            .metric-card .unit {
                font-size: 0.8em;
                color: #999;
            }
            
            /* 图表容器 */
            .figure {
                margin: 30px 0;
                text-align: center;
            }
            
            .figure img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .figure-caption {
                margin-top: 10px;
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            
            /* 化合物卡片 */
            .compound-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .compound-card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border: 1px solid var(--border-color);
            }
            
            .compound-card .rank {
                display: inline-block;
                background: var(--primary-color);
                color: white;
                width: 30px;
                height: 30px;
                line-height: 30px;
                text-align: center;
                border-radius: 50%;
                margin-bottom: 10px;
            }
            
            .compound-card .smiles {
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                background: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                word-break: break-all;
                margin: 10px 0;
            }
            
            /* 结论框 */
            .conclusion {
                background: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            
            .conclusion h3 {
                color: #155724;
                margin-bottom: 10px;
            }
            
            /* 页脚 */
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.85em;
                border-top: 1px solid var(--border-color);
                margin-top: 40px;
            }
            
            /* 打印样式 */
            @media print {
                .cover {
                    page-break-after: always;
                }
                .section {
                    page-break-inside: avoid;
                }
                .figure {
                    page-break-inside: avoid;
                }
            }
            
            /* 响应式 */
            @media (max-width: 768px) {
                .cover h1 {
                    font-size: 1.8em;
                }
                .metrics-grid {
                    grid-template-columns: 1fr;
                }
                .compound-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """
    
    def add_data(self, key: str, data: Any):
        """
        添加数据到报告
        
        参数:
            key: 数据键名
            data: 数据内容
        """
        self.data[key] = data
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图片转为base64"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"无法加载图片 {image_path}: {e}")
            return ""
    
    def _create_cover(self) -> str:
        """创建封面"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        texts = {
            'zh': {
                'subtitle': '基于机器学习的登革病毒抑制剂虚拟筛选',
                'generated': f'生成时间: {now}',
                'version': 'DrugScreen AI v2.0'
            },
            'en': {
                'subtitle': 'Machine Learning-Based Virtual Screening for Dengue Virus Inhibitors',
                'generated': f'Generated: {now}',
                'version': 'DrugScreen AI v2.0'
            }
        }
        
        t = texts.get(self.language, texts['zh'])
        
        return f"""
        <div class="cover">
            <h1>{self.title}</h1>
            <div class="subtitle">{t['subtitle']}</div>
            <div class="meta">
                <p>{t['generated']}</p>
                <p>{t['version']}</p>
            </div>
        </div>
        """
    
    def _create_abstract(self) -> str:
        """创建摘要"""
        texts = {
            'zh': {
                'title': '摘要',
                'content': '本报告展示了基于机器学习的虚拟筛选流程结果，包括靶点分析、化合物库预处理、分子特征计算、模型训练与评估、分子对接和ADMET性质预测。筛选流程旨在发现潜在的登革病毒抑制剂候选化合物。'
            },
            'en': {
                'title': 'Abstract',
                'content': 'This report presents the results of a machine learning-based virtual screening workflow, including target analysis, compound library preprocessing, molecular feature computation, model training and evaluation, molecular docking, and ADMET property prediction. The screening pipeline aims to discover potential dengue virus inhibitor candidates.'
            }
        }
        
        t = texts.get(self.language, texts['zh'])
        
        # 汇总统计
        stats = self.data.get('summary_stats', {})
        
        return f"""
        <div class="abstract">
            <h2>{t['title']}</h2>
            <p>{t['content']}</p>
            {self._create_metrics_grid(stats) if stats else ''}
        </div>
        """
    
    def _create_metrics_grid(self, stats: Dict[str, Any]) -> str:
        """创建指标网格"""
        cards = []
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                display_value = f"{value:.3f}" if isinstance(value, float) and value < 100 else str(value)
                cards.append(f"""
                    <div class="metric-card">
                        <div class="label">{key}</div>
                        <div class="value">{display_value}</div>
                    </div>
                """)
        
        return f"<div class='metrics-grid'>{''.join(cards)}</div>"
    
    def _create_dataframe_table(self, df: pd.DataFrame, 
                                 title: str = "",
                                 highlight_column: Optional[str] = None,
                                 n_rows: int = 20) -> str:
        """将DataFrame转为HTML表格"""
        if len(df) > n_rows:
            df = df.head(n_rows)
            note = f"<p style='color:#666;font-size:0.9em;'>显示前 {n_rows} 行 (共 {len(df)} 行)</p>"
        else:
            note = ""
        
        html = df.to_html(classes='data-table', index=False, escape=False)
        
        return f"""
        <div class="section">
            <h3>{title}</h3>
            {html}
            {note}
        </div>
        """
    
    def _create_figure_section(self, image_path: str, 
                                caption: str = "",
                                title: str = "") -> str:
        """创建图表章节"""
        img_data = self._image_to_base64(image_path)
        if not img_data:
            return ""
        
        ext = Path(image_path).suffix.lower().replace('.', '')
        
        return f"""
        <div class="section">
            <h3>{title}</h3>
            <div class="figure">
                <img src="data:image/{ext};base64,{img_data}" alt="{caption}">
                <div class="figure-caption">{caption}</div>
            </div>
        </div>
        """
    
    def _create_compound_cards(self, compounds_df: pd.DataFrame, 
                                n_display: int = 10) -> str:
        """创建化合物卡片"""
        if len(compounds_df) == 0:
            return ""
        
        df = compounds_df.head(n_display)
        cards = []
        
        for idx, row in df.iterrows():
            rank = idx + 1 if isinstance(idx, int) else 1
            smiles = row.get('SMILES', row.get('smiles', 'N/A'))
            name = row.get('Name', row.get('name', f'Compound_{rank}'))
            affinity = row.get('binding_affinity', row.get('affinity', 'N/A'))
            
            card = f"""
            <div class="compound-card">
                <span class="rank">{rank}</span>
                <h4>{name}</h4>
                <div class="smiles">{smiles}</div>
                <p><strong>结合能:</strong> {affinity} kcal/mol</p>
            </div>
            """
            cards.append(card)
        
        return f"""
        <div class="section">
            <h3>Top {len(df)} 候选化合物</h3>
            <div class="compound-grid">
                {''.join(cards)}
            </div>
        </div>
        """
    
    def _create_conclusion(self) -> str:
        """创建结论"""
        texts = {
            'zh': {
                'title': '结论与建议',
                'content': '基于虚拟筛选结果，我们推荐对Top化合物进行进一步的实验验证，包括体外酶活抑制实验和细胞水平抗病毒活性测试。建议优先考虑具有良好ADMET性质且结合能较低的化合物。'
            },
            'en': {
                'title': 'Conclusions and Recommendations',
                'content': 'Based on the virtual screening results, we recommend further experimental validation of the top-ranked compounds, including in vitro enzymatic inhibition assays and cell-based antiviral activity tests. Priority should be given to compounds with favorable ADMET properties and low binding energies.'
            }
        }
        
        t = texts.get(self.language, texts['zh'])
        
        return f"""
        <div class="conclusion">
            <h3>{t['title']}</h3>
            <p>{t['content']}</p>
        </div>
        """
    
    def generate_report(self, 
                       model_plots: List[str] = None,
                       docking_plots: List[str] = None,
                       admet_plots: List[str] = None,
                       chemical_space_plots: List[str] = None,
                       compound_plots: List[str] = None,
                       output_filename: Optional[str] = None) -> str:
        """
        生成完整HTML报告
        
        参数:
            model_plots: 模型性能图表路径列表
            docking_plots: 对接结果图表路径列表
            admet_plots: ADMET图表路径列表
            chemical_space_plots: 化学空间图表路径列表
            compound_plots: 化合物结构图表路径列表
            output_filename: 输出文件名
            
        返回:
            str: 生成的HTML文件路径
        """
        texts = {
            'zh': {
                'model_section': '1. 模型性能评估',
                'docking_section': '2. 分子对接结果',
                'admet_section': '3. ADMET性质分析',
                'chemical_space_section': '4. 化学空间分析',
                'compound_section': '5. 化合物结构展示',
                'data_section': '6. 详细数据'
            },
            'en': {
                'model_section': '1. Model Performance Evaluation',
                'docking_section': '2. Molecular Docking Results',
                'admet_section': '3. ADMET Property Analysis',
                'chemical_space_section': '4. Chemical Space Analysis',
                'compound_section': '5. Compound Structure Display',
                'data_section': '6. Detailed Data'
            }
        }
        
        t = texts.get(self.language, texts['zh'])
        
        # 构建HTML内容
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            f"<title>{self.title}</title>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<style>{self.css_style}</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            self._create_cover(),
            self._create_abstract(),
        ]
        
        # 模型性能
        if model_plots:
            html_parts.append(f"<div class='section'><h2>{t['model_section']}</h2>")
            for plot_path in model_plots:
                if Path(plot_path).exists():
                    html_parts.append(self._create_figure_section(
                        plot_path,
                        caption=Path(plot_path).stem.replace('_', ' ').title()
                    ))
            html_parts.append("</div>")
        
        # 分子对接
        if docking_plots:
            html_parts.append(f"<div class='section'><h2>{t['docking_section']}</h2>")
            for plot_path in docking_plots:
                if Path(plot_path).exists():
                    html_parts.append(self._create_figure_section(
                        plot_path,
                        caption=Path(plot_path).stem.replace('_', ' ').title()
                    ))
            html_parts.append("</div>")
        
        # ADMET
        if admet_plots:
            html_parts.append(f"<div class='section'><h2>{t['admet_section']}</h2>")
            for plot_path in admet_plots:
                if Path(plot_path).exists():
                    html_parts.append(self._create_figure_section(
                        plot_path,
                        caption=Path(plot_path).stem.replace('_', ' ').title()
                    ))
            html_parts.append("</div>")
        
        # 化学空间
        if chemical_space_plots:
            html_parts.append(f"<div class='section'><h2>{t['chemical_space_section']}</h2>")
            for plot_path in chemical_space_plots:
                if Path(plot_path).exists():
                    html_parts.append(self._create_figure_section(
                        plot_path,
                        caption=Path(plot_path).stem.replace('_', ' ').title()
                    ))
            html_parts.append("</div>")
        
        # 化合物结构
        if compound_plots:
            html_parts.append(f"<div class='section'><h2>{t['compound_section']}</h2>")
            for plot_path in compound_plots:
                if Path(plot_path).exists():
                    html_parts.append(self._create_figure_section(
                        plot_path,
                        caption=Path(plot_path).stem.replace('_', ' ').title()
                    ))
            html_parts.append("</div>")
        
        # 详细数据表
        html_parts.append(f"<div class='section'><h2>{t['data_section']}</h2>")
        
        # 模型性能表
        model_metrics = self.data.get('model_metrics')
        if model_metrics and isinstance(model_metrics, pd.DataFrame):
            html_parts.append(self._create_dataframe_table(
                model_metrics,
                title="模型性能指标"
            ))
        
        # Top化合物表
        top_compounds = self.data.get('top_compounds')
        if top_compounds and isinstance(top_compounds, pd.DataFrame):
            html_parts.append(self._create_dataframe_table(
                top_compounds,
                title="Top候选化合物"
            ))
            html_parts.append(self._create_compound_cards(top_compounds))
        
        html_parts.append("</div>")
        
        # 结论
        html_parts.append(self._create_conclusion())
        
        # 页脚
        html_parts.append(f"""
        <div class="footer">
            <p>Generated by DrugScreen AI v2.0 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        """)
        
        html_parts.extend(["</div>", "</body>", "</html>"])
        
        # 保存文件
        if output_filename is None:
            output_filename = f"screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        
        logger.info(f"报告已生成: {output_path}")
        return str(output_path)
    
    def generate_summary_json(self, output_filename: Optional[str] = None) -> str:
        """
        生成JSON摘要报告
        
        参数:
            output_filename: 输出文件名
            
        返回:
            str: JSON文件路径
        """
        summary = {
            'project_name': self.title,
            'generated_at': datetime.now().isoformat(),
            'data': {}
        }
        
        for key, value in self.data.items():
            if isinstance(value, pd.DataFrame):
                summary['data'][key] = value.head(20).to_dict('records')
            elif isinstance(value, np.ndarray):
                summary['data'][key] = value.tolist()
            else:
                summary['data'][key] = value
        
        if output_filename is None:
            output_filename = f"screening_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return str(output_path)


def generate_batch_reports(results_dir: Path,
                           report_configs: List[Dict[str, Any]]) -> List[str]:
    """
    批量生成报告
    
    参数:
        results_dir: 结果目录
        report_configs: 报告配置列表
        
    返回:
        List[str]: 生成的报告路径列表
    """
    generated = []
    
    for config in report_configs:
        generator = HTMLReportGenerator(
            output_dir=results_dir / "reports",
            title=config.get('title', 'Virtual Screening Report'),
            language=config.get('language', 'zh')
        )
        
        # 添加数据
        for key, data in config.get('data', {}).items():
            generator.add_data(key, data)
        
        # 生成报告
        report_path = generator.generate_report(
            model_plots=config.get('model_plots', []),
            docking_plots=config.get('docking_plots', []),
            admet_plots=config.get('admet_plots', []),
            chemical_space_plots=config.get('chemical_space_plots', []),
            compound_plots=config.get('compound_plots', []),
            output_filename=config.get('output_filename')
        )
        
        generated.append(report_path)
    
    return generated
