#!/usr/bin/env python3
"""
PyInstaller Hook: Streamlit

确保Streamlit的所有静态资源和子模块被正确打包
"""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# 收集Streamlit数据文件 (静态资源、模板等)
datas = collect_data_files('streamlit')

# 收集Streamlit子模块
hiddenimports = collect_submodules('streamlit')

# 收集Streamlit依赖包的元数据
datas += copy_metadata('streamlit')
datas += copy_metadata('altair')
datas += copy_metadata('pyarrow')
datas += copy_metadata('tornado')
datas += copy_metadata('click')
datas += copy_metadata('rich')
datas += copy_metadata('blinker')
