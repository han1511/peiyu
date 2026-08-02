#!/bin/bash
# ==============================================================================
# DrugScreen AI 服务器部署脚本
# 
# 用法:
#   1. 将本脚本上传到服务器: scp deploy_server.sh user@server:/opt/
#   2. SSH登录服务器: ssh user@server
#   3. 执行: chmod +x /opt/deploy_server.sh && sudo /opt/deploy_server.sh
#
# 部署完成后:
#   - 从Windows浏览器访问: http://服务器IP:8501
#   - 上传Windows本地CSV数据文件进行筛选
#   - Windows本地代码完全不受影响
# ==============================================================================

set -e

# 配置
INSTALL_DIR="/opt/DrugScreenAI"
STREAMLIT_PORT=8501
PYTHON_VERSION="3.10"

echo "================================================"
echo "  DrugScreen AI 服务器部署脚本 v3.0"
echo "================================================"

# 1. 安装系统依赖
echo "[1/6] 安装系统依赖..."
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y python3 python3-pip python3-venv git wget \
        build-essential libfreetype6-dev libpng-dev liblapack-dev \
        libblas-dev gfortran
elif command -v yum &> /dev/null; then
    yum install -y python3 python3-pip git wget gcc gcc-c++ \
        freetype-devel libpng-devel lapack-devel blas-devel gcc-gfortran
fi

# 2. 创建目录
echo "[2/6] 创建项目目录..."
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/data
mkdir -p $INSTALL_DIR/results
mkdir -p $INSTALL_DIR/logs

# 3. 创建Python虚拟环境
echo "[3/6] 创建Python虚拟环境..."
python3 -m venv $INSTALL_DIR/venv
source $INSTALL_DIR/venv/bin/activate

# 4. 安装Python依赖
echo "[4/6] 安装Python依赖 (需要几分钟)..."
pip install --upgrade pip

pip install streamlit numpy pandas scikit-learn xgboost \
    matplotlib seaborn rdkit-pypi joblib \
    chembl-webresource-client scipy

# 尝试安装AutoDock Vina (可选)
pip install vina 2>/dev/null || echo "  Vina Python包不可用，将使用经验打分"

# 5. 创建配置文件
echo "[5/6] 创建服务器配置..."

cat > $INSTALL_DIR/run_server.sh << 'SCRIPT'
#!/bin/bash
# DrugScreen AI 启动脚本

cd /opt/DrugScreenAI
source /opt/DrugScreenAI/venv/bin/activate

# 确保进程模式 (Linux无中文路径问题)
export DRUGSCREEN_MODE=server

echo "=========================================="
echo "  DrugScreen AI 服务器启动"
echo "  访问地址: http://$(hostname -I | awk '{print $1}'):8501"
echo "  按 Ctrl+C 停止"
echo "=========================================="

streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 500 \
    --server.maxWebsocketSize 50
SCRIPT

chmod +x $INSTALL_DIR/run_server.sh

# 6. 创建systemd服务 (开机自启)
echo "[6/6] 配置开机自启服务..."

cat > /etc/systemd/system/drugscreen.service << EOF
[Unit]
Description=DrugScreen AI Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false --server.maxUploadSize 500
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable drugscreen

echo ""
echo "================================================"
echo "  部署完成!"
echo "================================================"
echo ""
echo "  项目目录: $INSTALL_DIR"
echo "  启动命令: $INSTALL_DIR/run_server.sh"
echo "  服务管理: systemctl start|stop|restart drugscreen"
echo "  查看日志: journalctl -u drugscreen -f"
echo ""
echo "  下一步: 将代码上传到 $INSTALL_DIR/"
echo "  然后执行: systemctl start drugscreen"
echo "================================================"
