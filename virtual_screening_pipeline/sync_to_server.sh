#!/bin/bash
# ==============================================================================
# Windows → 服务器同步脚本
# 
# 功能:
#   1. 将本地代码同步到服务器 (rsync, 不删除服务器上已有结果)
#   2. 自动将进程模式改回 loky (Linux无中文路径问题)
#   3. 重启服务器上的应用
#
# 用法:
#   在Windows Git Bash中运行:
#   bash sync_to_server.sh user@your-server-ip
# ==============================================================================

SERVER=${1:-"user@your-server-ip"}
REMOTE_DIR="/opt/DrugScreenAI"

echo "=========================================="
echo "  同步 DrugScreen AI → $SERVER"
echo "=========================================="

# 1. 同步代码 (排除结果和缓存，不破坏服务器上已有的数据)
echo "[1/3] 同步代码文件..."
rsync -avz --progress \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='.checkpoints/' \
    --exclude='*.pyc' \
    --exclude='_safe_temp/' \
    --exclude='myenv/' \
    --exclude='docking_work/' \
    --exclude='chembl_cache/' \
    --exclude='.git/' \
    --exclude='deploy_server.sh' \
    --exclude='sync_to_server.sh' \
    --exclude='build_exe.py' \
    --exclude='drugscreen.spec' \
    --exclude='hooks/' \
    ./ ${SERVER}:${REMOTE_DIR}/

echo "[2/3] 修复服务器端并行模式 (loky进程模式)..."
ssh $SERVER "cd $REMOTE_DIR && \
    sed -i \"s/prefer='threads'/prefer='processes'/g\" app.py && \
    sed -i \"s/prefer=\\\"threads\\\"/prefer=\\\"processes\\\"/g\" app.py && \
    echo '已将线程模式改回进程模式'"

echo "[3/3] 重启服务器应用..."
ssh $SERVER "systemctl restart drugscreen || \
    (cd $REMOTE_DIR && pkill -f streamlit; sleep 1; nohup $REMOTE_DIR/run_server.sh > $REMOTE_DIR/logs/server.log 2>&1 &)"

echo ""
echo "=========================================="
echo "  同步完成!"
echo "  访问: http://$SERVER:8501"
echo "=========================================="
