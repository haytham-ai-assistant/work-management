#!/bin/bash
# Cron任务配置脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="${SCRIPT_DIR}/sync_all.sh"
CRON_FILE="/etc/cron.d/work-management-sync"

# 检查脚本是否存在
if [[ ! -f "${SYNC_SCRIPT}" ]]; then
    echo "错误: 同步脚本不存在: ${SYNC_SCRIPT}"
    exit 1
fi

# 检查执行权限
chmod +x "${SYNC_SCRIPT}"

# 创建cron配置
echo "配置Cron定时任务..."
cat > "${CRON_FILE}" << EOF
# 工作区文档自动化同步任务
# 每天凌晨2点执行
0 2 * * * root ${SYNC_SCRIPT} >> /var/log/work-management-sync.log 2>&1

# 每小时检查（可选）
# 0 * * * * root ${SYNC_SCRIPT} --quick >> /var/log/work-management-sync-quick.log 2>&1
EOF

# 设置权限
chmod 644 "${CRON_FILE}"

echo "Cron配置已创建: ${CRON_FILE}"
echo ""
echo "当前Cron配置:"
cat "${CRON_FILE}"
echo ""
echo "要手动测试同步，运行:"
echo "  ${SYNC_SCRIPT}"
echo ""
echo "要查看cron日志，检查:"
echo "  /var/log/work-management-sync.log"
echo ""
echo "注意: 需要root权限创建系统cron任务"
echo "如果无root权限，可以使用用户cron:"
echo "  crontab -e"
echo "添加: 0 2 * * * ${SYNC_SCRIPT}"