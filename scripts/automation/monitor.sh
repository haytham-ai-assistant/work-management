#!/bin/bash
# 文档同步监控脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/workspace/工作"
LOG_DIR="${WORKSPACE_DIR}/logs/automation"
CONFIG_FILE="${SCRIPT_DIR}/../config/automation.conf"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 加载配置
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi

# 检查函数
check_disk_space() {
    local usage=$(df -h "${WORKSPACE_DIR}" | tail -1 | awk '{print $5}' | sed 's/%//')
    local available=$(df -h "${WORKSPACE_DIR}" | tail -1 | awk '{print $4}')
    
    if [[ ${usage} -gt 90 ]]; then
        echo -e "${RED}✗ 磁盘空间不足: ${usage}% 已使用${NC}"
        return 1
    elif [[ ${usage} -gt 70 ]]; then
        echo -e "${YELLOW}⚠ 磁盘空间警告: ${usage}% 已使用${NC}"
        return 2
    else
        echo -e "${GREEN}✓ 磁盘空间正常: ${usage}% 已使用 (可用: ${available})${NC}"
        return 0
    fi
}

check_git_status() {
    cd "${WORKSPACE_DIR}"
    
    if ! git status &> /dev/null; then
        echo -e "${RED}✗ Git仓库状态异常${NC}"
        return 1
    fi
    
    local changes=$(git status --short | wc -l)
    local branch=$(git branch --show-current)
    local remote_status=$(git status -uno | grep -c "Your branch is" || true)
    
    echo -e "${GREEN}✓ Git仓库正常${NC}"
    echo "  分支: ${branch}"
    echo "  未提交更改: ${changes} 个"
    
    if [[ ${changes} -gt 0 ]]; then
        echo -e "${YELLOW}  提示: 有未提交的更改${NC}"
    fi
    
    return 0
}

check_log_files() {
    local latest_log=$(find "${LOG_DIR}" -name "*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -z "${latest_log}" ]]; then
        echo -e "${YELLOW}⚠ 未找到日志文件${NC}"
        return 2
    fi
    
    local log_age=$(($(date +%s) - $(stat -c %Y "${latest_log}")))
    local log_size=$(stat -c %s "${latest_log}")
    
    if [[ ${log_age} -gt 86400 ]]; then # 24小时
        echo -e "${YELLOW}⚠ 最近日志超过24小时: $(basename "${latest_log}")${NC}"
        return 2
    fi
    
    # 检查错误
    local error_count=$(grep -c "错误\|ERROR\|失败\|FAILED" "${latest_log}" 2>/dev/null || echo 0)
    
    if [[ ${error_count} -gt 0 ]]; then
        echo -e "${RED}✗ 日志中发现错误: ${error_count} 个 (文件: $(basename "${latest_log}"))${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ 日志文件正常${NC}"
    echo "  最新日志: $(basename "${latest_log}")"
    echo "  大小: $((${log_size}/1024))KB"
    echo "  年龄: $((${log_age}/3600)) 小时前"
    
    return 0
}

check_sync_status() {
    local sync_enabled="${GIT_ENABLED:-true}"
    
    if [[ "${sync_enabled}" != "true" ]]; then
        echo -e "${YELLOW}⚠ 自动同步已禁用${NC}"
        return 2
    fi
    
    # 检查最后一次同步时间
    local sync_logs=("${LOG_DIR}"/*.log)
    local latest_sync=""
    
    for log in "${sync_logs[@]}"; do
        if [[ -f "${log}" ]] && grep -q "开始文档自动化同步" "${log}" 2>/dev/null; then
            if [[ -z "${latest_sync}" ]] || [[ "${log}" -nt "${latest_sync}" ]]; then
                latest_sync="${log}"
            fi
        fi
    done
    
    if [[ -z "${latest_sync}" ]]; then
        echo -e "${YELLOW}⚠ 未找到同步记录${NC}"
        return 2
    fi
    
    local sync_time=$(stat -c %Y "${latest_sync}")
    local current_time=$(date +%s)
    local hours_since=$(( (current_time - sync_time) / 3600 ))
    
    if [[ ${hours_since} -gt 24 ]]; then
        echo -e "${RED}✗ 同步已停止: 最后一次同步在 ${hours_since} 小时前${NC}"
        return 1
    elif [[ ${hours_since} -gt 6 ]]; then
        echo -e "${YELLOW}⚠ 同步延迟: 最后一次同步在 ${hours_since} 小时前${NC}"
        return 2
    else
        echo -e "${GREEN}✓ 同步状态正常${NC}"
        echo "  最后一次同步: ${hours_since} 小时前"
        echo "  日志文件: $(basename "${latest_sync}")"
        return 0
    fi
}

check_tapd_integration() {
    local tapd_enabled="${TAPD_ENABLED:-false}"
    
    if [[ "${tapd_enabled}" != "true" ]]; then
        echo -e "${YELLOW}⚠ TAPD集成已禁用${NC}"
        return 2
    fi
    
    local tapd_dir="${WORKSPACE_DIR}/scripts/tapd_integration"
    local config_file="${tapd_dir}/config.py"
    
    if [[ ! -f "${config_file}" ]]; then
        echo -e "${RED}✗ TAPD配置文件不存在${NC}"
        return 1
    fi
    
    # 检查配置是否有效
    if grep -q "your_company_id\|your_api_key\|your_api_secret" "${config_file}"; then
        echo -e "${RED}✗ TAPD配置未填写${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ TAPD集成配置正常${NC}"
    return 0
}

generate_report() {
    echo ""
    echo "========================================"
    echo "文档同步系统监控报告"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "工作目录: ${WORKSPACE_DIR}"
    echo "========================================"
    echo ""
    
    local overall_status=0
    
    echo "1. 磁盘空间检查:"
    if ! check_disk_space; then
        overall_status=1
    fi
    echo ""
    
    echo "2. Git仓库状态:"
    if ! check_git_status; then
        overall_status=1
    fi
    echo ""
    
    echo "3. 日志文件状态:"
    if ! check_log_files; then
        overall_status=1
    fi
    echo ""
    
    echo "4. 同步状态检查:"
    if ! check_sync_status; then
        overall_status=1
    fi
    echo ""
    
    echo "5. TAPD集成检查:"
    if ! check_tapd_integration; then
        overall_status=1
    fi
    echo ""
    
    echo "========================================"
    if [[ ${overall_status} -eq 0 ]]; then
        echo -e "${GREEN}✅ 所有检查通过 - 系统正常${NC}"
    elif [[ ${overall_status} -eq 2 ]]; then
        echo -e "${YELLOW}⚠ 系统有警告 - 需要关注${NC}"
    else
        echo -e "${RED}❌ 系统有问题 - 需要修复${NC}"
    fi
    echo "========================================"
    
    return ${overall_status}
}

# 主函数
main() {
    echo "开始文档同步系统监控..."
    echo ""
    
    # 确保日志目录存在
    mkdir -p "${LOG_DIR}"
    
    # 生成报告
    generate_report
    
    local exit_code=$?
    
    echo ""
    echo "监控完成"
    
    exit ${exit_code}
}

# 执行主函数
main "$@"