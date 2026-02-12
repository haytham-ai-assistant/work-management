#!/bin/bash
# 文档自动化同步脚本
# 协调所有文档同步任务

set -euo pipefail

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/workspace/工作"
LOG_DIR="${WORKSPACE_DIR}/logs/automation"
LOG_FILE="${LOG_DIR}/sync_$(date +%Y%m%d_%H%M%S).log"
CONFIG_FILE="${SCRIPT_DIR}/../config/automation.conf"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# 错误处理
handle_error() {
    log "错误: $1"
    log "同步失败，退出码: $2"
    exit "$2"
}

# 加载配置
load_config() {
    if [[ -f "${CONFIG_FILE}" ]]; then
        source "${CONFIG_FILE}"
        log "配置文件已加载: ${CONFIG_FILE}"
    else
        log "警告: 配置文件不存在: ${CONFIG_FILE}"
        log "使用默认配置"
    fi
}

# 检查依赖
check_dependencies() {
    local missing_deps=()
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # 检查Git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        handle_error "缺少依赖: ${missing_deps[*]}" 1
    fi
    
    log "依赖检查通过"
}

# 同步TAPD数据
sync_tapd() {
    local tapd_enabled="${TAPD_ENABLED:-false}"
    
    if [[ "${tapd_enabled}" != "true" ]]; then
        log "TAPD同步已禁用"
        return 0
    fi
    
    log "开始TAPD数据同步..."
    
    local tapd_script="${SCRIPT_DIR}/../tapd_integration/sync_requirements.py"
    if [[ -f "${tapd_script}" ]]; then
        cd "${WORKSPACE_DIR}/scripts/tapd_integration"
        if python3 "${tapd_script}"; then
            log "TAPD需求同步成功"
        else
            log "警告: TAPD需求同步失败"
        fi
    else
        log "警告: TAPD同步脚本不存在: ${tapd_script}"
    fi
    
    log "TAPD数据同步完成"
}

# 同步Git仓库
sync_git() {
    local git_enabled="${GIT_ENABLED:-true}"
    
    if [[ "${git_enabled}" != "true" ]]; then
        log "Git同步已禁用"
        return 0
    fi
    
    log "开始Git仓库同步..."
    
    cd "${WORKSPACE_DIR}"
    
    # 检查是否有更改
    if git diff --quiet && git diff --cached --quiet; then
        log "没有需要提交的更改"
        return 0
    fi
    
    # 添加所有更改
    git add .
    
    # 提交更改
    local commit_message="自动化同步: $(date '+%Y-%m-%d %H:%M:%S')"
    if git commit -m "${commit_message}"; then
        log "Git提交成功: ${commit_message}"
    else
        log "警告: Git提交失败"
        return 0
    fi
    
    # 推送到远程
    if git push origin main; then
        log "Git推送成功"
    else
        log "警告: Git推送失败"
    fi
    
    log "Git仓库同步完成"
}

# 清理旧日志
cleanup_logs() {
    local keep_days="${LOG_KEEP_DAYS:-7}"
    
    log "清理超过 ${keep_days} 天的旧日志..."
    
    find "${LOG_DIR}" -name "*.log" -type f -mtime "+${keep_days}" -delete 2>/dev/null || true
    
    log "日志清理完成"
}

# 生成报告
generate_report() {
    local report_enabled="${REPORT_ENABLED:-true}"
    
    if [[ "${report_enabled}" != "true" ]]; then
        return 0
    fi
    
    log "生成同步报告..."
    
    local report_file="${LOG_DIR}/report_$(date +%Y%m%d).md"
    
    cat > "${report_file}" << EOF
# 文档自动化同步报告
## 同步时间: $(date '+%Y-%m-%d %H:%M:%S')

## 同步状态
- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')
- 日志文件: ${LOG_FILE}

## 同步任务
EOF
    
    # 添加任务状态
    echo "- TAPD同步: ${TAPD_ENABLED:-false}" >> "${report_file}"
    echo "- Git同步: ${GIT_ENABLED:-true}" >> "${report_file}"
    
    # 添加系统信息
    cat >> "${report_file}" << EOF

## 系统信息
- 工作目录: ${WORKSPACE_DIR}
- 可用磁盘空间: $(df -h "${WORKSPACE_DIR}" | tail -1 | awk '{print $4}')
- Git状态: $(cd "${WORKSPACE_DIR}" && git status --short | wc -l) 个更改

## 错误日志
\`\`\`
$(tail -20 "${LOG_FILE}" 2>/dev/null || echo "无错误")
\`\`\`
EOF
    
    log "报告已生成: ${report_file}"
}

# 主函数
main() {
    log "开始文档自动化同步"
    log "工作目录: ${WORKSPACE_DIR}"
    log "日志文件: ${LOG_FILE}"
    
    # 加载配置
    load_config
    
    # 检查依赖
    check_dependencies
    
    # 执行同步任务
    sync_tapd
    sync_git
    
    # 清理和报告
    cleanup_logs
    generate_report
    
    log "文档自动化同步完成"
    
    # 发送通知（如果配置）
    if [[ "${NOTIFY_ENABLED:-false}" == "true" ]]; then
        send_notification
    fi
}

# 发送通知
send_notification() {
    # 这里可以实现邮件、Slack等通知
    log "通知功能未实现"
}

# 执行主函数
main "$@"