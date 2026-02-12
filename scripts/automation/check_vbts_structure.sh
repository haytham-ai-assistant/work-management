#!/bin/bash
# 视触技术要点目录结构检查脚本
# 检查学习框架的完整性和更新状态

set -euo pipefail

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/workspace/工作"
VBTS_DIR="${WORKSPACE_DIR}/data/视触技术要点"
LOG_DIR="${WORKSPACE_DIR}/logs/automation"
LOG_FILE="${LOG_DIR}/vbts_check_$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="${LOG_DIR}/vbts_report_$(date +%Y%m%d).md"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_color() {
    local color="$1"
    local message="$2"
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}" | tee -a "${LOG_FILE}"
}

# 检查目录结构
check_directory_structure() {
    log_color "${BLUE}" "检查目录结构..."
    
    local required_dirs=(
        "papers"
        "concepts"
        "companies"
        "universities"
        "references"
        "applications"
        "market_analysis"
    )
    
    local missing_dirs=()
    local existing_dirs=()
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "${VBTS_DIR}/${dir}" ]]; then
            existing_dirs+=("${dir}")
            log "目录存在: ${dir}"
        else
            missing_dirs+=("${dir}")
            log_color "${YELLOW}" "警告: 目录不存在: ${dir}"
        fi
    done
    
    echo "${#existing_dirs[@]}" "${#missing_dirs[@]}" "${existing_dirs[*]}" "${missing_dirs[*]}"
}

# 检查文档文件
check_documents() {
    log_color "${BLUE}" "检查文档文件..."
    
    local required_files=(
        "references/公开技术资料综述.md"
        "references/开源项目列表.md"
        "universities/研究机构列表.md"
        "companies/相关公司列表.md"
        "market_analysis/产业链上下游分析.md"
        "技术学习路线图.md"
    )
    
    local missing_files=()
    local existing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ -f "${VBTS_DIR}/${file}" ]]; then
            existing_files+=("${file}")
            local line_count=$(wc -l < "${VBTS_DIR}/${file}" || echo "0")
            log "文件存在: ${file} (${line_count}行)"
        else
            missing_files+=("${file}")
            log_color "${YELLOW}" "警告: 文件不存在: ${file}"
        fi
    done
    
    echo "${#existing_files[@]}" "${#missing_files[@]}" "${existing_files[*]}" "${missing_files[*]}"
}

# 检查论文分析
check_paper_analysis() {
    log_color "${BLUE}" "检查论文分析..."
    
    local paper_files=(
        "papers/DTact_技术分析.md"
        "papers/DelTact_技术分析.md"
        "papers/DTactive_技术分析.md"
        "papers/Simulation_VBTS_技术分析.md"
    )
    
    local analyzed_count=0
    local total_count=4
    
    for paper in "${paper_files[@]}"; do
        if [[ -f "${VBTS_DIR}/${paper}" ]]; then
            analyzed_count=$((analyzed_count + 1))
            local size_kb=$(du -k "${VBTS_DIR}/${paper}" | cut -f1)
            log "论文分析完成: $(basename "${paper}") (${size_kb}KB)"
        else
            log_color "${YELLOW}" "警告: 论文分析缺失: $(basename "${paper}")"
        fi
    done
    
    echo "${analyzed_count}" "${total_count}"
}

# 检查PDF状态
check_pdf_status() {
    log_color "${BLUE}" "检查PDF状态..."
    
    local pdf_file="${WORKSPACE_DIR}/data/视触技术要点.pdf"
    local png_file="${WORKSPACE_DIR}/data/视触技术要点.png"
    
    if [[ -f "${pdf_file}" ]]; then
        local pdf_size=$(du -h "${pdf_file}" | cut -f1)
        log "PDF文件存在: 视触技术要点.pdf (${pdf_size})"
        echo "pdf_exists"
    else
        log_color "${RED}" "错误: PDF文件不存在"
        echo "pdf_missing"
    fi
    
    if [[ -f "${png_file}" ]]; then
        local png_size=$(du -h "${png_file}" | cut -f1)
        log "PNG文件存在: 视触技术要点.png (${png_size})"
        echo "png_exists"
    else
        log "PNG文件不存在"
        echo "png_missing"
    fi
}

# 检查Git状态
check_git_status() {
    log_color "${BLUE}" "检查Git状态..."
    
    cd "${WORKSPACE_DIR}"
    
    # 检查是否有未提交的更改
    local uncommitted_changes=$(git status --porcelain | wc -l)
    local vbts_changes=$(git status --porcelain | grep "data/视触技术要点" | wc -l)
    
    if [[ "${uncommitted_changes}" -gt 0 ]]; then
        log_color "${YELLOW}" "有未提交的更改: ${uncommitted_changes}个文件"
        if [[ "${vbts_changes}" -gt 0 ]]; then
            log "其中视触技术要点相关: ${vbts_changes}个文件"
        fi
    else
        log "所有更改已提交"
    fi
    
    # 检查最近提交
    local last_commit=$(git log -1 --oneline --format="%h %s" 2>/dev/null || echo "无提交记录")
    log "最近提交: ${last_commit}"
    
    echo "${uncommitted_changes}" "${vbts_changes}" "${last_commit}"
}

# 生成报告
generate_report() {
    log_color "${BLUE}" "生成检查报告..."
    
    # 获取检查结果
    local dir_result=$(check_directory_structure)
    local dir_existing=$(echo "${dir_result}" | awk '{print $1}')
    local dir_missing=$(echo "${dir_result}" | awk '{print $2}')
    local dir_existing_list=$(echo "${dir_result}" | awk '{for(i=3;i<=NF-1;i++) printf $i" "; print $NF}')
    local dir_missing_list=$(echo "${dir_result}" | awk '{for(i=NF;i>=1;i--) if(i>2+dir_existing+dir_missing) printf $i" "; print ""}' dir_existing="${dir_existing}" dir_missing="${dir_missing}")
    
    local doc_result=$(check_documents)
    local doc_existing=$(echo "${doc_result}" | awk '{print $1}')
    local doc_missing=$(echo "${doc_result}" | awk '{print $2}')
    
    local paper_result=$(check_paper_analysis)
    local paper_analyzed=$(echo "${paper_result}" | awk '{print $1}')
    local paper_total=$(echo "${paper_result}" | awk '{print $2}')
    
    local pdf_status=$(check_pdf_status)
    local git_result=$(check_git_status)
    local git_changes=$(echo "${git_result}" | awk '{print $1}')
    local vbts_changes=$(echo "${git_result}" | awk '{print $2}')
    
    # 生成报告文件
    cat > "${REPORT_FILE}" << EOF
# 视触技术要点学习框架检查报告
## 检查时间: $(date '+%Y-%m-%d %H:%M:%S')

## 总体状态
- **完整性评分**: $(( (dir_existing * 100 / 7 + doc_existing * 100 / 6 + paper_analyzed * 100 / 4) / 3 ))%
- **检查时间**: $(date '+%Y-%m-%d %H:%M:%S')
- **报告文件**: ${REPORT_FILE}

## 目录结构检查 (${dir_existing}/7)
**存在的目录**: ${dir_existing_list}
**缺失的目录**: ${dir_missing_list}

## 文档文件检查 (${doc_existing}/6)
- 技术资料综述: $( [[ -f "${VBTS_DIR}/references/公开技术资料综述.md" ]] && echo "✓" || echo "✗" )
- 开源项目列表: $( [[ -f "${VBTS_DIR}/references/开源项目列表.md" ]] && echo "✓" || echo "✗" )
- 研究机构列表: $( [[ -f "${VBTS_DIR}/universities/研究机构列表.md" ]] && echo "✓" || echo "✗" )
- 相关公司列表: $( [[ -f "${VBTS_DIR}/companies/相关公司列表.md" ]] && echo "✓" || echo "✗" )
- 产业链分析: $( [[ -f "${VBTS_DIR}/market_analysis/产业链上下游分析.md" ]] && echo "✓" || echo "✗" )
- 学习路线图: $( [[ -f "${VBTS_DIR}/技术学习路线图.md" ]] && echo "✓" || echo "✗" )

## 论文分析检查 (${paper_analyzed}/${paper_total})
- DTact技术分析: $( [[ -f "${VBTS_DIR}/papers/DTact_技术分析.md" ]] && echo "✓" || echo "✗" )
- DelTact技术分析: $( [[ -f "${VBTS_DIR}/papers/DelTact_技术分析.md" ]] && echo "✓" || echo "✗" )
- DTactive技术分析: $( [[ -f "${VBTS_DIR}/papers/DTactive_技术分析.md" ]] && echo "✓" || echo "✗" )
- Simulation_VBTS技术分析: $( [[ -f "${VBTS_DIR}/papers/Simulation_VBTS_技术分析.md" ]] && echo "✓" || echo "✗" )

## PDF状态检查
- PDF文件: $( [[ -f "${WORKSPACE_DIR}/data/视触技术要点.pdf" ]] && echo "存在" || echo "缺失" )
- PNG文件: $( [[ -f "${WORKSPACE_DIR}/data/视触技术要点.png" ]] && echo "存在" || echo "缺失" )

## Git状态检查
- 未提交更改: ${git_changes}个文件
- 视触相关更改: ${vbts_changes}个文件
- 最近提交: $(echo "${git_result}" | awk '{print $3, $4, $5}')

## 问题与建议

### 1. 紧急问题
$(if [[ "${dir_missing}" -gt 0 ]]; then
  echo "- 目录缺失: 需要创建 ${dir_missing} 个目录"
fi
if [[ "${doc_missing}" -gt 0 ]]; then
  echo "- 文档缺失: 需要补充 ${doc_missing} 个文档"
fi
if [[ "${paper_analyzed}" -lt "${paper_total}" ]]; then
  echo "- 论文分析不完整: 仅完成 ${paper_analyzed}/${paper_total}"
fi)

### 2. 改进建议
1. **完善文档**: 补充缺失的文档和目录
2. **PDF处理**: 解决PDF读取问题，获取详细技术要点
3. **定期更新**: 建立定期更新机制
4. **质量检查**: 定期运行此检查脚本

### 3. 下一步行动
1. 解决PDF内容读取问题
2. 补充缺失的文档和目录
3. 定期同步到GitHub仓库
4. 扩展学习框架内容

## 详细日志
\`\`\`
$(tail -30 "${LOG_FILE}" 2>/dev/null || echo "无详细日志")
\`\`\`

---

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**检查脚本**: check_vbts_structure.sh
**建议运行频率**: 每周一次
**自动修复**: 部分问题可自动修复，参见修复脚本
EOF
    
    log_color "${GREEN}" "报告已生成: ${REPORT_FILE}"
}

# 显示总结
show_summary() {
    log_color "${BLUE}" "检查完成，生成总结..."
    
    local dir_result=$(check_directory_structure)
    local dir_existing=$(echo "${dir_result}" | awk '{print $1}')
    
    local doc_result=$(check_documents)
    local doc_existing=$(echo "${doc_result}" | awk '{print $1}')
    
    local paper_result=$(check_paper_analysis)
    local paper_analyzed=$(echo "${paper_result}" | awk '{print $1}')
    
    local pdf_status=$(check_pdf_status | head -1)
    local git_result=$(check_git_status)
    local git_changes=$(echo "${git_result}" | awk '{print $1}')
    
    local score=$(( (dir_existing * 100 / 7 + doc_existing * 100 / 6 + paper_analyzed * 100 / 4) / 3 ))
    
    echo ""
    echo "========================================="
    echo "  视触技术要点学习框架检查总结"
    echo "========================================="
    echo ""
    echo "📊 完整性评分: ${score}%"
    echo ""
    echo "📁 目录结构: ${dir_existing}/7 个目录"
    echo "📄 文档文件: ${doc_existing}/6 个文档"
    echo "📝 论文分析: ${paper_analyzed}/4 篇论文"
    echo ""
    if [[ "${pdf_status}" == "pdf_exists" ]]; then
        echo "📎 PDF状态: 文件存在（需要处理）"
    else
        echo "📎 PDF状态: 文件缺失"
    fi
    echo ""
    if [[ "${git_changes}" -gt 0 ]]; then
        echo "🔄 Git状态: ${git_changes} 个未提交更改"
    else
        echo "🔄 Git状态: 所有更改已提交"
    fi
    echo ""
    echo "📋 详细报告: ${REPORT_FILE}"
    echo "📝 详细日志: ${LOG_FILE}"
    echo ""
    echo "========================================="
    
    if [[ "${score}" -lt 80 ]]; then
        log_color "${YELLOW}" "警告: 学习框架不完整，建议尽快完善"
    else
        log_color "${GREEN}" "良好: 学习框架基本完整"
    fi
}

# 主函数
main() {
    log_color "${GREEN}" "开始视触技术要点学习框架检查"
    log "工作目录: ${WORKSPACE_DIR}"
    log "检查目录: ${VBTS_DIR}"
    log "日志文件: ${LOG_FILE}"
    
    # 执行检查
    check_directory_structure > /dev/null
    check_documents > /dev/null
    check_paper_analysis > /dev/null
    check_pdf_status > /dev/null
    check_git_status > /dev/null
    
    # 生成报告
    generate_report
    
    # 显示总结
    show_summary
    
    log_color "${GREEN}" "视触技术要点学习框架检查完成"
}

# 执行主函数
main "$@"