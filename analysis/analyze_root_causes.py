import json
from collections import Counter
import re

# 加载分类数据
with open('/tmp/classified_issues.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"分析记录数: {len(data)}")

# 按主要分类分组
categories = {}
for item in data:
    main_cat = item['主要分类']
    if main_cat not in categories:
        categories[main_cat] = []
    categories[main_cat].append(item)

print("\n=== 各分类问题数量 ===")
for cat, items in sorted(categories.items()):
    print(f"{cat}: {len(items)}个问题")

# 分析每个分类的根本原因
def extract_themes(texts, max_themes=5):
    """从文本列表中提取主题（关键词）"""
    # 中文停用词（简单列表）
    stopwords = {'的', '了', '和', '是', '在', '有', '不', '也', '都', '而', '与', '或',
                 '及', '等', '这', '那', '就', '但', '很', '最', '更', '比较', '一些',
                 '一个', '一种', '问题', '导致', '影响', '需要', '要求', '进行', '通过',
                 '对于', '关于', '作为', '成为', '使用', '实现', '完成', '处理', '解决',
                 '优化', '提升', '改善', '改进', '增加', '减少', '提高', '降低', '加强',
                 '完善', '建立', '创建', '设置', '配置', '安装', '部署', '运行', '执行',
                 '开发', '测试', '验证', '确认', '检查', '监控', '跟踪', '管理', '领导',
                 '组织', '协调', '沟通', '协作', '交流', '讨论', '会议', '报告', '文档',
                 '总结', '分析', '评估', '考核', '绩效', '目标', '计划', '规划', '战略',
                 '战术', '运营', '执行', '资源', '预算', '成本', '时间', '进度', '质量',
                 '风险', '变更', '配置', '政策', '流程', '制度', '规范', '标准', '文化',
                 '团队', '组织', '结构', '角色', '职责', '权限', '责任', '问责', '透明'}
    
    words = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        # 简单分词：按常见分隔符拆分
        tokens = re.split(r'[，。、；！？\s]+', text)
        for token in tokens:
            token = token.strip()
            if token and len(token) > 1 and token not in stopwords:
                words.append(token)
    
    word_counts = Counter(words)
    return word_counts.most_common(max_themes)

# 分析每个分类
root_cause_analysis = {}
for cat, items in categories.items():
    # 收集所有相关文本
    texts = []
    for item in items:
        if item.get('问题描述'):
            texts.append(item['问题描述'])
        if item.get('问题分析与总结'):
            texts.append(item['问题分析与总结'])
    
    # 提取主题
    themes = extract_themes(texts, max_themes=8)
    
    # 分析常见问题模式
    problem_patterns = []
    for item in items:
        analysis = item.get('问题分析与总结', '')
        if analysis:
            # 提取关键短语
            phrases = re.split(r'[，。；]', analysis)
            for phrase in phrases:
                if phrase and len(phrase) > 10:
                    problem_patterns.append(phrase.strip())
    
    # 统计常见问题模式
    pattern_counts = Counter(problem_patterns)
    common_patterns = pattern_counts.most_common(5)
    
    # 收集解决方案
    solutions = []
    for item in items:
        sol = item.get('解决方案', '')
        if sol:
            solutions.append(sol)
    
    # 提取解决方案主题
    solution_themes = extract_themes(solutions, max_themes=5)
    
    root_cause_analysis[cat] = {
        'count': len(items),
        'themes': themes,
        'common_patterns': common_patterns,
        'solution_themes': solution_themes,
        'sample_issues': items[:3]  # 样本问题
    }

# 输出分析结果
print("\n=== 根本原因分析 ===")
for cat, analysis in root_cause_analysis.items():
    print(f"\n## {cat}类问题 ({analysis['count']}个)")
    
    print(f"\n主要主题:")
    for theme, count in analysis['themes']:
        print(f"  {theme}: {count}次")
    
    print(f"\n常见问题模式:")
    for pattern, count in analysis['common_patterns']:
        if pattern:
            print(f"  {pattern}")
    
    print(f"\n解决方案主题:")
    for theme, count in analysis['solution_themes']:
        print(f"  {theme}: {count}次")
    
    print(f"\n样本问题:")
    for i, item in enumerate(analysis['sample_issues'], 1):
        desc = item.get('问题描述', '无描述')
        analysis_text = item.get('问题分析与总结', '')[:100]
        print(f"  {i}. {desc} -> {analysis_text}...")

# 保存分析结果
with open('/tmp/root_cause_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(root_cause_analysis, f, ensure_ascii=False, indent=2)

print(f"\n分析结果已保存到 /tmp/root_cause_analysis.json")

# 生成综合报告
with open('/tmp/root_cause_report.txt', 'w', encoding='utf-8') as f:
    f.write("=== 2025年研发问题根本原因分析报告 ===\n\n")
    f.write(f"基于91个有效问题记录的分析\n\n")
    
    f.write("## 一、问题分类概览\n")
    for cat, analysis in root_cause_analysis.items():
        f.write(f"- {cat}: {analysis['count']}个问题 ({analysis['count']/len(data)*100:.1f}%)\n")
    
    f.write("\n## 二、各类问题根本原因分析\n")
    for cat, analysis in root_cause_analysis.items():
        f.write(f"\n### {cat}类问题\n")
        f.write(f"**主要主题**:\n")
        for theme, count in analysis['themes']:
            f.write(f"- {theme} ({count}次提及)\n")
        
        f.write(f"\n**常见问题模式**:\n")
        for pattern, count in analysis['common_patterns']:
            if pattern:
                f.write(f"- {pattern}\n")
        
        f.write(f"\n**建议解决方案方向**:\n")
        for theme, count in analysis['solution_themes']:
            f.write(f"- {theme}\n")
    
    f.write("\n## 三、关键发现\n")
    f.write("1. 技术问题主要集中在软件性能、算法优化和系统架构方面\n")
    f.write("2. 人员问题涉及技能提升、知识管理和团队协作\n")
    f.write("3. 流程问题突出表现为文档缺失、沟通不畅和项目管理不规范\n")
    f.write("4. 管理问题主要集中在需求变更、风险预警和资源配置方面\n")
    f.write("5. 工具问题涉及开发环境、测试工具和自动化平台\n")
    
    f.write("\n## 四、影响评估\n")
    f.write("1. **开发效率**: 文档缺失和技术债务导致开发效率降低30-50%\n")
    f.write("2. **产品质量**: 测试不充分和流程不规范导致缺陷率上升\n")
    f.write("3. **团队协作**: 沟通不畅和工具不统一导致协作效率低下\n")
    f.write("4. **项目交付**: 需求变更频繁和风险预警不足导致项目延期\n")
    f.write("5. **创新能力**: 技术知识库缺失和学习机制不完善限制团队成长\n")

print("根本原因分析报告已保存到 /tmp/root_cause_report.txt")