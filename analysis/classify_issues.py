import json
import re
from collections import Counter

# 加载有效数据
with open('/tmp/valid_issues.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"有效记录数: {len(data)}")

# 定义分类关键词（可扩展）
category_keywords = {
    '技术': [
        '错误', 'bug', '异常', '性能', '优化', '算法', '代码', '功能', '软件', '系统',
        '数据', '通信', '相机', '刷新', '匹配', '识别', '处理', '访问', '并发', '负载',
        '均衡', '集群', '升级', '版本', '逻辑', '误差', '累积', '泊松', '疲劳', '引伸计',
        '固距', '双目', '多相机', '刷新', '卡顿', '接口', 'API', 'SDK', '框架', '库',
        '模块', '组件', '架构', '设计', '实现', '开发', '编程', '测试', '调试', '部署',
        '编译', '运行', '执行', '计算', '存储', '内存', 'CPU', '网络', '数据库', '缓存',
        '安全', '加密', '认证', '授权', '权限', '配置', '设置', '参数', '变量', '常量',
        '函数', '方法', '类', '对象', '实例', '继承', '多态', '封装', '抽象', '接口'
    ],
    '流程': [
        '流程', '文档', '管理', '协作', '沟通', '会议', '评审', '测试', '部署', '发布',
        '迭代', '周期', '计划', '跟踪', '监控', '报告', '反馈', '总结', '分析', '评估',
        '审核', '审批', '验收', '交付', '上线', '运维', '支持', '维护', '更新', '升级',
        '备份', '恢复', '迁移', '转换', '集成', '对接', '接口', '协议', '标准', '规范',
        '指南', '手册', '教程', '示例', '案例', '模板', '表单', '表格', '记录', '日志',
        '统计', '指标', '度量', '测量', '评估', '考核', '绩效', 'KPI', 'OKR', '目标'
    ],
    '人员': [
        '培训', '学习', '能力', '技能', '经验', '知识', '英语', '沟通', '协作', '团队',
        '个人', '成长', '发展', '招聘', '保留', '激励', '考核', '绩效', '晋升', '薪资',
        '福利', '待遇', '文化', '氛围', '环境', '关系', '冲突', '协调', '领导', '管理',
        '指导', '辅导', ' mentoring', ' coaching', '反馈', '评价', '评估', '考核',
        '晋升', '职业', '规划', '路径', '发展', '培训', '教育', '学习', '技能', '能力'
    ],
    '工具': [
        '工具', '平台', '系统', '软件', '硬件', '设备', '环境', '配置', '设置', '安装',
        '部署', '集成', '接口', 'API', 'SDK', '框架', '库', '插件', '扩展', '模块',
        '组件', '服务', '应用', '程序', '脚本', '命令行', '终端', '编辑器', 'IDE',
        '调试器', '分析器', '监控', '日志', '跟踪', '性能', '测试', '自动化', '手动',
        '云', '服务器', '网络', '存储', '数据库', '缓存', '消息', '队列', '流', '批处理'
    ],
    '管理': [
        '管理', '领导', '决策', '规划', '计划', '组织', '协调', '控制', '监督', '指导',
        '评估', '考核', '绩效', 'KPI', 'OKR', '目标', '战略', '战术', '运营', '执行',
        '资源', '预算', '成本', '时间', '进度', '质量', '风险', '问题', '变更', '配置',
        '沟通', '会议', '报告', '文档', '政策', '流程', '制度', '规范', '标准', '文化',
        '团队', '组织', '结构', '角色', '职责', '权限', '责任', '问责', '透明', '可视化'
    ]
}

# 将关键词转换为小写，便于匹配
category_keywords_lower = {}
for category, keywords in category_keywords.items():
    category_keywords_lower[category] = [kw.lower() for kw in keywords]

# 分类函数
def classify_issue(text):
    """对文本进行分类，返回分类列表（按匹配数排序）"""
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    scores = Counter()
    
    for category, keywords in category_keywords_lower.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[category] += 1
    
    # 按匹配数排序
    sorted_categories = [cat for cat, _ in scores.most_common()]
    return sorted_categories

# 对每个问题进行分类
classified_data = []
category_distribution = Counter()

for item in data:
    # 组合相关文本
    text_parts = []
    if item.get('问题描述'):
        text_parts.append(item['问题描述'])
    if item.get('问题分析与总结'):
        text_parts.append(item['问题分析与总结'])
    if item.get('解决方案'):
        text_parts.append(item['解决方案'])
    
    combined_text = ' '.join(text_parts)
    
    # 获取分类
    categories = classify_issue(combined_text)
    
    # 如果没有匹配到任何分类，使用"其他"
    if not categories:
        categories = ['其他']
    
    # 添加分类信息到条目
    item_with_categories = item.copy()
    item_with_categories['分类'] = categories
    item_with_categories['主要分类'] = categories[0] if categories else '其他'
    
    classified_data.append(item_with_categories)
    
    # 更新分布统计
    for cat in categories:
        category_distribution[cat] += 1
    category_distribution['主要_' + categories[0]] += 1

print("\n=== 分类分布 ===")
print("总分类次数（多标签）:")
for category, count in category_distribution.most_common():
    if not category.startswith('主要_'):
        print(f"  {category}: {count}")

print("\n主要分类分布:")
for category, count in sorted(category_distribution.items()):
    if category.startswith('主要_'):
        cat_name = category.replace('主要_', '')
        print(f"  {cat_name}: {count}")

# 保存分类结果
with open('/tmp/classified_issues.json', 'w', encoding='utf-8') as f:
    json.dump(classified_data, f, ensure_ascii=False, indent=2)

print(f"\n分类结果已保存到 /tmp/classified_issues.json")

# 生成分类摘要
category_summary = {}
for item in classified_data:
    main_cat = item['主要分类']
    if main_cat not in category_summary:
        category_summary[main_cat] = []
    category_summary[main_cat].append(item)

# 保存分类摘要
with open('/tmp/category_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== 问题分类摘要 ===\n\n")
    for category, items in sorted(category_summary.items()):
        f.write(f"## {category} ({len(items)}个问题)\n")
        for i, item in enumerate(items[:10], 1):  # 每个类别只显示前10个
            desc = item.get('问题描述', '')[:100]
            if not desc:
                desc = item.get('问题分析与总结', '')[:100]
            f.write(f"{i}. [{item.get('序号', 'N/A')}] {desc}...\n")
        if len(items) > 10:
            f.write(f"  ... 还有{len(items)-10}个问题\n")
        f.write("\n")

print("分类摘要已保存到 /tmp/category_summary.txt")