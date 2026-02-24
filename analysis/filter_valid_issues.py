import json

# 加载数据
with open('/tmp/issues_parsed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤有效记录：有"问题提出人"或"问题描述"的记录
valid_issues = []
for item in data:
    # 检查是否至少有一个关键字段有内容
    has_person = item.get('问题提出人') and item['问题提出人'].strip()
    has_desc = item.get('问题描述') and item['问题描述'].strip()
    has_analysis = item.get('问题分析与总结') and item['问题分析与总结'].strip()
    
    if has_person or has_desc or has_analysis:
        valid_issues.append(item)

print(f"原始记录数: {len(data)}")
print(f"有效记录数: {len(valid_issues)}")

# 保存有效记录
with open('/tmp/valid_issues.json', 'w', encoding='utf-8') as f:
    json.dump(valid_issues, f, ensure_ascii=False, indent=2)

print(f"有效记录已保存到 /tmp/valid_issues.json")

# 统计有效记录的基本信息
if valid_issues:
    print("\n有效记录字段统计:")
    for col in ['序号', '问题提出人', '岗位', '问题描述', '问题分析与总结', '解决方案']:
        present = sum(1 for item in valid_issues if item.get(col) and str(item[col]).strip())
        empty = sum(1 for item in valid_issues if item.get(col) == '' or (item.get(col) and not str(item[col]).strip()))
        missing = sum(1 for item in valid_issues if col not in item or item[col] is None)
        print(f"{col}: 有内容={present}, 空值={empty}, 缺失={missing}")