#!/usr/bin/env python3
"""
更新视触传感器市场报告脚本
基于数据准确性评估更新报告内容
"""

import os

def read_report():
    """读取原始报告"""
    with open('视触传感器市场与技术分析报告.md', 'r', encoding='utf-8') as f:
        return f.read()

def update_report(content):
    """更新报告内容"""
    
    # 1. 更新报告版本和日期
    content = content.replace('报告版本：v1.0', '报告版本：v1.1')
    content = content.replace('编制日期：2026年2月24日', '编制日期：2026年2月27日')
    content = content.replace('更新计划：每半年更新一次', '更新计划：每季度更新一次')
    
    # 2. 更新执行摘要 - 添加人形机器人驱动
    if '核心发现' in content:
        # 在核心发现中添加第六条
        core_findings_pos = content.find('### 核心发现')
        if core_findings_pos != -1:
            end_pos = content.find('### 报告结构', core_findings_pos)
            if end_pos != -1:
                current_findings = content[core_findings_pos:end_pos]
                if '6.' not in current_findings:
                    # 添加第六条
                    updated_findings = current_findings.replace(
                        '5. **进入门槛较高**：技术、制造、认证等多重壁垒构成市场进入挑战',
                        '5. **进入门槛较高**：技术、制造、认证等多重壁垒构成市场进入挑战\n6. **人形机器人驱动**：2025年人形机器人热潮成为市场新增长引擎，资本密集涌入'
                    )
                    content = content[:core_findings_pos] + updated_findings + content[end_pos:]
    
    # 3. 更新市场数据 - 添加整体触觉传感器市场最新数据
    market_section = '### 1.1 全球市场总体规模'
    if market_section in content:
        # 在广义触觉传感器市场后添加SNS Insider数据
        insert_point = content.find('#### 1.1.3 中国市场分析')
        if insert_point != -1:
            new_data = """
#### 1.1.4 整体触觉传感器市场（最新数据）
根据SNS Insider（2026年2月）报告：
- **2025年市场规模**：144.7亿美元（USD 14.47 billion）
- **2035年预测规模**：475.3亿美元（USD 47.53 billion）
- **复合年增长率(CAGR)**：12.63%
- **主要驱动因素**：自动化与机器人技术普及，智能穿戴设备需求增长，医疗应用扩展

#### 1.1.5 机器人专用视触传感器市场
根据QY Research（2025年10月）专项报告：
- **2024年市场规模**：1.01亿美元（机器人专用）
- **2031年预测规模**：2.45亿美元
- **复合年增长率(CAGR)**：13.5%
- **市场特点**：人形机器人成为主要增长驱动力，灵巧手触觉感知需求迫切
"""
            content = content[:insert_point] + new_data + content[insert_point:]
    
    # 4. 更新地区分布 - 调整亚太份额
    content = content.replace('| **亚太** | 30% |', '| **亚太** | 35% |')
    
    # 5. 更新增长驱动因素 - 添加人形机器人驱动
    demand_drivers = '#### 1.3.2 需求驱动'
    if demand_drivers in content:
        # 在需求驱动中添加第五条
        pos = content.find(demand_drivers)
        end_pos = content.find('#### 1.3.3 政策驱动', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            if '5.' not in current:
                updated = current.replace(
                    '4. **劳动力成本上升**：全球范围内自动化替代需求',
                    '4. **劳动力成本上升**：全球范围内自动化替代需求\n5. **人形机器人热潮**：2025年人形机器人产业化加速，触觉感知成为核心需求'
                )
                content = content[:pos] + updated + content[end_pos:]
    
    # 6. 更新GelSight信息
    gelsight_section = '#### 2.2.1 GelSight Inc.（美国）'
    if gelsight_section in content:
        pos = content.find(gelsight_section)
        end_pos = content.find('#### 2.2.2', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            # 更新技术优势
            if '2025年最新进展' not in current:
                updated = current.replace(
                    '- **技术优势**：',
                    '- **技术优势**：\n   4. 2025年最新进展：发布新凝胶技术（寿命延长40%），推出Modulus可互换镜头系统'
                )
                content = content[:pos] + updated + content[end_pos:]
    
    # 7. 更新Pressure Profile Systems信息
    pps_section = '#### 2.2.2 Pressure Profile Systems（美国）'
    if pps_section in content:
        pos = content.find(pps_section)
        end_pos = content.find('#### 2.2.3', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            if '2025年动态' not in current:
                updated = current.replace(
                    '- **应用领域**：医疗康复、机器人技术、人机交互',
                    '- **应用领域**：医疗康复、机器人技术、人机交互\n- **2025年动态**：在RobotWorld 2025展示RoboTact机器人触觉解决方案'
                )
                content = content[:pos] + updated + content[end_pos:]
    
    # 8. 更新中国本土企业分析 - 添加新章节
    china_section = '#### 2.3.3 其他中国相关企业'
    if china_section in content:
        pos = content.find(china_section)
        end_pos = content.find('### 2.4 产业链分析', pos)
        if end_pos != -1:
            # 添加新的章节
            new_china_content = """
#### 2.3.4 2025年中国触觉传感器行业十大潜力企业
根据中商产业研究院（2025年9月）排行榜，中国触觉传感器行业十大潜力企业为：

| 排名 | 企业简称 | 主要产品 | 核心竞争力 |
|------|----------|----------|------------|
| 1 | **能斯达电子** | 柔性触觉传感器、电子皮肤 | 柔性电子领域头部企业，与华为、小米合作 |
| 2 | **柯力传感** | 工业级触觉传感器 | 工业传感器龙头企业，深耕工业物联网20年 |
| 3 | **敏芯微电子** | MEMS触觉传感器 | MEMS全产业链国产化龙头，通过车规级认证 |
| 4 | **韦尔股份** | 车规级触觉传感器 | 半导体设计全球龙头，打破国外垄断 |
| 5 | **汇顶科技** | 触控反馈传感器 | 全球触控芯片龙头，向汽车、AR/VR扩展 |
| 6 | **帕西尼感知** | 多模态触觉传感器 | 中科院技术孵化，人形机器人领域市占率超20% |
| 7 | **他山科技** | 机器人触觉反馈传感器 | 中科院自动化所团队创业，多模态数据融合算法 |
| 8 | **华威科** | 微型化MEMS触觉传感器 | 华为哈勃投资企业，物联网领域核心供应商 |
| 9 | **歌尔股份** | 消费电子触觉反馈模组 | 全球消费电子ODM龙头，VR/AR触觉交互领先 |
| 10 | **墨现科技** | 医疗级触觉传感器 | 清华大学技术转化，手术机器人触觉探测 |

#### 2.3.5 2025年中国市场融资动态分析
根据36氪（2025年8月）报道，人形机器人带火视触觉传感器赛道，资本密集涌入：

1. **纬钛机器人**：小米旗下瀚星创投投资，完成近亿元天使及天使+轮融资
2. **戴盟机器人**：招商局创投领投亿元级天使++轮，累计融资数亿元
3. **帕西尼感知**：京东战略领投数亿元A++++轮，2025年内完成三次A系列融资
4. **一目科技**：完成数亿元D轮融资，定位AI感知领域"英伟达"
5. **千觉机器人**：高瓴投资，为智元机器人提供触觉感知解决方案

**市场特点**：技术路线百家争鸣（压阻、电容、压电、磁电霍尔效应、光学、摩擦电、视触觉），市场尚未形成绝对龙头，为创业公司提供了弯道超车机会。
"""
            content = content[:end_pos] + new_china_content + content[end_pos:]
    
    # 9. 更新学术研究热点
    research_section = '### 3.4 学术研究热点'
    if research_section in content:
        pos = content.find(research_section)
        end_pos = content.find('---', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            
            # 更新国际研究团队
            if 'ThinTact' not in current:
                updated = current.replace(
                    '3. **斯坦福大学**：柔性电子与触觉传感器研究',
                    '3. **斯坦福大学**：柔性电子与触觉传感器研究\n4. **最新研究突破（2025年）**：\n   - ThinTact：无透镜成像视触传感器，厚度<10mm（arXiv 2025.01）\n   - VBTS分类综述：系统分类标记基与强度基传感原理（arXiv 2025.09）\n   - Taccel仿真平台：高性能GPU仿真，支持4096并行环境（NeurIPS 2025）'
                )
                content = content[:pos] + updated + content[end_pos:]
            
            # 更新中国研究团队
            if '丁文伯团队' not in current:
                # 在原有中国研究团队后添加新内容
                china_research_pos = content.find('#### 3.4.2 中国研究团队')
                if china_research_pos != -1 and china_research_pos < end_pos:
                    china_end = content.find('---', china_research_pos)
                    current_china = content[china_research_pos:china_end]
                    updated_china = current_china.replace(
                        '4. **哈尔滨工业大学**：机器人技术与系统国家重点实验室',
                        '4. **哈尔滨工业大学**：机器人技术与系统国家重点实验室\n\n#### 3.4.3 2025-2026年最新研究成果\n1. **清华大学丁文伯团队**（2026年1月）：在《Nature Sensors》发表SuperTac多模态触觉传感器，实现微米级分辨率，多模态感知精度超过94%\n2. **上海交通大学张文明团队**（2025年5月）：在《Applied Physics Reviews》发表视触传感器综述，系统分析性能参数与器件设计\n3. **技术趋势**：多模态融合（力、位置、温度、接近、振动）、仿生设计（鸽眼启发的多光谱成像）、触觉语言模型（8.5亿参数DOVE模型）'
                    )
                    content = content[:china_research_pos] + updated_china + content[china_end:]
    
    # 10. 更新技术性能指标
    performance_section = '#### 3.2.1 性能指标'
    if performance_section in content:
        pos = content.find(performance_section)
        end_pos = content.find('#### 3.2.2', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            # 更新空间分辨率领先水平
            updated = current.replace(
                '| **空间分辨率** | 0.5-1mm | 0.1mm | 光学系统限制 |',
                '| **空间分辨率** | 0.5-1mm | **微米级**（清华大学SuperTac） | 光学系统限制 |'
            )
            # 添加多模态感知指标
            if '多模态感知能力' not in updated:
                updated = updated + '\n| **多模态感知能力** | 单模态（力） | 多模态（力、位置、温度、接近、振动） | 传感器融合算法 |'
            content = content[:pos] + updated + content[end_pos:]
    
    # 11. 更新技术发展趋势
    trend_section = '#### 3.3.2 硬件发展趋势'
    if trend_section in content:
        pos = content.find(trend_section)
        end_pos = content.find('### 3.4', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            # 添加新趋势
            if '仿生化' not in current:
                updated = current.replace(
                    '4. **无线化**：无线传输减少布线约束',
                    '4. **无线化**：无线传输减少布线约束\n5. **仿生化**：生物启发设计（如鸽眼多光谱成像）\n6. **AI集成**：触觉语言模型与AI大模型融合'
                )
                content = content[:pos] + updated + content[end_pos:]
    
    # 12. 添加人形机器人市场影响章节
    challenge_section = '## 第四章：市场进入挑战与对策'
    if challenge_section in content:
        pos = content.find(challenge_section)
        # 在第四章前插入新章节
        new_section = """
## 第三章补充：人形机器人对视触传感器市场的影响

### 3.5.1 市场规模测算
根据国泰海通证券测算，人形机器人对触觉传感器市场的拉动效应：

| 人形机器人产量 | 触觉传感器市场规模（多技术路线） | 视触传感器市场规模（单一技术路线） |
|----------------|----------------------------------|------------------------------------|
| **1000万台** | 0.24万亿元 | 0.8-1.2万亿元 |
| **1亿台** | 1.18万亿元 | **3.76万亿元** |

### 3.5.2 技术路线竞争格局
触觉传感技术路线多样，各具优势：
- **压阻式**：成本低、灵敏度高，市场份额约37%
- **电容式**：设计灵活、温度稳定性好，市场份额约28%
- **磁电霍尔效应**：抗干扰能力强，适用于恶劣环境
- **视触觉**：空间分辨率高、数据格式适配AI，被寄予厚望

### 3.5.3 产业发展阶段
1. **当前阶段**（2025年）：市场尚未形成绝对龙头，技术路线百家争鸣
2. **中期阶段**（2027-2030年）：2-3种主流技术路线确立，头部企业出现
3. **长期阶段**（2030年后）：市场集中度提升，生态体系完善

"""
        # 找到插入位置（在第三章结束，第四章开始前）
        chap3_end = content.rfind('---', 0, pos)
        if chap3_end != -1:
            content = content[:chap3_end] + new_section + content[chap3_end:]
    
    # 13. 更新数据来源
    sources_section = '### A. 数据来源说明'
    if sources_section in content:
        pos = content.find(sources_section)
        end_pos = content.find('### B.', pos)
        if end_pos != -1:
            current = content[pos:end_pos]
            updated = current.replace(
                '1. 市场数据：Emergen Research, Mordor Intelligence, QY Research, Fortune Business Insights',
                '1. 市场数据：Emergen Research, Mordor Intelligence, QY Research, SNS Insider (2026), 国泰海通证券\n2. 企业信息：公司官方网站、财务报告、行业分析报告、36氪(2025)、中商产业研究院(2025)\n3. 技术信息：学术论文（Nature Sensors 2026, Applied Physics Reviews 2025, arXiv）、专利文献、开源项目文档\n4. 验证数据：本项目实验验证结果'
            )
            content = content[:pos] + updated + content[end_pos:]
    
    return content

def main():
    """主函数"""
    print("正在更新视触传感器市场报告...")
    
    # 读取原始报告
    content = read_report()
    
    # 更新报告
    updated_content = update_report(content)
    
    # 保存更新后的报告
    output_file = '视触传感器市场与技术分析报告_v1.1_更新.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"报告更新完成，保存为: {output_file}")
    
    # 统计更新内容
    original_lines = content.count('\n')
    updated_lines = updated_content.count('\n')
    print(f"原始报告行数: {original_lines}")
    print(f"更新报告行数: {updated_lines}")
    print(f"新增内容: {updated_lines - original_lines} 行")

if __name__ == '__main__':
    main()