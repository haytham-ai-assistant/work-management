#!/usr/bin/env python3
"""
TAPD需求同步脚本
将TAPD需求同步到本地文档
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tapd_client import TAPDClient
from config import TAPDConfig


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(TAPDConfig.LOG_CONFIG["file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_requirements_to_markdown(requirements: List[Dict[str, Any]], output_dir: str, project_name: str):
    """保存需求到Markdown文件"""
    
    # 按状态分类
    requirements_by_status = {}
    for req in requirements:
        status = req.get("status", "未知")
        if status not in requirements_by_status:
            requirements_by_status[status] = []
        requirements_by_status[status].append(req)
    
    # 生成主README
    readme_content = f"""# {project_name} - 需求文档

## 概述
此文档包含从TAPD同步的需求列表。

## 统计信息
- 总需求数: {len(requirements)}
- 同步时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 需求状态分布
"""
    
    for status, req_list in requirements_by_status.items():
        readme_content += f"- {status}: {len(req_list)} 个需求\n"
    
    readme_content += "\n## 需求列表\n"
    
    # 按ID排序
    requirements.sort(key=lambda x: x.get("id", ""))
    
    for req in requirements:
        req_id = req.get("id", "未知")
        name = req.get("name", "未命名")
        status = req.get("status", "未知")
        priority = req.get("priority", "未设置")
        creator = req.get("creator", "未知")
        created = req.get("created", "未知")
        
        readme_content += f"\n### {req_id} - {name}\n\n"
        readme_content += f"- **状态**: {status}\n"
        readme_content += f"- **优先级**: {priority}\n"
        readme_content += f"- **创建者**: {creator}\n"
        readme_content += f"- **创建时间**: {created}\n"
        
        description = req.get("description", "")
        if description:
            readme_content += f"\n**描述**:\n{description}\n"
    
    # 保存README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # 保存原始JSON数据
    json_path = os.path.join(output_dir, "requirements.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(requirements, f, ensure_ascii=False, indent=2)
    
    return readme_path, json_path


def sync_project_requirements(client: TAPDClient, project_id: str, output_dir: str):
    """同步单个项目的需求"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"开始同步项目 {project_id} 的需求...")
        
        # 获取需求列表
        requirements = client.get_requirements(project_id)
        logger.info(f"获取到 {len(requirements)} 个需求")
        
        # 创建项目输出目录
        project_output_dir = os.path.join(output_dir, f"project_{project_id}")
        os.makedirs(project_output_dir, exist_ok=True)
        
        # 保存需求文档
        readme_path, json_path = save_requirements_to_markdown(
            requirements, project_output_dir, f"项目 {project_id}"
        )
        
        logger.info(f"需求文档已保存: {readme_path}")
        logger.info(f"原始数据已保存: {json_path}")
        
        return len(requirements)
        
    except Exception as e:
        logger.error(f"同步项目 {project_id} 需求失败: {e}")
        return 0


def main():
    """主函数"""
    logger = setup_logging()
    
    try:
        logger.info("开始TAPD需求同步")
        
        # 初始化客户端
        client = TAPDClient()
        
        # 检查配置
        if not TAPDConfig.PROJECT_IDS:
            logger.error("未配置TAPD项目ID")
            return 1
        
        # 获取输出目录
        output_dir = TAPDConfig.SYNC_CONFIG["requirements"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # 同步每个项目
        total_requirements = 0
        for project_id in TAPDConfig.PROJECT_IDS:
            if project_id.strip():
                count = sync_project_requirements(client, project_id.strip(), output_dir)
                total_requirements += count
        
        logger.info(f"需求同步完成，共处理 {total_requirements} 个需求")
        
        # 生成汇总报告
        summary = {
            "sync_time": datetime.now().isoformat(),
            "total_requirements": total_requirements,
            "projects": TAPDConfig.PROJECT_IDS
        }
        
        summary_path = os.path.join(output_dir, "sync_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"同步摘要已保存: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"需求同步失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)