#!/usr/bin/env python3
"""
TAPD集成配置模板
请使用环境变量配置或直接修改此文件
"""

import os

class TAPDConfig:
    """TAPD配置类"""
    
    # TAPD API基础URL
    BASE_URL = "https://api.tapd.cn"
    
    # TAPD公司ID (从环境变量获取或直接设置)
    COMPANY_ID = os.getenv("TAPD_COMPANY_ID", "your_company_id")
    
    # API访问密钥 (从环境变量获取)
    API_KEY = os.getenv("TAPD_API_KEY", "your_api_key")
    
    # API密钥 (从环境变量获取)
    API_SECRET = os.getenv("TAPD_API_SECRET", "your_api_secret")
    
    # 项目ID列表 (逗号分隔)
    PROJECT_IDS = os.getenv("TAPD_PROJECT_IDS", "").split(",") if os.getenv("TAPD_PROJECT_IDS") else []
    
    # 同步配置
    SYNC_CONFIG = {
        # 需求同步设置
        "requirements": {
            "enabled": True,
            "sync_interval_hours": 24,  # 同步间隔（小时）
            "output_dir": "/workspace/工作/docs/projects/requirements/tapd",
            "include_fields": ["id", "name", "description", "status", "priority", "creator", "created", "modified"]
        },
        
        # 缺陷同步设置
        "defects": {
            "enabled": True,
            "sync_interval_hours": 12,
            "output_dir": "/workspace/工作/docs/projects/defects/tapd",
            "include_fields": ["id", "title", "description", "status", "severity", "priority", "reporter", "created", "modified"]
        },
        
        # 文档同步设置
        "documents": {
            "enabled": True,
            "sync_interval_hours": 24,
            "output_dir": "/workspace/工作/docs/references/tapd",
            "include_categories": ["需求文档", "设计文档", "测试文档", "用户手册"]
        }
    }
    
    # 日志配置
    LOG_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "/workspace/工作/scripts/tapd_integration/logs/tapd_sync.log"
    }
    
    @classmethod
    def validate_config(cls):
        """验证配置是否完整"""
        missing = []
        
        if not cls.COMPANY_ID or cls.COMPANY_ID == "your_company_id":
            missing.append("TAPD_COMPANY_ID")
        
        if not cls.API_KEY or cls.API_KEY == "your_api_key":
            missing.append("TAPD_API_KEY")
        
        if not cls.API_SECRET or cls.API_SECRET == "your_api_secret":
            missing.append("TAPD_API_SECRET")
        
        if not cls.PROJECT_IDS:
            missing.append("TAPD_PROJECT_IDS")
        
        if missing:
            raise ValueError(f"缺少必要的TAPD配置: {', '.join(missing)}")
        
        # 创建输出目录
        for sync_type in cls.SYNC_CONFIG.values():
            if sync_type["enabled"]:
                os.makedirs(sync_type["output_dir"], exist_ok=True)
        
        # 创建日志目录
        log_dir = os.path.dirname(cls.LOG_CONFIG["file"])
        os.makedirs(log_dir, exist_ok=True)
        
        return True

# 环境变量说明
ENV_VARS = """
# TAPD集成环境变量配置
TAPD_COMPANY_ID=your_company_id_here
TAPD_API_KEY=your_api_key_here
TAPD_API_SECRET=your_api_secret_here
TAPD_PROJECT_IDS=project_id_1,project_id_2

# 可选：代理设置
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=https://proxy.example.com:8080
"""

if __name__ == "__main__":
    print("TAPD配置模块")
    print("请配置以下环境变量：")
    print(ENV_VARS)