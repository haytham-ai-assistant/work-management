#!/usr/bin/env python3
"""
TAPD API客户端
封装TAPD开放平台的API调用
"""

import requests
import hashlib
import time
import json
from typing import Dict, List, Optional, Any
import logging

# 本地导入配置
try:
    from config import TAPDConfig
except ImportError:
    # 回退到模拟配置
    class TAPDConfig:
        COMPANY_ID = ""
        API_KEY = ""
        API_SECRET = ""
        PROJECT_IDS = []
        BASE_URL = "https://api.tapd.cn"
        
        @staticmethod
        def validate_config():
            """模拟配置验证"""
            pass

logger = logging.getLogger(__name__)


class TAPDClient:
    """TAPD API客户端类"""
    
    def __init__(self, config: TAPDConfig = None):
        """初始化TAPD客户端"""
        self.config = config or TAPDConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Work-Management-TAPD-Client/1.0"
        })
        
        # 验证配置（如果配置类有validate_config方法）
        if hasattr(self.config, 'validate_config'):
            try:
                self.config.validate_config()
            except ValueError as e:
                logger.error(f"TAPD配置验证失败: {e}")
                raise
        else:
            logger.warning("TAPD配置验证跳过（配置类无validate_config方法）")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """生成API签名"""
        # TAPD API签名算法
        params_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        sign_str = f"{self.config.API_KEY}{params_str}{self.config.API_SECRET}"
        return hashlib.md5(sign_str.encode()).hexdigest()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, method: str = "GET") -> Dict[str, Any]:
        """执行API请求"""
        url = f"{self.config.BASE_URL}{endpoint}"
        
        # 基础参数
        base_params = {
            "company_id": self.config.COMPANY_ID,
            "timestamp": str(int(time.time()))
        }
        
        # 合并参数
        if params:
            base_params.update(params)
        
        # 生成签名
        base_params["sign"] = self._generate_signature(base_params)
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=base_params, timeout=30)
            else:
                response = self.session.post(url, json=base_params, timeout=30)
            
            response.raise_for_status()
            result = response.json()
            
            # 检查TAPD返回状态
            if result.get("status") != 1:
                error_msg = result.get("info", "未知错误")
                logger.error(f"TAPD API错误: {error_msg}")
                raise Exception(f"TAPD API错误: {error_msg}")
            
            return result.get("data", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TAPD API请求失败: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"TAPD API响应解析失败: {e}")
            raise
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """获取项目列表"""
        endpoint = "/projects"
        result = self._make_request(endpoint)
        return result.get("data", [])
    
    def get_requirements(self, project_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """获取项目需求列表"""
        endpoint = "/stories"
        params = {"workspace_id": project_id}
        if filters:
            params.update(filters)
        
        result = self._make_request(endpoint, params)
        return result.get("data", [])
    
    def get_defects(self, project_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """获取项目缺陷列表"""
        endpoint = "/bugs"
        params = {"workspace_id": project_id}
        if filters:
            params.update(filters)
        
        result = self._make_request(endpoint, params)
        return result.get("data", [])
    
    def get_documents(self, project_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取项目文档列表"""
        endpoint = "/wikis"
        params = {"workspace_id": project_id}
        if category:
            params["category"] = category
        
        result = self._make_request(endpoint, params)
        return result.get("data", [])
    
    def get_document_content(self, document_id: str) -> Dict[str, Any]:
        """获取文档内容"""
        endpoint = "/wikis/detail"
        params = {"id": document_id}
        
        result = self._make_request(endpoint, params)
        return result
    
    def create_requirement(self, project_id: str, requirement_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建需求"""
        endpoint = "/stories/add"
        params = {"workspace_id": project_id}
        params.update(requirement_data)
        
        result = self._make_request(endpoint, params, method="POST")
        return result
    
    def update_requirement(self, requirement_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新需求"""
        endpoint = "/stories/update"
        params = {"id": requirement_id}
        params.update(update_data)
        
        result = self._make_request(endpoint, params, method="POST")
        return result


def test_connection():
    """测试TAPD连接"""
    try:
        client = TAPDClient()
        projects = client.get_projects()
        print(f"✅ TAPD连接成功，找到 {len(projects)} 个项目")
        
        for project in projects[:3]:  # 只显示前3个项目
            print(f"  - {project.get('Project', {}).get('name')} (ID: {project.get('Project', {}).get('id')})")
        
        return True
    except Exception as e:
        print(f"❌ TAPD连接失败: {e}")
        return False


if __name__ == "__main__":
    print("TAPD客户端测试")
    print("=" * 50)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 测试连接
    if test_connection():
        print("\n✅ TAPD客户端配置正确")
    else:
        print("\n❌ 请检查TAPD配置")
        print("\n配置步骤：")
        print("1. 复制 .env.example 为 .env")
        print("2. 填写TAPD凭证")
        print("3. 运行 python tapd_client.py 再次测试")