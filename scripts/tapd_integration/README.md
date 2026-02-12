# TAPD集成脚本

## 概述
此目录包含TAPD（腾讯敏捷产品开发平台）与工作区文档管理系统的集成脚本。

## 功能特性
1. **需求同步** - 将TAPD需求同步到本地文档
2. **缺陷跟踪** - 同步缺陷状态和工作项
3. **文档管理** - 保持TAPD文档与本地版本一致
4. **自动化报告** - 生成项目状态报告

## 配置要求
### 必需凭证
1. TAPD公司ID
2. API访问密钥
3. 项目ID列表

### 环境配置
```bash
# 安装依赖
pip install requests python-dotenv

# 配置环境变量
cp .env.example .env
# 编辑.env文件填写TAPD凭证
```

## 使用方法
```bash
# 同步需求文档
python sync_requirements.py

# 同步缺陷跟踪
python sync_defects.py

# 生成项目报告
python generate_report.py
```

## 文件说明
- `config.py` - 配置文件模板
- `tapd_client.py` - TAPD API客户端
- `sync_requirements.py` - 需求同步脚本
- `sync_defects.py` - 缺陷同步脚本
- `generate_report.py` - 报告生成脚本
- `utils/` - 工具函数目录

## 安全注意事项
- 不要将`.env`文件提交到Git仓库
- 在`.gitignore`中添加`.env`文件
- 使用环境变量存储敏感信息