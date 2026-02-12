# 文档自动化同步系统

## 概述
此系统提供工作区文档的自动化同步和监控功能，支持TAPD集成、Git版本控制和定期同步。

## 目录结构
```
scripts/
├── automation/           # 自动化脚本
│   ├── sync_all.sh      # 主同步脚本
│   ├── monitor.sh       # 监控脚本
│   ├── cron_setup.sh    # Cron配置脚本
│   └── README.md        # 本文档
├── config/              # 配置文件
│   └── automation.conf  # 自动化配置
└── tapd_integration/    # TAPD集成
    ├── config.py        # TAPD配置模板
    ├── tapd_client.py   # TAPD客户端
    ├── sync_requirements.py # 需求同步
    └── .env.example     # 环境变量示例
```

## 快速开始

### 1. 配置系统
```bash
# 进入脚本目录
cd /workspace/工作/scripts

# 设置执行权限
chmod +x automation/*.sh
chmod +x automation/*.py 2>/dev/null || true

# 复制配置文件（如果需要）
cp config/automation.conf.example config/automation.conf
```

### 2. 配置TAPD集成（可选）
```bash
cd tapd_integration

# 复制环境变量文件
cp .env.example .env

# 编辑.env文件，填写TAPD凭证
# TAPD_COMPANY_ID=your_company_id
# TAPD_API_KEY=your_api_key
# TAPD_API_SECRET=your_api_secret
# TAPD_PROJECT_IDS=project_id_1,project_id_2

# 启用TAPD同步
# 编辑 config/automation.conf，设置 TAPD_ENABLED=true
```

### 3. 手动测试同步
```bash
# 运行完整同步
./automation/sync_all.sh

# 运行监控检查
./automation/monitor.sh
```

### 4. 设置定时任务
```bash
# 方法A：使用系统cron（需要root）
sudo ./automation/cron_setup.sh

# 方法B：使用用户cron
crontab -e
# 添加：0 2 * * * /workspace/工作/scripts/automation/sync_all.sh
```

## 脚本说明

### sync_all.sh - 主同步脚本
**功能：**
- 同步TAPD需求数据（如果启用）
- 提交Git更改并推送到远程
- 生成同步报告
- 清理旧日志

**配置选项：** `config/automation.conf`

### monitor.sh - 监控脚本
**检查项：**
1. 磁盘空间使用情况
2. Git仓库状态
3. 日志文件状态
4. 同步任务状态
5. TAPD集成配置

**输出：** 彩色状态报告

### cron_setup.sh - Cron配置脚本
**功能：**
- 创建系统cron任务
- 设置日志轮转
- 提供用户cron示例

## 配置说明

### automation.conf 主要配置项
```bash
# 启用/禁用功能
TAPD_ENABLED=false
GIT_ENABLED=true

# 日志设置
LOG_KEEP_DAYS=7

# 通知设置（待实现）
NOTIFY_ENABLED=false
```

### TAPD配置
1. 在 `tapd_integration/.env` 中设置凭证
2. 在 `automation.conf` 中启用TAPD同步
3. 根据需要修改 `tapd_integration/config.py`

## 故障排除

### 常见问题

#### 1. 同步脚本权限错误
```bash
chmod +x /workspace/工作/scripts/automation/*.sh
```

#### 2. TAPD连接失败
- 检查 `.env` 文件中的凭证
- 验证TAPD API权限
- 检查网络连接

#### 3. Git推送失败
- 检查Git远程配置：`git remote -v`
- 验证GitHub权限
- 检查网络连接

#### 4. 磁盘空间不足
- 清理旧日志：`find /workspace/工作/logs -name "*.log" -mtime +7 -delete`
- 检查大文件：`du -sh /workspace/工作/* | sort -hr`

### 日志文件
- 同步日志：`/workspace/工作/logs/automation/sync_*.log`
- 监控报告：控制台输出
- Cron日志：`/var/log/work-management-sync.log`（系统cron）

## 扩展开发

### 添加新的同步源
1. 在 `sync_all.sh` 中添加新函数
2. 在 `automation.conf` 中添加配置项
3. 在 `monitor.sh` 中添加检查项

### 添加通知功能
1. 实现邮件通知（SMTP）
2. 集成Slack/Teams webhook
3. 添加企业微信/钉钉机器人

### 性能优化
1. 增量同步代替全量同步
2. 并行处理多个项目
3. 缓存已同步数据

## 维护建议
1. **定期检查**：每周运行 `monitor.sh`
2. **日志轮转**：配置logrotate管理日志
3. **备份配置**：定期备份 `.env` 和配置文件
4. **版本控制**：所有脚本和配置都应提交到Git

## 安全注意事项
- 不要将 `.env` 文件提交到Git
- 定期轮换API密钥
- 限制脚本执行权限
- 监控异常访问模式