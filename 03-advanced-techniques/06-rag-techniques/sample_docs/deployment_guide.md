# TechFlow 部署指南

## CI/CD 流水线

TechFlow 使用 GitHub Actions 进行持续集成和部署。每次推送到功能分支都会触发 CI 流水线；合并到 `main` 会触发测试环境部署；生产环境部署需要手动批准。

### 流水线阶段

1. **代码检查与格式化** — 对所有 Python 服务运行 ruff 和 black；发现任何违规项都会导致构建失败。Node.js 服务使用 ESLint。
2. **单元测试** — 运行 pytest 并生成覆盖率报告。最低覆盖率要求为 80%，测试由 4 个 Worker 并行执行。
3. **集成测试** — 在 Docker 容器中启动 PostgreSQL、Redis 和 Kafka。针对完整服务堆栈运行端到端 API 测试。超时：15 分钟。
4. **安全扫描** — 使用 Snyk 检查依赖项漏洞，使用 Bandit 检查 Python 安全问题。严重漏洞会阻止构建。
5. **构建并推送** — 构建 Docker 镜像并推送到 AWS ECR。镜像以 Git SHA 为标签，主分支的镜像还会添加 `latest` 标签。
6. **部署到测试环境** — `main` 分支自动执行。更新 ECS 任务定义并触发滚动部署。
7. **部署到生产环境** — 需要团队负责人手动批准，采用蓝绿部署策略。

### 构建时间
- CI 流水线平均耗时：8 分钟
- 测试环境部署平均耗时：4 分钟
- 生产环境部署平均耗时：6 分钟（包括蓝绿切换）

## 环境配置

### 环境变量
每个服务都从环境变量中读取配置，这些配置由 AWS Systems Manager Parameter Store 管理。不同环境的值（测试环境与生产环境）存储在不同路径下：

```
/techflow/staging/auth-service/DATABASE_URL
/techflow/production/auth-service/DATABASE_URL
```

### 所需的环境变量（所有服务）
- `ENVIRONMENT` — `staging` 或 `production`
- `LOG_LEVEL` — `DEBUG`、`INFO`、`WARNING`、`ERROR`（默认：`INFO`）
- `KAFKA_BOOTSTRAP_SERVERS` — 以逗号分隔的 Kafka Broker 地址
- `REDIS_URL` — 带身份验证的 Redis 连接字符串
- `SENTRY_DSN` — 错误跟踪端点

### 服务专用变量
- **认证服务**：`DATABASE_URL`、`JWT_SECRET_KEY`、`OAUTH2_CLIENT_IDS`、`SESSION_TTL_SECONDS`
- **项目服务**：`DATABASE_URL`、`READ_REPLICA_URL`、`PGBOUNCER_MAX_CONNECTIONS`
- **通知服务**：`SES_REGION`、`FCM_CREDENTIALS`、`WEBHOOK_SIGNING_SECRET`
- **文件服务**：`S3_BUCKET`、`CLOUDFRONT_DOMAIN`、`MAX_UPLOAD_SIZE_MB`、`CLAMAV_HOST`

## 数据库迁移

数据库迁移通过 Alembic（Python 服务）进行管理，并在部署过程中自动应用。

### 迁移流程
1. 开发人员创建迁移：`alembic revision --autogenerate -m "add_labels_table"`
2. 迁移在 PR 中进行审核（所有迁移必须向后兼容）
3. 部署时，ECS 任务的初始化容器会在服务启动前运行 `alembic upgrade head`
4. 如果迁移失败，部署会自动回滚

### 迁移规则
- **始终保持向后兼容**：滚动部署期间，旧代码必须能够使用新数据库 schema
- **同一版本中不删除列**：第一个版本删除代码引用，第二个版本删除列
- **并发添加索引**：使用 `CREATE INDEX CONCURRENTLY`，避免锁表
- **大数据迁移**：作为后台作业运行，而不是在迁移脚本中运行（以避免部署超时）
- **迁移超时**：每次迁移 60 秒。超过此限制的迁移将被终止并且部署失败。

## 回滚流程

### 自动回滚
ECS 会在部署期间监控容器健康检查。如果新容器在 5 分钟内未通过健康检查，部署将自动回滚到上一个任务定义。健康检查端点 `GET /health` 必须在 3 秒内返回 200。

### 手动回滚
手动回滚生产部署：

1. 进入 AWS 控制台 > ECS > 集群 > 服务
2. 点击“更新服务”
3. 选择之前的任务定义修订版本
4. 勾选“强制新部署”
5. 点击“更新”

或通过 CLI：
```bash
aws ecs update-service --cluster techflow-prod \
  --service api-gateway \
  --task-definition api-gateway:42 \
  --force-new-deployment
```

回滚通常会在 3-4 分钟内完成。之前的 Docker 镜像在 ECR 中始终可用（镜像保留 90 天）。

### 数据库回滚
如果需要撤消数据库迁移：
```bash
alembic downgrade -1  # 回滚一个修订版本
```
仅当迁移具有正确的 `downgrade()` 函数时，此操作才有效。所有迁移都必须包括降级步骤。

## 健康检查

每个服务都会公开两个健康端点：

- `GET /health` — 基本存活检查。进程正常运行时返回 200，供 ECS 监控容器健康状态。
- `GET /health/ready` — 就绪检查。验证数据库连接、Redis 连接和 Kafka 消费者组状态；只有服务能够处理请求时才返回 200，供负载均衡器路由流量。

健康检查间隔：ECS 每 30 秒检查一次 `/health`；负载均衡器每 10 秒检查一次 `/health/ready`。就绪检查连续失败 3 次后，服务将从负载均衡器中移除。

## 扩展策略

### 自动扩缩容配置
每个服务都有独立的自动伸缩规则：

| 服务 | 最小实例数 | 最大实例数 | 扩容触发条件 | 缩容触发条件 |
| --- | --- | --- | --- | --- |
| API 网关 | 4 | 12 | CPU > 60%，持续 3 分钟 | CPU < 30%，持续 10 分钟 |
| 身份验证服务 | 3 | 8 | CPU > 70%，持续 3 分钟 | CPU < 30%，持续 10 分钟 |
| 项目服务 | 4 | 16 | CPU > 65%，持续 3 分钟 | CPU < 25%，持续 15 分钟 |
| 通知服务 | 2 | 6 | 队列深度 > 10,000 | 队列深度 < 1,000 |
| 搜索服务 | 2 | 8 | CPU > 70%，持续 3 分钟 | CPU < 30%，持续 10 分钟 |
| 文件服务 | 2 | 6 | CPU > 70%，持续 5 分钟 | CPU < 30%，持续 15 分钟 |

### 高峰时段
工作日美国东部时间上午 9:00～11:00 和下午 2:00～4:00 为流量高峰。系统会在上午 8:45 预扩容至最大容量的 75%，以避免冷启动延迟。

## 监控

### Prometheus 与 Grafana
所有服务都在 `/metrics` 端点上公开指标（Prometheus 格式）。关键仪表板：
- **服务运行状况**：请求率、错误率、延迟百分位数（p50、p95、p99）
- **数据库**：查询延迟、连接池利用率、复制延迟
- **Kafka**：消费者延迟、分区分布、消息吞吐量
- **业务指标**：活跃用户数、任务创建数、按端点统计的 API 调用量

### 警报规则
- **P1（立即呼叫值班人员）**：错误率 > 5% 且持续 2 分钟、服务完全不可用、数据库复制延迟 > 30 秒
- **P2（Slack 警报）**：错误率 > 1%，持续 5 分钟，p95 延迟 > 500 毫秒，磁盘使用率 > 80%
- **P3（工单）**：p95 延迟 > 200 毫秒、内存使用率 > 70%、证书将在 14 天内过期

### 事故响应
1. **检测**：通过 PagerDuty (P1) 或 Slack (P2/P3) 自动发出警报
2. **初步研判**：值班工程师在 5 分钟内评估严重程度和影响范围
3. **沟通**：发生 P1 事故后，在 10 分钟内更新 status.techflow.com 状态页
4. **解决**：应用修复，并通过监控确认系统恢复
5. **事后复盘**：所有 P1 和 P2 事故都应在 48 小时内完成书面复盘，包括时间线、根本原因和后续行动项
