# TechFlow API v3 参考文档

## 身份验证

所有 API 请求都需要通过以下两种方法之一进行身份验证：

### API 密钥认证
在 `X-TechFlow-Key` 请求头中携带 API 密钥。API 密钥可在管理控制台的“设置 > API 密钥”中生成。每个密钥都可配置权限范围（读取、写入、管理）和可选的到期日期。密钥长度为 64 个字符，以 `tfk_` 为前缀。

```
X-TechFlow-Key: tfk_abc123...
```

### OAuth2 认证
面向用户的应用应使用 OAuth2 授权码流程。在 developers.techflow.com 注册应用，以获取 `client_id` 和 `client_secret`。授权端点为 `https://auth.techflow.com/oauth2/authorize`，令牌端点为 `https://auth.techflow.com/oauth2/token`。访问令牌的有效期为 1 小时；可以使用刷新令牌在无需用户交互的情况下获取新令牌。刷新令牌的有效期为 30 天。

## 速率限制

速率限制按 API 密钥或 OAuth2 令牌分别执行：

- **基础版**：每分钟 100 个请求，每天 5,000 个请求
- **专业版**：每分钟 500 个请求，每天 50,000 个请求
- **企业版**：每分钟 2,000 个请求，每日请求数不限

触发速率限制时，API 返回 HTTP 429，并通过 `Retry-After` 响应头指明需要等待的秒数。每个响应都包含以下速率限制响应头：`X-RateLimit-Limit`、`X-RateLimit-Remaining`、`X-RateLimit-Reset`。

## 分页

列表端点使用基于游标的分页。每个响应都包含 `next_cursor` 字段，将其作为 `cursor` 查询参数传入即可获取下一页。默认每页 25 项，最多 100 项；通过 `limit` 查询参数设置每页数量。

```json
{
  "data": [...],
  "next_cursor": "eyJpZCI6MTAwfQ==",
  "has_more": true
}
```

## 核心端点

### 项目
- `GET /v3/projects` — 列出所有项目。支持 `status` 筛选条件（`active`、`archived`、`draft`）和 `sort` 排序字段（`created_at`、`updated_at`、`name`）。
- `POST /v3/projects` — 创建项目。必填字段：`name`（最多 128 个字符）、`workspace_id`。可选字段：`description`（最多 2,000 个字符）、`template_id`、`visibility`（`private`、`team`、`public`）。
- `GET /v3/projects/{id}` — 获取项目详情，包括成员数、任务数和存储用量。
- `PATCH /v3/projects/{id}` — 更新项目字段。支持部分更新。
- `DELETE /v3/projects/{id}` — 存档项目（软删除）。存档的项目将保留 90 天。

### 任务
- `GET /v3/projects/{id}/tasks` — 列出任务。支持以下筛选条件：`status`（`todo`、`in_progress`、`review`、`done`）、`assignee_id`、`priority`（`low`、`medium`、`high`、`critical`）、`due_before`、`due_after`、`label`。
- `POST /v3/projects/{id}/tasks` — 创建任务。必填字段：`title`（最多 256 个字符）。可选字段：`description`（Markdown，最多 10,000 个字符）、`assignee_id`、`priority`、`due_date`、`labels[]`、`parent_task_id`。
- `GET /v3/tasks/{id}` — 获取任务的完整详情、评论和活动历史。
- `PATCH /v3/tasks/{id}` — 更新任务。所有字段都是可选的。

### 用户
- `GET /v3/users` — 列出工作区成员。支持 `role` 筛选条件（`owner`、`admin`、`member`、`guest`）。
- `GET /v3/users/{id}` — 获取用户个人资料，包括角色、团队和活动统计数据。
- `POST /v3/users/invite` — 通过电子邮件邀请用户。必填字段：`email`、`role`。可选字段：`team_ids[]`。

### Webhook
- `POST /v3/webhooks` — 注册 Webhook。必填字段：`url`（仅限 HTTPS）、`events[]`。支持的事件：`project.created`、`project.updated`、`task.created`、`task.updated`、`task.completed`、`member.added`、`member.removed`。
- `GET /v3/webhooks` — 列出已注册的 Webhook 及其投递统计信息。
- `DELETE /v3/webhooks/{id}` — 删除 Webhook 注册。

Webhook 载荷会使用 Webhook 密钥通过 HMAC-SHA256 签名。处理载荷前，请验证 `X-TechFlow-Signature` 请求头。投递失败后会按指数退避策略重试 3 次（分别在 1 分钟、5 分钟和 30 分钟后）。

## 错误代码

| 状态码 | 含义 | 常见原因 |
| --- | --- | --- |
| 400 | 请求错误 | JSON 无效或缺少必填字段 |
| 401 | 未通过身份验证 | 缺少 API 密钥、API 密钥无效或令牌已过期 |
| 403 | 禁止访问 | 权限范围或访问权限不足 |
| 404 | 未找到 | 资源不存在或已归档 |
| 409 | 冲突 | 资源重复（例如项目名称重复） |
| 422 | 无法处理 | JSON 有效，但存在语义错误（例如日期无效） |
| 429 | 超出速率限制 | 请求过多，请检查 `Retry-After` 响应头 |
| 500 | 服务器错误 | 内部错误，请联系支持人员并提供请求 ID |

所有错误响应都包含便于支持人员追踪问题的 `request_id`，以及内容便于阅读的 `message` 字段。

## 版本控制

API 使用基于 URL 的版本控制（`/v3/`）。破坏性变更只会在新的主版本中引入。已弃用的端点会返回 `Sunset` 响应头，其中包含移除日期。当前版本 v3 于 2024 年 1 月发布；v2 将于 2025 年 6 月停止服务。
