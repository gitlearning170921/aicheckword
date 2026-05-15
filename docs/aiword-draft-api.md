# aiword 对接：文档初稿生成（aicheckword HTTP API）

本文描述 aicheckword 已提供的集成接口，供 **aiword** 后端代理调用。业务侧「谁调用了初稿、耗时、成功失败统计」请在 **aiword 本地库** 记录；本接口不在响应体中回传用户 API Key。

## aiword 侧路由前缀（仓库 `aiword`）

若使用仓库内已实现的 Flask 代理页，浏览器与前端脚本访问的是 **aiword 本机路径**（再由内网请求 aicheckword），前缀为：

- **页面**：`GET /draft-gen/`（需页面2登录后的 `session.user_id`）
- **API**（均在此前缀下）：
  - `GET/POST /draft-gen/api/llm-settings`：初稿 LLM（页面2**个人** Key，加密存；**必须**保存 Key；通义可选模型名；**不修改** aicheckword 系统设置）；**可选 provider 列表与「须个人 Key」策略**宜与上游 `GET .../draft/interop-config` 合并后再展示与校验
  - `GET /draft-gen/api/meta`：代理上游 `.../api/integration/draft/meta`
  - `GET /draft-gen/api/jobs`：本地任务列表
  - `POST /draft-gen/api/jobs`：`multipart` 代理上游 `.../api/integration/draft/jobs`
  - `GET /draft-gen/api/jobs/<local_id>/status`：代理上游 `.../jobs/{upstream_job_id}`
  - `GET /draft-gen/api/jobs/<local_id>/download`：代理上游 ZIP，并可缓存到 `OUTPUT_FOLDER/draft_zips/`

aiword 系统配置：`AICHECKWORD_DRAFT_API_BASE`（未配时回退 `QUIZ_API_BASE_URL`），超时 `AICHECKWORD_DRAFT_TIMEOUT_SECONDS`；与考试中心相同的上游鉴权头：`QUIZ_API_BEARER_TOKEN`、`QUIZ_API_SECRET`。

## 基址与路径（aicheckword 上游）

- 前缀：`/api/integration/draft`
- 示例：`POST {AICHECKWORD_API_BASE}/api/integration/draft/jobs`

## LLM 凭据（HTTP Header，勿写入 URL）

| Header | 说明 |
|--------|------|
| `X-Client-Llm-Api-Key` | DeepSeek：OpenAI 兼容 Key。**Cursor**：Dashboard Key。**通义**：DashScope Key。与 `X-Client-Llm-Personal-Keys-Only` 联用时**不回退**系统 `settings` 中的同厂商 Key |
| `X-Client-Llm-Personal-Keys-Only` | `1`/`true`：Key **仅**来自本 Header，禁止与系统管理员配置的 Key 回落合并；当上游 `interop-config` 中 `personalKeysOnly` 为 **false** 时建议**不发送**本头，与 aicheckword 系统配置「初稿集成」一致 |
| `X-Client-Llm-Base-Url` | 可选；OpenAI 兼容网关 Base URL；**Cursor** 时为 Cursor API Base |
| `X-Client-Llm-Model` | 可选；模型名（**通义**为 DashScope 模型名，如 `qwen-plus`） |
| `X-Client-Llm-Provider` | 可选；`deepseek` / `cursor` / `tongyi` / …；缺省使用 payload `provider` 再缺省 `settings.provider` |
| `X-Client-Cursor-Repository` | 可选；GitHub 仓库（与系统 `cursor_repository` 合并，请求优先） |
| `X-Client-Cursor-Ref` | 可选；分支/ref（与系统 `cursor_ref` 合并，请求优先） |

**Cursor 提交校验**：`provider=cursor` 时，合并后须存在 **API Key**（个人模式下须来自请求头）与 **repository**（可来自系统设置），否则 400。

**aiword 初稿页**：**deepseek / cursor / tongyi**，见 [`docs/integration-draft-provider-status.md`](integration-draft-provider-status.md)。个人 Key **必填**；与 aicheckword **系统**侧各厂商 Key **独立**（通过 `X-Client-Llm-Personal-Keys-Only` 禁止 Key 回落）。

## `POST /api/integration/draft/jobs`

`multipart/form-data`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `payload` | string (JSON) | 见下表 `DraftGeneratePayload` |
| `input_files` | file[] | 输入/参考文档（可多个）；`filename` 须为用户原始文件名 |
| `base_files` | file[] | 可选；就地修改用的基底文件 |

### `payload` JSON 字段（与 Streamlit 初稿页对齐的子集）

常用字段：`collection`, `base_case_id`, `template_file_names`, `project_id`, `document_language`, `registration_country`, `registration_type`, `registration_component`, `project_form`, `project_name`, `project_code`, … , `provider`, `inplace_patch`, `save_as_case`, `multi_base_auto_route`, `draft_strategy`, `author_role`, `author_role_map`, `audit_remediation_by_target`, `skip_case_template_text`, `docx_track_changes`, `base_files_by_target`, `aiword_user_id`, `aiword_task_id`。

- **`base_files_by_target`**：`{ "模板目标文件名.docx": "用户上传的基底文件名.docx" }`，其中值必须与 `base_files` 里某个文件的 **原始文件名** 一致。
- **`multi_base_auto_route`**：为 `true` 时，使用本次上传的全部 `base_files` 自动构造 `base_files_manifest`（按文件名排序）；与 Streamlit 中「多基础自动路由」一致时需同时上传多份 base。
- **`aiword_user_id` / `aiword_task_id`**：写入 aicheckword `operation_logs.extra`（便于与 Streamlit 运维日志关联），**不得**把 API Key 放进 payload。

响应：`{ "ok": true, "job_id": "<id>", "status": "queued" }`

## `GET /api/integration/draft/jobs/{job_id}`

返回 `status`：`queued` | `running` | `succeeded` | `failed`，以及 `progress`（0–1）、`message`、`error`、`result`（成功时含 `project_id`、`zip_path` 等）。

## `GET /api/integration/draft/jobs/{job_id}/download`

仅在 `status === succeeded` 时返回 `artifacts.zip`（内含本次导出落盘的文件）。

## `GET /api/integration/draft/meta`

查询参数：`collection`（默认 `regulations`）、可选 `base_case_id`。

返回项目列表、案例列表；若提供 `base_case_id` 则附带 `template_file_names`。

## `GET /api/integration/draft/interop-config`

无需表单体；与 aicheckword **系统配置 → 初稿集成** 入库字段一致，供 aiword 拉取后同步 UI 与提交前校验。

成功时 JSON 示例字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ok` | bool | 固定 `true` |
| `restrictProviders` | bool | `true` 表示管理员配置了非空的 `draft_interop_allowed_providers`，上游 **`POST .../jobs`** 会按 provider 白名单拒绝；`false` 表示留空不限制 |
| `allowedProviders` | array | 每项含 `id`（小写）、`label`、`requiresApiKey`（当前均为 `true`）；无白名单时为内置展示用全量 id 列表 |
| `personalKeysOnly` | bool | 与 `draft_interop_personal_keys_only` 一致；为 `false` 时上游不强制「仅个人 Key」语义 |
| `adminNotes` | string | `draft_interop_notes` |

aiword 实现建议：在 `GET/POST /draft-gen/api/llm-settings` 与提交初稿任务前拉取本接口（可短缓存 30–60 秒）；失败时回退本地默认白名单并提示「未同步上游联调策略」。

## aiword 本地「初稿任务」表（建议）

用于查询与统计（示例字段）：

- `id`, `user_id`, `created_at`, `updated_at`, `status`
- `upstream_job_id`（aicheckword 返回的 `job_id`）
- `collection`, `base_case_id`, `project_id`（成功后由轮询结果回填）
- `input_display_names_json`（原始文件名列表）
- `error_summary`, `duration_ms`, `local_zip_path`（下载成功后本地缓存 ZIP 路径，可选）

密钥仅存服务端加密字段；轮询与下载均由 **aiword 后端** 携带 Header 调用 aicheckword，避免浏览器持有 Key。

## 部署注意

- 初稿任务使用进程内 job 表 + 线程池；**uvicorn 多 worker** 时 job 状态不共享，首版请 **单 worker** 或后续改为 Redis/DB 任务表。
