# 初稿集成 API：与 aiword 联调状态（产品侧白名单）

本文件用于在仓库内**显式标记**各 LLM 提供方在「aicheckword 集成初稿 API + aiword 代理页」路径上的联调结论。

## 管理员可改的「联调状态」与 aiword 同步

在 **aicheckword** Streamlit **系统配置 →「初稿集成」** 分区可编辑并保存（与其它项一同写入 `runtime_settings_json`）：

- **`draft_interop_allowed_providers`**：逗号分隔的小写 provider id；**留空**表示服务端**不**按白名单拒绝任务（兼容旧部署）。非空时，`POST .../draft/jobs` 会在解析出有效 `provider` 后校验，不在列表中则 400。
- **`draft_interop_personal_keys_only`**：为 **False** 时，服务端**忽略**请求里「仅个人 Key」的严格语义（等价于不强制个人 Key 不回退系统 Key），便于联调；为 **True** 时与 `X-Client-Llm-Personal-Keys-Only` 说明一致。
- **`draft_interop_notes`**：纯文本，经接口下发给客户端展示（运维说明、窗口期提示等）。

**aiword** 应在进入初稿相关页或保存个人 LLM 前请求：

`GET {AICHECKWORD_API_BASE}/api/integration/draft/interop-config`

响应字段约定见 [`docs/aiword-draft-api.md`](aiword-draft-api.md) 中「interop-config」一节；用于下拉白名单、`personalKeysOnly` 与是否发送 `X-Client-Llm-Personal-Keys-Only` 等行为的同步。

## aiword（页面2）与 aicheckword（系统设置）职责划分

| 配置位置 | 用途 | 是否影响对方已保存配置 |
|----------|------|--------------------------|
| **aiword**「个人 LLM 设置」 | **个人** API Key（加密，必填）；可选 **API Base URL**、**审核/对话模型**（空则用上游默认） | **否**：仅通过 HTTP Header 传给 aicheckword 单次请求，不写 aicheckword 库 |
| **aicheckword** 系统设置 / 侧栏 | **管理员**全局 AI（含 `cursor_repository`、`DASHSCOPE_API_KEY` 等） | **否**：在「须个人 Key」开启且 aiword 发送 `X-Client-Llm-Personal-Keys-Only` 时，**各厂商 Key 不回退**到此处；Cursor 的仓库/Base/ref 仍从此处合并 |

## 初稿 provider 白名单（aiword 初稿页）

| provider | aiword 必填 | 发往 aicheckword 的 Key 头 | 其他回落 |
|----------|-------------|---------------------------|----------|
| **deepseek** | 个人 API Key | `X-Client-Llm-Api-Key`；在 `personalKeysOnly` 开启时加 `X-Client-Llm-Personal-Keys-Only`；可选 `X-Client-Llm-Base-Url` / `X-Client-Llm-Model` | 未传 Base/模型时回落 aicheckword 系统默认；个人 Key 模式下 **Key 不回退系统** |
| **cursor** | 个人 Cursor API Key | 同上 + `X-Client-Llm-Provider: cursor`；可选 Base/Model 头 | **仓库 / ref** 等与系统 `cursor_*` 合并；个人 Key 模式下 **Key 不回退系统** |
| **tongyi** | 个人 DashScope API Key | 同上 + `X-Client-Llm-Provider: tongyi`；可选 `X-Client-Llm-Model` | 模型可空则上游默认；个人 Key 模式下 **Key 不回退系统** |

## HTTP Header 补充

| Header | 说明 |
|--------|------|
| `X-Client-Llm-Personal-Keys-Only` | 为 `1` / `true` 时：`X-Client-Llm-Api-Key` **不得**为空且 **不得**与 `settings` 中的各厂商 Key 做「二选一回落」；若 aicheckword 将 `draft_interop_personal_keys_only` 设为 **False**，服务端对该请求不再强制此语义（aiword 宜根据 `interop-config.personalKeysOnly` 决定是否发此头） |
| `X-Client-Cursor-Repository` / `X-Client-Cursor-Ref` | 可选；与系统设置合并（非个人 Key 范畴） |

创建任务时若 `provider=cursor`，合并后仍须存在 **repository**（来自 Header 或系统设置），否则 400。
