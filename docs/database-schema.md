# 数据库表结构说明

本文档说明各表、字段含义及可输入项（用于训练统计类型修正与维护参考）。

**统计约定**：页面上的「法规向量块数」「审核点向量块数」「训练统计」等一律以数据库表统计为准（`knowledge_docs` / `checkpoint_docs`）；所有训练（法规、审核点、项目专属）均写入对应表。

---

## 训练统计类型对应字段（你要改的）

**训练统计**里「法规文件 / 程序文件 / 项目案例文件」的类型，来自下表字段：

| 表名 | 字段名 | 含义 | 可选值（当前） | 可输入项个数 |
|------|--------|------|----------------|--------------|
| **knowledge_docs** | **category** | 法规知识库文档分类，用于训练统计按类型汇总 | `regulation`（法规文件）、`program`（程序文件）、`project_case`（项目案例文件） | **3 种** |

- **要改统计类型时**：  
  1. **只改界面显示**：改 `src/app.py` 里的 `CATEGORY_LABELS`、`CATEGORY_VALUES`，以及训练页「文件分类」下拉选项，保持与下面一致。  
  2. **要改或新增分类**：  
     - 在 `src/core/db.py` 里 `get_knowledge_stats_by_category` 的 `by_category` 补全逻辑中，为你新增的 category 值预留（如新加 `"custom"`）。  
     - 在 `src/app.py` 的 `CATEGORY_LABELS`、`CATEGORY_VALUES` 和训练页下拉里增加对应选项。  
     - 新训练时选择新类型，则 `knowledge_docs.category` 会存新值，统计会按新类型汇总。

---

## 表与字段一览

### 1. app_settings（应用配置，单行）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | INT | 主键，固定 1 | 1 条 |
| provider | VARCHAR(32) | AI 服务：ollama / openai / cursor | 3 种 |
| openai_api_key | VARCHAR(1024) | OpenAI 类 API Key | 1 项 |
| openai_base_url | VARCHAR(512) | OpenAI 类 Base URL | 1 项 |
| ollama_base_url | VARCHAR(256) | Ollama 服务地址 | 1 项 |
| cursor_api_key | VARCHAR(512) | Cursor API Key | 1 项 |
| cursor_api_base | VARCHAR(512) | Cursor API Base | 1 项 |
| cursor_repository | VARCHAR(512) | Cursor 关联仓库 | 1 项 |
| cursor_ref | VARCHAR(64) | 分支/标签 | 1 项 |
| cursor_embedding | VARCHAR(32) | 向量化用 ollama/openai | 2 种 |
| llm_model | VARCHAR(128) | 大模型名称 | 1 项 |
| embedding_model | VARCHAR(128) | 向量模型名称 | 1 项 |
| **review_extra_instructions** | **LONGTEXT** | **自定义审核要求/提示词，会追加到审核上下文中以提升审核质量** | 在「文档审核」页「自定义审核要求」中填写并保存 |
| created_at / updated_at | DATETIME | 创建/更新时间 | 自动 |

---

### 2. operation_logs（操作记录）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| op_type | VARCHAR(64) | 操作类型 | 见下 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| file_name | VARCHAR(512) | 文件名/批次标识 | 1 项 |
| source | VARCHAR(1024) | 来源路径/说明 | 1 项 |
| extra_json | LONGTEXT | 扩展信息 JSON | 任意 |
| model_info | VARCHAR(256) | 使用模型描述 | 1 项 |
| created_at | DATETIME | 发生时间 | 自动 |

**op_type 可选值（可输入项约 11 种）**：  
`train_batch`、`train`、`train_error`、`generate_checklist`、`train_checklist`、`train_project`、`review_batch`、`review`、`review_error`、`review_text`、`review_text_error`、`correction`。

---

### 3. audit_reports（审核报告）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| file_name | VARCHAR(512) | 被审文件名 | 1 项 |
| report_json | LONGTEXT | 报告全文 JSON | 1 项 |
| model_info | VARCHAR(256) | 使用模型 | 1 项 |
| total_points | INT | 审核点总数 | 1 项 |
| high_count / medium_count / low_count / info_count | INT | 各严重程度数量 | 各 1 项 |
| summary | TEXT | 总结文案 | 1 项 |
| created_at | DATETIME | 创建时间 | 自动 |

---

### 4. knowledge_docs（法规知识库文档块）★ 训练统计类型在此

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| file_name | VARCHAR(512) | 来源文件名 | 1 项 |
| chunk_index | INT | 块序号 | 1 项 |
| content | LONGTEXT | 块文本内容 | 1 项 |
| metadata_json | LONGTEXT | 元数据 JSON | 任意 |
| **category** | **VARCHAR(32)** | **文档分类（训练统计类型）** | **当前 3 种：regulation / program / project_case** |
| case_id | BIGINT | 关联 project_cases.id，仅 category=project_case 时有值 | 可选 |
| created_at | DATETIME | 创建时间 | 自动 |

---

### 4.1. project_cases（项目案例元数据）

第一步训练「项目案例文件」时创建/选择案例，知识库块通过 `knowledge_docs.case_id` 关联到本表。

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| case_name / case_name_en | VARCHAR | 案例名称（中/英） | 各 1 项 |
| product_name / product_name_en | VARCHAR | 产品名称（中/英） | 各 1 项 |
| registration_country / registration_country_en | VARCHAR | 注册国家（中/英） | 各 1 项 |
| registration_type / registration_component / project_form | VARCHAR | 注册类别/组成/形态 | 来自维度选项 |
| scope_of_application | TEXT | 产品适用范围 | 1 项 |
| document_language | VARCHAR(32) | 案例文档语言：zh/en/both | 3 种 |
| **project_key** | **VARCHAR(256)** | **关联项目标识：同一项目下多语言/多国家案例填相同值，用于分组与「新建另一语言/国家」** | 可选，训练时在「关联项目（可选）」中填写或选择 |
| created_at | DATETIME | 创建时间 | 自动 |

---

### 5. audit_corrections（审核纠正记录）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| report_id | BIGINT | 关联 audit_reports.id | 1 项 |
| point_index | INT | 审核点下标 | 1 项 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| file_name | VARCHAR(512) | 被审文件名 | 1 项 |
| original_json | LONGTEXT | 纠正前 JSON | 1 项 |
| corrected_json | LONGTEXT | 纠正后 JSON | 1 项 |
| fed_to_kb | TINYINT | 是否已回馈知识库 0/1 | 2 种 |
| created_at | DATETIME | 创建时间 | 自动 |

---

### 6. audit_checklists（审核点清单）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| name | VARCHAR(256) | 清单名称 | 1 项 |
| checklist_json | LONGTEXT | 审核点数组 JSON | 1 项 |
| total_points | INT | 审核点数量 | 1 项 |
| base_file | VARCHAR(512) | 基础文件名（如有） | 1 项 |
| model_info | VARCHAR(256) | 生成时模型 | 1 项 |
| status | VARCHAR(32) | draft / trained | 2 种 |
| created_at / updated_at | DATETIME | 创建/更新时间 | 自动 |

---

### 7. dimension_options（维度选项，单行，页面可配置）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | INT | 主键，固定 1 | 1 条 |
| registration_countries | LONGTEXT | 注册国家列表 JSON 数组 | 侧栏配置，默认 3 项：中国、美国、欧盟 |
| project_forms | LONGTEXT | 项目形态列表 JSON 数组 | 侧栏配置，默认 3 项：Web、APP、PC |
| updated_at | DATETIME | 更新时间 | 自动 |

---

### 8. projects（项目）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| collection | VARCHAR(128) | 所属知识库 | 1 项 |
| name | VARCHAR(256) | 项目名称 | 1 项 |
| registration_country | VARCHAR(128) | 注册国家 | 来自 dimension_options.registration_countries |
| registration_type | VARCHAR(128) | 注册类别 | 固定 5 种：医疗器械一类Ι/二类Ⅱ/二类Ⅱa/二类Ⅱb/三类Ⅲ |
| registration_component | VARCHAR(128) | 注册组成 | 固定 3 种：有源医疗器械、软件组件、独立软件 |
| project_form | VARCHAR(128) | 项目形态 | 来自 dimension_options.project_forms |
| **basic_info_text** | **TEXT** | **从项目资料中提取的基本信息（项目名称、产品名称、型号等），训练后若知识库中无此条则写入此处，审核时与待审文档一致性核对** | 系统根据项目知识库内容自动提取写入 |
| created_at / updated_at | DATETIME | 创建/更新时间 | 自动 |

---

### 9. project_knowledge_docs（项目专属知识库文档块）

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| project_id | BIGINT | 项目 id | 1 项 |
| collection | VARCHAR(128) | 知识库名 | 1 项 |
| file_name | VARCHAR(512) | 来源文件名 | 1 项 |
| chunk_index | INT | 块序号 | 1 项 |
| content | LONGTEXT | 块文本内容 | 1 项 |
| metadata_json | LONGTEXT | 元数据 JSON | 任意 |
| created_at | DATETIME | 创建时间 | 自动 |

---

### 10. checkpoint_docs（审核点知识库文档块）

审核点训练（第二步「训练此清单」）写入此表，统计以本表为准。

| 字段 | 类型 | 含义 | 可输入项/说明 |
|------|------|------|----------------|
| id | BIGINT | 主键 | 自增 |
| collection | VARCHAR(128) | 知识库名（与 regulations 等一致） | 1 项 |
| file_name | VARCHAR(512) | 来源（如「审核点清单」） | 1 项 |
| chunk_index | INT | 块序号 | 1 项 |
| content | LONGTEXT | 块文本内容 | 1 项 |
| metadata_json | LONGTEXT | 元数据 JSON | 任意 |
| created_at | DATETIME | 创建时间 | 自动 |

---

## 修改训练统计类型时需同步的地方

1. **数据库**：`knowledge_docs.category` 存的是英文键（如 `regulation`、`program`、`project_case`），若新增类型请用新英文键。  
2. **代码**：  
   - `src/app.py`：`CATEGORY_LABELS`（英文→中文）、`CATEGORY_VALUES`（中文→英文）、训练页「文件分类」下拉的选项列表。  
   - `src/core/db.py`：`get_knowledge_stats_by_category` 里对 `by_category` 的补全（保证所有类型在统计里都有 0 的初始值）。  
3. **训练页**：上传/目录训练时的「文件分类」下拉选项需与上面一致，选中的值会写入 `knowledge_docs.category`。
