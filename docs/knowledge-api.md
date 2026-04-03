# 对外开放 API 文档（仅知识库查询）

当前仅开放知识库查询接口，**文档审核 API 暂不对外开放**。

服务入口：`src/api/server.py`

## 1. 基本信息

- Base URL：`http://192.168.6.10:8000`
- Swagger：`/docs`
- OpenAPI：`/openapi.json`
- 建议网关鉴权（API Key/JWT）

---

## 2. 查询接口

### 2.1 `POST /knowledge/search`

`Content-Type: application/json`

#### 请求参数（仅开放以下）

**必填参数**

- `query`（自由文本）
- `registration_country`
- `registration_type`
- `registration_component`
- `project_form`
- `document_language`（建议值：`zh` / `en` / `both`）

**可选参数**

- `project_name` / `project_name_en`
- `product_name` / `product_name_en`
- `model` / `model_en`
- `registration_country_en`

> 说明：接口按上述条件检索并返回满足条件的数据。

#### 请求示例

```json
{
  "query": "风险需求中的 CS 编号",
  "registration_country": "欧盟",
  "registration_type": "医疗器械二类Ⅱb",
  "registration_component": "独立软件",
  "project_form": "Web",
  "document_language": "en",
  "project_name": "OXGWIS",
  "project_name_en": "OXGWIS",
  "product_name": "氧网关信息系统",
  "product_name_en": "Oxygen Gateway Web Information System",
  "model": "A1",
  "model_en": "A1"
}
```

#### 响应示例

```json
{
  "conditions": {
    "query": "风险需求中的 CS 编号",
    "registration_country": "欧盟",
    "registration_type": "医疗器械二类Ⅱb",
    "registration_component": "独立软件",
    "project_form": "Web",
    "document_language": "en",
    "project_name": "OXGWIS",
    "project_name_en": "OXGWIS",
    "product_name": "氧网关信息系统",
    "product_name_en": "Oxygen Gateway Web Information System",
    "model": "A1",
    "model_en": "A1"
  },
  "results": [
    {
      "content": "......命中片段......",
      "source": "OXGWIS-RAS-001 Risk Analysis.xlsx",
      "metadata": {
        "source_file": "OXGWIS-RAS-001 Risk Analysis.xlsx",
        "file_type": ".xlsx"
      }
    }
  ],
  "total": 1
}
```

---

## 3. 参数可选值接口

### 3.1 `GET /knowledge/search/options`

返回当前系统配置下的可选值（与页面配置一致）。
该接口**已包含** `project_name / project_name_en / product_name / product_name_en / model / model_en`，无需再新增接口。

#### 响应示例

```json
{
  "registration_country": ["中国", "美国", "欧盟"],
  "registration_type": [
    "医疗器械一类Ι",
    "医疗器械二类Ⅱ",
    "医疗器械二类Ⅱa",
    "医疗器械二类Ⅱb",
    "医疗器械三类Ⅲ"
  ],
  "registration_component": ["有源医疗器械", "软件组件", "独立软件"],
  "project_form": ["Web", "APP", "PC"],
  "document_language": ["zh", "en", "both"],
  "project_name": {"type": "string", "required": false, "description": "可选，自由文本"},
  "project_name_en": {"type": "string", "required": false, "description": "可选，自由文本"},
  "product_name": {"type": "string", "required": false, "description": "可选，自由文本"},
  "product_name_en": {"type": "string", "required": false, "description": "可选，自由文本"},
  "model": {"type": "string", "required": false, "description": "可选，自由文本"},
  "model_en": {"type": "string", "required": false, "description": "可选，自由文本"},
  "fields": {
    "query": {"type": "string", "required": true, "description": "自由文本检索词"},
    "registration_country": {"type": "enum", "required": true, "options": ["中国", "美国", "欧盟"]},
    "registration_type": {"type": "enum", "required": true, "options": ["医疗器械一类Ι", "医疗器械二类Ⅱ", "医疗器械二类Ⅱa", "医疗器械二类Ⅱb", "医疗器械三类Ⅲ"]},
    "registration_component": {"type": "enum", "required": true, "options": ["有源医疗器械", "软件组件", "独立软件"]},
    "project_form": {"type": "enum", "required": true, "options": ["Web", "APP", "PC"]},
    "document_language": {"type": "enum", "required": true, "options": ["zh", "en", "both"]},
    "project_name": {"type": "string", "required": false},
    "project_name_en": {"type": "string", "required": false},
    "product_name": {"type": "string", "required": false},
    "product_name_en": {"type": "string", "required": false},
    "model": {"type": "string", "required": false},
    "model_en": {"type": "string", "required": false}
  }
}
```

---

## 4. 当前参数可选值（静态说明）

- `document_language`：`zh` / `en` / `both`
- `registration_type`：
  - `医疗器械一类Ι`
  - `医疗器械二类Ⅱ`
  - `医疗器械二类Ⅱa`
  - `医疗器械二类Ⅱb`
  - `医疗器械三类Ⅲ`
- `registration_component`：
  - `有源医疗器械`
  - `软件组件`
  - `独立软件`
- `registration_country`、`project_form` 以 `GET /knowledge/search/options` 返回值为准（可在后台页面配置后动态变化）

---

## 5. 错误码

- `200`：成功
- `400`：参数错误（例如 `query` 为空，或必填维度为空）
- `500`：服务内部错误

---

## 6. 建议的对外网关白名单

- `POST /knowledge/search`
- `GET /knowledge/search/options`
- `GET /`（可选健康检查）

