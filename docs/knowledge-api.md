# 对外开放 API 文档（仅知识库查询）



## 1. 基本信息

- Base URL：`http://192.168.6.10:8000`

---

## 2. 查询接口（第三方调用）

### 2.1 `POST /knowledge/search`

`Content-Type: application/json`

#### 请求参数

**必填参数**

- `query`（自由文本）
- `registration_country`
- `registration_type`
- `registration_component`
- `project_form`
- `document_language`（建议值：`zh` / `en` / `both`）

**可选参数（来自 `GET /knowledge/search/options`）**

- `project_name` / `project_name_en`
- `product_name` / `product_name_en`
- `model` / `model_en`
- `registration_country_en`
- `case_name`（项目案例名称）
- `case_country`（项目案例国家）
- `case_type`（项目案例类别）
- `collection`（默认 `regulations`，多知识库场景可指定）

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
  "model_en": "A1",
  "case_name": "BCMAS-ITR-001",
  "case_country": "中国",
  "case_type": "医疗器械一类Ι"
}
```

#### 响应示例（节选）

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
    "model_en": "A1",
    "case_name": "BCMAS-ITR-001",
    "case_country": "中国",
    "case_type": "医疗器械一类Ι"
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

## 3. 参数可选值接口（用于前端下拉）

### 3.1 `GET /knowledge/search/options`

支持可选查询参数：`collection`（默认 `regulations`）。

返回当前系统配置下的可选值（与页面配置一致，且自动去重）。
推荐第三方前端优先使用：
- `project_list`：项目对象列表（适合联动下拉）
- `*_options`：扁平选项列表（适合简单下拉）
- `fields`：字段约束/必填规则

#### 响应示例（节选）

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
  "project_list": [
    {
      "project_name": "OXGWIS",
      "project_name_en": "OXGWIS",
      "product_name": "氧网关信息系统",
      "product_name_en": "Oxygen Gateway Web Information System",
      "model": "A1",
      "model_en": "A1",
      "registration_country": "欧盟",
      "registration_type": "医疗器械二类Ⅱb",
      "registration_component": "独立软件",
      "project_form": "Web"
    }
  ],
  "project_name_options": ["OXGWIS"],
  "project_name_en_options": ["OXGWIS"],
  "product_name_options": ["氧网关信息系统"],
  "product_name_en_options": ["Oxygen Gateway Web Information System"],
  "model_options": ["A1"],
  "model_en_options": ["A1"],
  "project_name": {"type": "string", "required": false, "description": "可选，自由文本"},
  "project_name_en": {"type": "string", "required": false, "description": "可选，自由文本"},
  "product_name": {"type": "string", "required": false, "description": "可选，自由文本"},
  "product_name_en": {"type": "string", "required": false, "description": "可选，自由文本"},
  "model": {"type": "string", "required": false, "description": "可选，自由文本"},
  "model_en": {"type": "string", "required": false, "description": "可选，自由文本"},
  "case_name": ["BCMAS-ITR-001", "BCMAS-IRPR-001"],
  "case_country": ["中国", "欧盟"],
  "case_type": ["医疗器械一类Ι", "医疗器械二类Ⅱ"],
  "fields": {
    "query": {"type": "string", "required": true, "description": "自由文本检索词"},
    "registration_country": {"type": "enum", "required": true, "options": ["中国", "美国", "欧盟"]},
    "registration_type": {"type": "enum", "required": true, "options": ["医疗器械一类Ι", "医疗器械二类Ⅱ", "医疗器械二类Ⅱa", "医疗器械二类Ⅱb", "医疗器械三类Ⅲ"]},
    "registration_component": {"type": "enum", "required": true, "options": ["有源医疗器械", "软件组件", "独立软件"]},
    "project_form": {"type": "enum", "required": true, "options": ["Web", "APP", "PC"]},
    "document_language": {"type": "enum", "required": true, "options": ["zh", "en", "both"]},
    "project_name": {"type": "enum", "required": false, "options": ["OXGWIS"]},
    "project_name_en": {"type": "enum", "required": false, "options": ["OXGWIS"]},
    "product_name": {"type": "enum", "required": false, "options": ["氧网关信息系统"]},
    "product_name_en": {"type": "enum", "required": false, "options": ["Oxygen Gateway Web Information System"]},
    "model": {"type": "enum", "required": false, "options": ["A1"]},
    "model_en": {"type": "enum", "required": false, "options": ["A1"]},
    "case_name": {"type": "enum", "required": false, "options": ["BCMAS-ITR-001", "BCMAS-IRPR-001"]},
    "case_country": {"type": "enum", "required": false, "options": ["中国", "欧盟"]},
    "case_type": {"type": "enum", "required": false, "options": ["医疗器械一类Ι", "医疗器械二类Ⅱ"]}
  }
}
```

#### 第三方接入建议

- 页面初始化：先调 `GET /knowledge/search/options?collection=...`
- 表单渲染：必填项按 `fields` 中 `required=true` 控制
- 联动逻辑：优先使用 `project_list` 填充“项目案例名称/国家/类别/项目名/产品名/型号”联动
- 简单下拉：可直接使用 `project_name_options`、`case_name`、`case_country`、`case_type`

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

