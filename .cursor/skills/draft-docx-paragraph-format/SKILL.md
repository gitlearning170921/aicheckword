---
name: draft-docx-paragraph-format
description: Docx 初稿/就地修改时插入段与基底版式一致：缩略语与法规分项换行、字体/缩进继承、参考文献 [n] 续号；与项目案例知识库模板对齐。用户提到「插入段落格式、Definitions、References、换行、首行缩进、就地 patch、draft_export」时使用。
---

# Docx 段落版式与分项（初稿生成 + 就地修改）

## 知识库与格式来源

- **项目案例知识库（`project_case`）**：通过 `get_knowledge_docs_by_case_id_and_file_name` 等拼成提示中的 **模板/参考正文**，体现目标注册文档常见的 **章节标题、段落缩进、Times New Roman/小四类习惯、Definitions / References 写法**。
- **用户基底**：**EXISTING_DRAFT_TEXT** 或上传的 **Base Word** 是结构与样式的**优先真源**；生成/补丁须 **对齐该基底**，而不是单凭空重构版式。

## 规则总表（写作侧 / 模型）

| 场景 | 要求 |
|------|------|
| 多条缩略语定义 | **每条单独成段或单独一行**；`new_text` 用换行分隔；禁止一大段内用仅空格串联多个 `X refers to` |
| 多条法规 / 标准引用 | **分项**；长串用换行或分号分隔；模型在 `PATCH_JSON` 的 `new_text` 中优先 **换行** |
| 参考文献 `[1]` `[2]`… | 与基底小节已有样式一致；新增时 **续号**，不要与现有 `[n]` 冲突或全部挤在 `[1]` 下 |
| REQ/URS/CS 等追溯号 | **逐字复制**基底或输入，不随拆段改写（见 `document-authoring-and-audit`） |
| 项目标识 | **NEW_BASIC_INFO / 项目约束** 为准；参考文件里的他案公司名/产品名不得覆盖（见 `document_draft_generator` 提示） |

## 规则总表（程序侧 / `draft_export.export_docx_inplace_patch`）

| 能力 | 说明 |
|------|------|
| 拆段 | `_expand_docx_insert_text_to_paragraphs`：按换行 + `_split_definitions_and_standards_blob`（多 `[n]`、多 `refers to`、长法规分号等）拆成多段 |
| 版式 | `_copy_paragraph_layout_and_style`：新段 `style` + `paragraph_format` 对齐锚点段 |
| 字体 | `_paragraph_effective_run_rpr` + `_replace_paragraph_with_track_changes` / `_insert_paragraph_with_track_changes`：修订内继承 **rPr** |
| `[n]` 续号 | `_insert_needs_bracket_renumber` 判定语境；`_normalize_reference_chunks` 基于 `_max_bracket_ref_index_in_doc` 续编 |
| 去重 | `_find_duplicate_followup_paragraph`：插入前若后续段已高度相似则 **改写** 而非再插一段 |

## 与案例模板核对的操作建议

1. 在 **文档初稿生成** 中选定的 **模板案例**，打开知识库中同名文件的 **Definitions / References** 样例：看是「一段一条」还是「编号列表」。
2. 生成 `PATCH_JSON` 时，`new_text` **显式换行** 对齐该样例密度。
3. 导出后若仍与模板列表样式不一致：检查基底是否用 **Word 列表样式** 而无显式 `[n]`——此类需在模板或后续开发中单独处理（见 rule 第 5 节）。

## 验证

- 修改 `draft_export.py` / 提示词后：`python -m py_compile src/core/draft_export.py src/core/document_draft_generator.py`
- 用含 Definitions + References 的样例 docx 跑一轮就地导出，目视 **字体/首行缩进/分段数/`[n]` 连续性**。
