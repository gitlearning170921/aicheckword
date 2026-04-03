---
name: signature-field-identification
description: Identify signature-related fields in Chinese medical-regulatory documents (e.g., 编制人/审核人/批准人/企业负责人/签字/日期/盖章) using project knowledge-base patterns. Use when users request signature block detection, sign-off field extraction, or signature position recognition rules.
---
# Signature Field Identification

## Purpose
Provide stable rules for detecting "签名位/签字位/签章位" fields from document text blocks, based on existing project knowledge-base cases.

## Knowledge-Base Evidence (queried from current project)
High-value patterns repeatedly found in project-case/program documents:

1. `QR-QP7.3.8-01 设计和开发转换计划.docx`  
   - `编制人` + `日期` + `审核人` + `日期` + `批准人` + `日期`
2. `QR-QP7.3.8-01 设计和开发转换计划（2025.07.22）-MDSAP（已翻译）.docx`  
   - same triad pattern: `编制人/审核人/批准人` and paired `日期`
3. `QR-SMP 7.3-02-01 软件发布说明（2025.07.25）已翻译.docx`  
   - `作者` + `日期` + `审核` + `日期` + `批准` + `日期`  
   - plus approval options: `□同意发布  □不同意发布`
4. `PAPUWIS-RMR-001风险管理报告.docx`  
   - group sign-off table: `签字` + `日期` + `部门` + `代表人`
5. `QR-QP6.2-06 任命书-黄华（管理者代表）.docx`  
   - organization signature block: `企业负责人` + `日期` + `盖章`

## Canonical Signature Field Set
Treat the following as canonical "signature-related fields":

- Person role fields  
  - `编制人`, `审核人`, `批准人`, `作者`, `审核`, `批准`, `企业负责人`, `代表人`, `法定代表人`
- Signature action fields  
  - `签字`, `签名`, `签章`, `盖章`
- Date fields  
  - `日期`, `签署日期`, `Date`, `Signed Date`
- Decision/approval fields  
  - `同意发布`, `不同意发布`, checkbox-style approval states

## Detection Rules
Use these rules in order (high -> low confidence):

1. **Strong block rule (high confidence)**  
   If a local region (same paragraph/table area/page zone) contains at least:
   - one role field (`编制人/审核人/批准人/...`) AND
   - one date field (`日期/...`)  
   then classify as signature block.

2. **Triad rule (high confidence)**  
   If any ordered or unordered combination appears:
   - `编制人 + 审核人 + 批准人`  
   (with or without repeated `日期`)  
   classify as formal sign-off block.

3. **Org-seal rule (high confidence)**  
   `企业负责人` near (`日期` or `盖章`) => organization-level signature area.

4. **Table-header rule (medium confidence)**  
   In table-like structures containing columns/adjacent labels:
   - `签字`, `日期`, `部门`, `代表人`  
   classify as multi-person sign table.

5. **Approval-checkbox rule (medium confidence)**  
   Presence of `□同意发布` / `□不同意发布` with reviewer roles indicates approval signature section.

6. **Single-token fallback (low confidence)**  
   A standalone token like `签字` or `盖章` without role/date context is only a candidate, not a final signature block.

## Noise Filtering (important)
Current KB has many glossary rows and punctuation-only chunks. Filter out:

- `metadata.category` not in `{project_case, program}` unless no better hit
- content length < 3 or content in `{".", "。"}`
- generic glossary text that lacks role/date/action keywords

## Minimal Output Schema (recommended)
When returning detected fields, use:

```json
{
  "block_id": "sig_block_001",
  "confidence": 0.93,
  "matched_rules": ["triad_rule", "strong_block_rule"],
  "fields": [
    {"name": "编制人", "type": "role"},
    {"name": "审核人", "type": "role"},
    {"name": "批准人", "type": "role"},
    {"name": "日期", "type": "date"}
  ],
  "source_hint": "QR-QP7.3.8-01 设计和开发转换计划.docx"
}
```

## Confidence Guidance
- `>= 0.85`: role + date co-occurrence, or triad present
- `0.60 ~ 0.84`: action/date present but role incomplete
- `< 0.60`: isolated keywords only, require human confirmation

## Quick Checklist
- Prefer table/nearby context over single-token matching
- Prioritize `编制/审核/批准` and `企业负责人+盖章+日期` patterns
- Always keep date fields linked to their nearest role/action labels
- Keep fallback candidates separate from confirmed signature blocks
