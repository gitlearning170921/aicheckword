# Signature Field Detection Examples

## Example 1: Triad Sign-off Block

### Input
```text
编制人：

日期：

审核人：

日期：

批准人：

日期：
```

### Expected Output
```json
{
  "block_id": "sig_block_001",
  "confidence": 0.95,
  "matched_rules": ["triad_rule", "strong_block_rule"],
  "fields": [
    {"name": "编制人", "type": "role"},
    {"name": "日期", "type": "date"},
    {"name": "审核人", "type": "role"},
    {"name": "日期", "type": "date"},
    {"name": "批准人", "type": "role"},
    {"name": "日期", "type": "date"}
  ],
  "is_confirmed_signature_block": true
}
```

---

## Example 2: Organization Signature Area

### Input
```text
企业负责人：

日      期：

盖      章：
```

### Expected Output
```json
{
  "block_id": "sig_block_002",
  "confidence": 0.93,
  "matched_rules": ["org_seal_rule", "strong_block_rule"],
  "fields": [
    {"name": "企业负责人", "type": "role"},
    {"name": "日期", "type": "date"},
    {"name": "盖章", "type": "signature_action"}
  ],
  "is_confirmed_signature_block": true
}
```

---

## Example 3: Table-style Multi-person Signatures

### Input
```text
风险管理小组
签字    日期    部门    代表人
研发部  方凯
研发部  陆亮
质量部  王珍临
```

### Expected Output
```json
{
  "block_id": "sig_block_003",
  "confidence": 0.9,
  "matched_rules": ["table_header_rule", "strong_block_rule"],
  "fields": [
    {"name": "签字", "type": "signature_action"},
    {"name": "日期", "type": "date"},
    {"name": "部门", "type": "org_unit"},
    {"name": "代表人", "type": "role"}
  ],
  "is_confirmed_signature_block": true
}
```

---

## Example 4: Approval with Checkbox

### Input
```text
产品负责人审核意见
□同意发布  □不同意发布

测试负责人审核意见
□同意发布  □不同意发布

项目负责人审核意见
□同意发布  □不同意发布

作者：
日期：
审核：
日期：
批准：
日期：
```

### Expected Output
```json
{
  "block_id": "sig_block_004",
  "confidence": 0.91,
  "matched_rules": ["approval_checkbox_rule", "strong_block_rule"],
  "fields": [
    {"name": "同意发布", "type": "approval_decision"},
    {"name": "不同意发布", "type": "approval_decision"},
    {"name": "作者", "type": "role"},
    {"name": "日期", "type": "date"},
    {"name": "审核", "type": "role"},
    {"name": "日期", "type": "date"},
    {"name": "批准", "type": "role"},
    {"name": "日期", "type": "date"}
  ],
  "is_confirmed_signature_block": true
}
```

---

## Example 5: Low-confidence Candidate (Single Token)

### Input
```text
请在此处签字。
```

### Expected Output
```json
{
  "block_id": "sig_candidate_001",
  "confidence": 0.42,
  "matched_rules": ["single_token_fallback"],
  "fields": [
    {"name": "签字", "type": "signature_action"}
  ],
  "is_confirmed_signature_block": false,
  "needs_human_review": true
}
```

---

## Example 6: Noise to Exclude

### Input
```text
。
```

### Expected Output
```json
{
  "ignored": true,
  "reason": "noise_content",
  "rules": ["noise_filter"]
}
```

