# Cursor Rules / Skills 开工前预检

在修改 `.cursor/rules` 或 `.cursor/skills` 后，建议先运行预检脚本，避免 frontmatter 缺失、skill `name` 重复等问题进入协作流程。

## 运行方式

仓库根目录执行：

```bash
python scripts/preflight_cursor.py
```

Windows：

```text
preflight.bat
```

## 校验规则摘要

| 类型 | 要求 |
|------|------|
| `.cursor/rules/*.mdc` | 含 `description`；`alwaysApply` 或 `globs` 至少其一；正文非空 |
| `.cursor/skills/*/SKILL.md` | 含 `name`、`description`；`name` 小写连字符、≤64 字符、全局不重复；正文非空；超过 500 行时 WARN |

退出码：`0` 成功（可有 WARN），`1` 存在 ERROR。
