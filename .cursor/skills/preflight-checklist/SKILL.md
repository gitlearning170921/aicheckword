---
name: preflight-checklist
description: 开工前运行 scripts/preflight_cursor.py 校验 .cursor/rules 与 .cursor/skills 的 frontmatter 与结构，减少元数据错误与重复踩坑。用户提到“开工前检查/预检/rules/skills 校验”时使用。
---

# 开工前预检（Rules / Skills）

## 何时运行

- 新增或修改 `.cursor/rules/*.mdc`、`.cursor/skills/*/SKILL.md` 之后
- 合并分支或拉取他人改动涉及 `.cursor` 时
- 本地开发/发布前做一次快速自检

## 命令（仓库根目录）

```bash
python scripts/preflight_cursor.py
```

Windows 也可双击项目根目录下的 `preflight.bat`。

## 检查内容（摘要）

- **Rules**：存在 `description`；`alwaysApply` 或 `globs` 至少其一；正文非空
- **Skills**：每个技能目录含 `SKILL.md`；`name` / `description` 合法且不重复；正文非空；行数过多时 WARN（建议 ≤ 500 行）

## 退出码

- `0`：无 ERROR（可有 WARN）
- `1`：存在 ERROR，需修复后重跑
