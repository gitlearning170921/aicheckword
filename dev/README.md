# aicheckword — 开发机脚本与目录说明

与 **aiword**、**aiprintword** 采用相同 `dev/` 三层结构：

```
aicheckword/
├── src/                    ★ API / Streamlit 产品代码
├── config/ scripts/ docs/  ★ 配置、工具、文档
├── dev/
│   ├── git-no_tag/         ◆ 日常 commit+push（不打 tag）
│   ├── git-tag_release/    ◆ 说明：tag 由 aiword release.bat 统一打
│   └── local-run/          ◆ 本机启停 API / Streamlit
└── uploads/ knowledge_store/  ✗ 运行时数据
```

## 常用命令

```cmd
:: 日常提交（不打 tag）
dev\git-no_tag\commit.bat "API 调整"

:: 本机只跑 API（生产联调 aiword 时用）
dev\local-run\start_api.bat

:: Streamlit 知识库 UI
dev\local-run\start_streamlit.bat

:: Web + API
dev\local-run\start_all.bat

:: 发版（双仓库 tag）— 在 aiword 目录执行
cd ..\aiword
dev\git-tag_release\release.bat 1.0.5 "发版说明"
```

完整发版→打包→Linux 流程见 `..\aiword\dev\README.md`。
