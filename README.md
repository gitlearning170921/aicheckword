# 注册文档审核工具 (AI Check Word)

基于 RAG 的注册文档智能审核：训练法规/标准构建知识库，自动审核文档并输出审核报告。

- **Web UI**：`streamlit run src/app.py`
- **API 服务**：`python -m src.api.server`
- **详细说明**：[docs/README.md](docs/README.md)

## 目录说明

| 目录 | 说明 |
|------|------|
| **config/** | 配置（settings、环境变量示例在根目录 .env.example） |
| **docs/** | 文档（README、使用说明） |
| **src/** | 业务代码（Web 入口 app.py、核心 core、API api） |

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 后：
streamlit run src/app.py
```

## 系统配置与迁机（MySQL）

- 应用启动后，**除首次连接数据库外**，其余运行时配置可从表 `app_settings.runtime_settings_json` 自动加载（侧栏首次连库后生效）。
- 在 Web 端顶部 **「⚙️ 系统配置」** 可查看、编辑并**保存到数据库**；表单默认带入当前内存中的配置。
- **迁到新电脑**：新机器请先在本项目目录执行 **`pip install -r requirements.txt`**（需含 `langchain-core`、`langchain-text-splitters`；仅装旧版 `langchain` 会缺 `langchain.schema` 等模块）。`.env` 中至少配置 **MySQL 能连上原库**；启动后其余项会从库恢复。也可在系统配置页 **导出 .env 片段** 作备份。
- 修改 **MySQL 连接字段** 后需**重启应用**才会使用新连接。
- **多机共享向量库**：在一台服务器运行 Chroma Server，各客户端配置 `chroma_server_host`（见 [docs/chroma-shared.md](docs/chroma-shared.md)）。

## 多用户 / 多设备访问

- **本机打开**：用 **`http://localhost:8501`**（或 `http://127.0.0.1:8501`）。**不要**在浏览器里输入 `http://0.0.0.0:8501`（0.0.0.0 仅表示“监听所有网卡”，不能当网址用）。
- **仅本机**：运行 `start.bat` 或 `streamlit run src/app.py` 即可。
- **局域网内其他电脑也要访问**：请运行 **`start-lan.bat`**（会监听 `0.0.0.0`）；本机仍用 `http://localhost:8501`，其他电脑用 `http://<本机IP>:8501`，并放行防火墙 8501 端口。
- **多会话**：`.streamlit/config.toml` 中 `runner.fastReruns` 默认为 **false**，减轻查看/编辑报告时的整页重跑与焦点丢失；多浏览器仍可同时操作。若更在意侧栏切换的瞬时响应，可改为 `true`（报告页大量控件时可能更易感觉「自动刷新」）。

## 服务控制（Windows）

| 脚本 | 说明 |
|------|------|
| `start.bat` | 启动 Web UI（本机访问，推荐用 localhost:8501） |
| `start-lan.bat` | 启动 Web UI（局域网可访问；本机仍用 localhost:8501 打开） |
| `restart_api.bat` | 重启 API 服务（会杀占用端口的进程；默认端口 8000） |
| `start_all.bat` | 同时启动 Web + API（分别在窗口中运行） |
| `restart_all.bat` | 同时重启 Web + API |
| `stop.bat` | 停止运行中的服务（端口 8501/8000） |
| `stop_api.bat` | 仅停止 API 服务（默认端口 8000；不影响 Web UI） |
| `restart.bat` | 先停止再在新窗口启动 |
