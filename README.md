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

## 多用户 / 多设备访问

- **本机打开**：用 **`http://localhost:8501`**（或 `http://127.0.0.1:8501`）。**不要**在浏览器里输入 `http://0.0.0.0:8501`（0.0.0.0 仅表示“监听所有网卡”，不能当网址用）。
- **仅本机**：运行 `start.bat` 或 `streamlit run src/app.py` 即可。
- **局域网内其他电脑也要访问**：请运行 **`start-lan.bat`**（会监听 `0.0.0.0`）；本机仍用 `http://localhost:8501`，其他电脑用 `http://<本机IP>:8501`，并放行防火墙 8501 端口。
- **多会话**：`.streamlit/config.toml` 中已开启 `runner.fastReruns`，多浏览器可同时操作；若某会话在跑大批量训练/审核，其他会话可能短暂变慢，建议错峰。

## 服务控制（Windows）

| 脚本 | 说明 |
|------|------|
| `start.bat` | 启动 Web UI（本机访问，推荐用 localhost:8501） |
| `start-lan.bat` | 启动 Web UI（局域网可访问；本机仍用 localhost:8501 打开） |
| `stop.bat` | 停止运行中的服务（端口 8501/8000） |
| `restart.bat` | 先停止再在新窗口启动 |
