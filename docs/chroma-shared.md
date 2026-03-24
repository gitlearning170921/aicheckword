# 多机共享 Chroma 向量库

默认情况下，向量数据在每台机器本地的 `chroma_persist_dir`（如 `./knowledge_store`）。若希望 **多台电脑共用同一套知识库**，在一台服务器上运行 **Chroma 的 HTTP 服务**，各客户端在配置中填写 **`chroma_server_host`**（非空即启用远程模式）。

## 服务端（任选一种）

### Docker（常见）

```bash
docker run -d --name chroma -p 8000:8000 chromadb/chroma
```

默认监听 `8000`。生产环境请配合防火墙、内网或反向代理，**不要**把无鉴权的 Chroma 端口直接暴露公网。

### Python（与项目依赖一致）

可参考 [Chroma 官方文档](https://docs.trychroma.com/) 中「Running Chroma Server」部分，使用与客户端兼容的 chromadb 版本。

## 客户端（本应用）

在 **「⚙️ 系统配置」** 或 `.env` 中设置：

| 变量 | 说明 |
|------|------|
| `CHROMA_SERVER_HOST` | Chroma 服务主机，如 `10.0.0.5`；**留空** 则仍用本机目录 |
| `CHROMA_SERVER_PORT` | 端口，默认 `8000` |
| `CHROMA_SERVER_SSL` | `true` 时使用 `https://` |
| `CHROMA_SERVER_HEADERS_JSON` | 可选，JSON 字符串，如自建网关要求 `Authorization` 头 |

保存到数据库后，应用会 **重置 Chroma 连接缓存**；若仍异常，**重启 Streamlit**。

## 重要说明

1. **嵌入模型一致**：各客户端 `embedding_model` 与训练时一致，否则向量空间不匹配，检索质量会异常。
2. **Ollama 位置**：若用 Ollama 做嵌入，每台客户端需能访问同一 `ollama_base_url`（或各自部署同模型）。
3. **MySQL**：文档块元数据仍双写到 MySQL；Chroma 存向量与索引。迁机时 MySQL +（远程 Chroma 或备份的 Chroma 数据）需一致。
4. **并发**：多机同时写入同一 Chroma 集合时，请控制并发或错峰大批量训练，避免服务端压力过大。
