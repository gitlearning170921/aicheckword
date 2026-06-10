---
name: aiword Docker 部署
overview: 在 Linux 服务器上用 Docker Compose 部署 aiword 与 aicheckword（FastAPI API），连接外部 MySQL 与云端 LLM；aiprintword 保留在 Windows 服务器。镜像支持版本化升级，业务数据与配置通过外部 MySQL + 命名卷持久化，升级不丢数据。
todos:
  - id: dockerfile-aiword
    content: 编写 aiword/Dockerfile + entrypoint（gunicorn workers=1、卷目录、database_url.txt 引导）
    status: completed
  - id: dockerfile-aicheckword
    content: 编写 aicheckword/Dockerfile（uvicorn workers=1、系统依赖、持久化卷）
    status: completed
  - id: compose-env
    content: 编写 deploy/docker-compose.yml + .env.example（内外网 URL、MySQL、LLM、密钥模板）
    status: completed
  - id: nginx-docs
    content: 可选 Nginx 反代配置 + deploy/README.md 部署与联调文档
    status: completed
  - id: db-uri-hardening
    content: 弱化 webapp/__init__.py 硬编码默认 DB URI，避免 Docker 误连
    status: completed
  - id: smoke-test
    content: 部署后跑 exam_center_smoke / 手工验证 handoff 与初稿审核链路
    status: completed
  - id: upgrade-backup-scripts
    content: 编写 deploy/upgrade.sh 与 deploy/backup.sh（升级前备份、拉取新镜像、保留卷重建容器）
    status: completed
  - id: entrypoint-idempotent
    content: entrypoint 幂等：仅首次创建 database_url.txt / 目录，升级时不覆盖已有配置与数据
    status: completed
isProject: false
---

# aiword Docker 混合部署方案

## 架构总览

```mermaid
flowchart TB
  subgraph users [用户浏览器]
    Browser[浏览器]
  end

  subgraph linux [Linux 服务器 Docker]
    Nginx[Nginx 可选 TLS 反代]
    Aiword[aiword Flask :5000]
    Aicheckword[aicheckword FastAPI :8000]
  end

  subgraph external [外部依赖]
    MySQL[(MySQL 云/自建)]
    LLM[云端 LLM API]
  end

  subgraph windows [Windows 服务器]
    Aiprintword[aiprintword :5050]
  end

  Browser --> Nginx --> Aiword
  Browser -->|302 去签字| Aiprintword
  Aiword -->|HTTP 集成| Aicheckword
  Aiword -->|POST /api/handoff| Aiprintword
  Aiprintword -->|GET /api/integration/projects| Aiword
  Aiword --> MySQL
  Aicheckword --> MySQL
  Aicheckword --> LLM
```



**你已选择的策略：**

- **混合部署**：aiword + aicheckword 容器化上 Linux；aiprintword 留在 Windows（COM 批量打印完整可用）
- **外部 MySQL + 云端 LLM**：compose 不内置 MySQL/Ollama，镜像更小、运维更简单

---

## 1. 服务边界与端口


| 服务                    | 部署位置       | 端口   | 对外暴露           | 说明                              |
| --------------------- | ---------- | ---- | -------------- | ------------------------------- |
| aiword                | Docker     | 5000 | 是（经 Nginx 或直连） | 主 Web 入口                        |
| aicheckword API       | Docker     | 8000 | 可选（建议仅内网）      | 考试/初稿/审核/翻译集成                   |
| aicheckword Streamlit | 按需         | 8501 | 否              | **非 aiword 集成必需**；知识库训练/运维时单独启动 |
| aiprintword           | Windows 裸机 | 5050 | 是（内网/LAN）      | 签字 + 批量打印                       |
| MySQL                 | 外部         | 3306 | 否              | 建议 aiword / aicheckword 各用独立库   |


**集成契约（现有代码，无需改业务逻辑）：**

- aiword → aicheckword：`[webapp/_integration_common.py](F:/wzl/learning/python/aiword/webapp/_integration_common.py)` 解析 `AICHECKWORD_DRAFT_API_BASE` / `QUIZ_API_BASE_URL`，出站头含 `X-Aiword-Company-Id`（多租户时）
- aiword → aiprintword：`[webapp/routes.py](F:/wzl/learning/python/aiword/webapp/routes.py)` 的 `_aiprintword_go_redirect` POST `{AIPRINTWORD_BASE_URL}/api/handoff`
- aiprintword → aiword：`GET {AIWORD_BASE_URL}/api/integration/projects`（`[integration_routes.py](F:/wzl/learning/python/aiword/webapp/integration_routes.py)`）

---

## 2. 目录与文件布局（建议新增）

在 **aiword 仓库** 增加部署资产（编排入口），各服务 Dockerfile 放在各自仓库：

```
aiword/
  deploy/
    docker-compose.yml       # 编排 aiword + aicheckword
    .env.example             # 部署变量模板（不含真实密钥）
    nginx/nginx.conf         # 可选反代
  Dockerfile                 # aiword 镜像
  docker-entrypoint.sh       # 冷启动引导、目录初始化

aicheckword/
  Dockerfile                 # 仅 FastAPI API（默认 CMD）
  Dockerfile.streamlit       # 可选，运维训练用
  docker-entrypoint.sh
```

`docker-compose.yml` 通过 `build.context` 指向 sibling 目录（部署机需三仓库同级 checkout），或 CI 构建后 push 到私有 Registry 再 `image:` 引用。

---

## 3. 镜像构建要点

### 3.1 aiword 镜像

**基础镜像：** `python:3.11-slim-bookworm`

**依赖：** `[requirements.txt](F:/wzl/learning/python/aiword/requirements.txt)`（Flask、PyMySQL、APScheduler 等）

**生产进程（替换开发态 `[run_web.py](F:/wzl/learning/python/aiword/run_web.py)` 的 `debug=True`）：**

```bash
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 --access-logfile - "webapp:app"
```

- `**workers=1` 必须**：`[webapp/scheduler.py](F:/wzl/learning/python/aiword/webapp/scheduler.py)` 内 APScheduler 在多 worker 下会重复触发钉钉定时任务
- 超时 ≥600s：初稿/审核代理长请求

**持久化卷：**


| 挂载路径            | 用途                                   |
| --------------- | ------------------------------------ |
| `/app/instance` | `database_url.txt` 冷启动缓存、scheduler 锁 |
| `/app/uploads`  | 上传/备注附件（部分已迁 MySQL，仍建议保留）            |
| `/app/outputs`  | 生成结果                                 |


**冷启动数据库（关键）：**

`[resolve_database_uri()](F:/wzl/learning/python/aiword/webapp/app_settings.py)` **不读** 环境变量 `DATABASE_URL`，顺序为：内置默认 URI → `instance/database_url.txt` → MySQL `app_configs`。

Docker 首次部署二选一：

1. **推荐**：init 容器或 entrypoint 写入 `instance/database_url.txt`：
  ```
   mysql+pymysql://user:pass@mysql-host:3306/aiword?charset=utf8mb4
  ```
2. 从现网 mysqldump 恢复 `app_configs`（含 `DATABASE_URL`），同时拷贝 `instance/database_url.txt`

**首次配置种子：** 利用现有 `[ensure_environment_variables_migrated_to_db()](F:/wzl/learning/python/aiword/webapp/app_settings.py)`，在 compose 中为 aiword 注入环境变量（仅库中无值时写入）：

```env
QUIZ_API_BASE_URL=http://aicheckword:8000
AICHECKWORD_DRAFT_API_BASE=http://aicheckword:8000
AIPRINTWORD_BASE_URL=http://<windows-host-ip>:5050
AIPRINTWORD_HANDOFF_SECRET=<与 Windows 一致>
INTEGRATION_SECRET=<与 aiprintword AIWORD_INTEGRATION_SECRET 一致>
SECRET_KEY=<随机强密钥>
BASE_URL=https://aiword.yourdomain.com
FEATURE_MULTI_TENANT=0   # 按需
```

> **注意**：`[webapp/__init__.py](F:/wzl/learning/python/aiword/webapp/__init__.py)` 第 1413 行仍有硬编码默认 MySQL URI；Docker 部署应**务必**通过 `database_url.txt` 或库内配置覆盖，避免连错库。建议在 Dockerfile 阶段将默认 URI 改为占位符或文档化此风险（可作为后续小改动）。

### 3.2 aicheckword 镜像（API only）

**基础镜像：** `python:3.11-slim-bookworm`

**系统包（文档解析）：** `libgl1`、`libgomp1`（PyMuPDF）、可选 `tesseract-ocr`（扫描件 OCR）

**依赖：** `[requirements.txt](F:/wzl/learning/python/aicheckword/requirements.txt)`（`pywin32` 已有 `sys_platform == "win32"` 标记，Linux 可装）

**生产进程：**

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

- `**workers=1` 必须**：初稿/审核/翻译 job 存进程内存（`[server.py](F:/wzl/learning/python/aicheckword/src/api/server.py)` 文档要求）

**持久化卷：**


| 挂载路径                   | 用途            |
| ---------------------- | ------------- |
| `/app/knowledge_store` | Chroma 向量库    |
| `/app/uploads`         | 集成 job 工作区与导出 |
| `/app/training_docs`   | 训练文档（可选）      |


**环境变量（`.env` / compose）：**

```env
MYSQL_HOST=<外部 MySQL>
MYSQL_PORT=3306
MYSQL_DATABASE=aicheckword
MYSQL_USER=...
MYSQL_PASSWORD=...
PROVIDER=deepseek          # 或 tongyi / openai 等
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small   # 若 embedding 仍走 OpenAI 兼容端点
API_HOST=0.0.0.0
API_PORT=8000
CHROMA_PERSIST_DIR=/app/knowledge_store
UPLOADS_DIR=/app/uploads
```

迁机时同步：**MySQL dump + `knowledge_store/` + `uploads/`**（见 aicheckword `.env.example` 注释）。

---

## 4. docker-compose 核心结构

```yaml
services:
  aiword:
    build: ../aiword
    ports: ["5000:5000"]
    env_file: .env
    volumes:
      - aiword_instance:/app/instance
      - aiword_uploads:/app/uploads
      - aiword_outputs:/app/outputs
    depends_on:
      aicheckword:
        condition: service_healthy
    networks: [appnet]
    restart: unless-stopped

  aicheckword:
    build: ../aicheckword
    expose: ["8000"]          # 不对公网暴露时可去掉 ports
    env_file: .env.aicheckword
    volumes:
      - aicheck_knowledge:/app/knowledge_store
      - aicheck_uploads:/app/uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      retries: 3
    networks: [appnet]
    restart: unless-stopped

networks:
  appnet:
    driver: bridge
```

aiword 容器内 upstream 地址用 `**http://aicheckword:8000**`（Docker DNS），不要用 `127.0.0.1`。

**compose 必须声明命名卷（已在上方示例），并在文件末尾显式列出：**

```yaml
volumes:
  aiword_instance:
  aiword_uploads:
  aiword_outputs:
  aicheck_knowledge:
  aicheck_uploads:
```

> 可选：生产环境将命名卷改为**宿主机 bind mount**（如 `/data/aiword/instance`），便于直接用 `rsync`/备份工具；原则相同——数据路径与镜像/容器生命周期分离。

---

## 4.1 数据持久化与「可升级不丢数」设计（核心）

### 数据分层

```mermaid
flowchart LR
  subgraph ephemeral [可丢弃 随镜像替换]
    Image[应用代码与依赖]
    Container[容器文件系统]
  end

  subgraph persistent [必须保留 升级不动]
    MySQL[(外部 MySQL)]
    VolInstance[卷 aiword_instance]
    VolUploads[卷 uploads / outputs]
    VolKnowledge[卷 knowledge_store]
    EnvFile[宿主机 .env]
  end

  Image --> Container
  Container -.挂载.-> VolInstance
  Container -.挂载.-> VolUploads
  Container -.挂载.-> VolKnowledge
  Container --> MySQL
```

| 层级 | 存放位置 | 升级时 | 内容 |
|------|----------|--------|------|
| **业务主数据** | 外部 MySQL | 不动 | 用户、项目、模板 BLOB、app_configs、审核/考试记录等 |
| **aicheckword 元数据** | 外部 MySQL | 不动 | 知识库块元数据、公司映射、操作日志 |
| **向量索引** | 卷 `aicheck_knowledge` | 保留卷 | Chroma 持久化目录 |
| **文件作业区** | 卷 `aicheck_uploads` / `aiword_uploads` | 保留卷 | 初稿/审核导出、交接中间文件 |
| **冷启动与锁** | 卷 `aiword_instance` | 保留卷 | `database_url.txt`、`scheduler_locks/` |
| **部署配置** | 宿主机 `deploy/.env` | 不动 | 密钥、MySQL 地址、镜像版本号 |
| **应用代码** | 镜像内 | **每次升级替换** | Python 代码、静态资源 |

**原则：镜像只打包代码；一切状态落 MySQL 或命名卷。**

### 镜像版本化

- 构建时打**语义化标签**：`registry.example.com/aiword:1.4.2`、`aicheckword:1.4.2`
- `deploy/.env` 中固定版本，便于回滚：
  ```env
  AIWORD_IMAGE=registry.example.com/aiword:1.4.2
  AICHECKWORD_IMAGE=registry.example.com/aicheckword:1.4.2
  ```
- compose 使用 `image: ${AIWORD_IMAGE}`，**不用** `latest` 作为生产默认（可额外打 `latest` 仅供开发）
- 同版本 aiword / aicheckword 联调记录写入 `deploy/CHANGELOG.md` 或 Release Notes

### entrypoint 幂等（升级安全）

[`docker-entrypoint.sh`](F:/wzl/learning/python/aiword/docker-entrypoint.sh) 必须遵守：

1. **仅当不存在时**写入 `instance/database_url.txt`（读自环境变量 `AIWORD_BOOTSTRAP_DATABASE_URL`）；已存在则跳过，避免覆盖生产连接串
2. **仅当目录为空时**初始化 uploads/outputs 占位；不删除已有文件
3. 启动前清理 `instance/scheduler_locks/*.lock`（与现网 [`webapp/__init__.py`](F:/wzl/learning/python/aiword/webapp/__init__.py) 行为一致），避免旧锁导致定时任务跳过——这是**预期行为**，不丢业务数据
4. **不在 entrypoint 里**执行 `ensure_environment_variables_migrated_to_db` 的全量覆盖；该逻辑已在应用启动时「仅空项补全」

### 应用层自动 schema 迁移

aiword 每次启动会执行 [`ensure_schema()`](F:/wzl/learning/python/aiword/webapp/__init__.py) + `db.create_all()`，新增列/表**向前兼容**升级，无需手工跑 SQL（与现网 Windows 部署一致）。

aicheckword 配置以 MySQL `runtime_settings_json` 为主；升级后首次连库即生效，Chroma 索引仍在卷内，**不需重建向量库**（除非版本说明要求 reindex）。

### 升级标准流程（`deploy/upgrade.sh`）

```bash
# 1. 升级前备份（见下节）
./deploy/backup.sh

# 2. 修改 .env 中的 AIWORD_IMAGE / AICHECKWORD_IMAGE 为新版本

# 3. 拉取新镜像并重建容器（保留卷）
docker compose -f deploy/docker-compose.yml pull
docker compose -f deploy/docker-compose.yml up -d --no-deps --force-recreate aiword aicheckword

# 4. 看日志确认 schema 迁移与健康检查通过
docker compose -f deploy/docker-compose.yml logs -f --tail=100 aiword aicheckword
```

**禁止在生产升级时使用：**

```bash
docker compose down -v    # -v 会删除命名卷，导致 uploads/knowledge_store/instance 全丢
docker system prune -a --volumes
```

`docker compose down`（无 `-v`）只停容器，**卷保留**，可安全使用。

### 升级前备份（`deploy/backup.sh`）

每次升级前自动或手工执行：

| 备份项 | 命令/方式 |
|--------|-----------|
| MySQL aiword 库 | `mysqldump aiword > backup/aiword_YYYYMMDD.sql` |
| MySQL aicheckword 库 | `mysqldump aicheckword > backup/aicheckword_YYYYMMDD.sql` |
| Docker 卷 | `docker run --rm -v aiword_instance:/data -v $(pwd)/backup:/backup alpine tar czf /backup/aiword_instance.tgz /data`（各卷同理） |
| 宿主机 `.env` | `cp deploy/.env backup/.env.YYYYMMDD` |

备份目录 `deploy/backup/` 加入 `.gitignore`，不进镜像。

### 回滚

1. 将 `.env` 中镜像 tag 改回上一版本
2. `docker compose pull && docker compose up -d --force-recreate`
3. 若新版本已执行破坏性 schema 变更（极少见），从备份 SQL 恢复对应库

### 升级时的已知短暂影响

| 现象 | 原因 | 是否丢数据 |
|------|------|------------|
| 升级过程中 Web 短暂不可用 | 容器重建 | 否 |
| 进行中的 aicheckword 初稿/审核/翻译 job 失败 | job 状态在 API 进程内存，单 worker 重启即失 | **进行中的任务会丢**；已完成导出在 `uploads/` 卷内仍保留 |
| 钉钉定时任务同一分钟可能重复或跳过 | scheduler 锁在重启时清空 | 否（业务数据不受影响） |

**建议**：升级安排在低峰期；升级前在页面确认无长时间 running 的集成 job（或接受个别任务需重跑）。

### Windows aiprintword

aiprintword 不在 Docker 升级范围内，独立发版；只要 `AIWORD_BASE_URL` / handoff 密钥不变，Linux 侧 aiword 升级不影响签字链路。

---

## 5. Windows aiprintword 侧配置

在现有 Windows 服务器继续 `python app.py`（端口 5050 单实例自检，见 [startup-port-single-instance 规则](F:/wzl/learning/python/aiprintword/.cursor/rules/startup-port-single-instance.mdc)）。

`**.env` / 页面设置：**

```env
AIWORD_BASE_URL=https://aiword.yourdomain.com    # 或 http://linux-ip:5000
AIWORD_HANDOFF_SECRET=<与 aiword AIPRINTWORD_HANDOFF_SECRET 相同>
AIWORD_INTEGRATION_SECRET=<与 aiword INTEGRATION_SECRET 相同>
```

**网络要求：**


| 方向                                 | 说明                                      |
| ---------------------------------- | --------------------------------------- |
| Linux aiword → Windows:5050        | 服务端 handoff POST，需 Linux 能访问 Windows IP |
| 用户浏览器 → Windows:5050               | 302 跳转签字页，需用户能访问同一地址                    |
| Windows aiprintword → Linux aiword | 项目同步 API，需 Windows 能访问 aiword 对外 URL    |


因此 `AIPRINTWORD_BASE_URL` 应填 **用户与 Linux 服务器都能访问的 Windows 地址**（内网 IP 或 DNS），不能填 `127.0.0.1`。

**防火墙：** 开放 Windows 5050（限内网/VPN）；Linux 5000/443 按同样策略。

---

## 6. 可选 Nginx 反代（生产推荐）

- 对外只暴露 443，`proxy_pass` 到 `aiword:5000`
- 设置 `client_max_body_size 30m`（aiword `MAX_CONTENT_LENGTH=25MB`）
- 长连接/超时：`proxy_read_timeout 600s`（初稿/审核轮询）
- **不要**把 aicheckword:8000 直接暴露公网（除非加独立鉴权）；aiword 已做 BFF 代理
- TLS：Let's Encrypt 或企业证书

设置 aiword `BASE_URL=https://aiword.yourdomain.com`（钉钉催办、外链）。

---

## 7. 部署步骤（实施顺序）

### 阶段 A：准备外部 MySQL

1. 创建库 `aiword`、`aicheckword`（utf8mb4）
2. 若从 Windows 迁机：`mysqldump` 两库并导入
3. 记录连接串供 `database_url.txt` 使用

### 阶段 B：构建并启动容器

1. 在 Linux 克隆三仓库（或仅 aiword + aicheckword + deploy 资产）
2. 复制 `deploy/.env.example` → `.env`，填写 MySQL、LLM、密钥、Windows aiprintword 地址
3. `docker compose -f deploy/docker-compose.yml up -d --build`
4. 查看日志确认 aiword 完成 schema 迁移与环境变量入库

### 阶段 C：页面3 系统配置核对

登录 aiword → 页面3，确认：

- `DATABASE_URL`、`QUIZ_API_BASE_URL`、`AICHECKWORD_DRAFT_API_BASE`
- `AIPRINTWORD_BASE_URL`、`AIPRINTWORD_HANDOFF_SECRET`、`INTEGRATION_SECRET`
- `BASE_URL`、钉钉相关项

保存后重启 aiword 容器。

### 阶段 D：Windows aiprintword 联调

1. 启动 aiprintword，确认 5050 单实例
2. 配置 `AIWORD_BASE_URL` 指向 Linux aiword
3. 测试链路：
  - aiword 页面1 →「去签字」→ 跳转 aiprintword 签字页
  - aiword 初稿/审核/翻译页 → 上游 health
  - aiprintword 签字页 →「同步项目」

### 阶段 E：回归脚本

```bash
# aiword 多租户（若开启）
cd aiword && python scripts/multi_tenant_smoke.py

# 考试中心（需 API 可达）
python scripts/exam_center_smoke.py
```

---

## 8. 健康检查与监控


| 端点                                        | 服务             |
| ----------------------------------------- | -------------- |
| `GET /api/integration/health`             | aiword 开放集成    |
| `GET /health` 或 `/api/integration/health` | aicheckword    |
| `GET http://windows:5050/`                | aiprintword 存活 |


建议 compose `healthcheck` + 日志集中（json-file 或 Loki），磁盘监控 aicheckword `uploads/` 与 `knowledge_store/` 增长。

---

## 9. 安全清单

- 所有 `*_SECRET`、LLM API Key 只通过 `.env`/密钥管理注入，不入镜像
- MySQL 仅允许 Linux 服务器 IP 访问
- aiword `SECRET_KEY`、`INTEGRATION_SECRET` 生产必须更换
- 多租户开启时验证 `X-Aiword-Company-Id` 与公司同步（`[company_routes._sync_org_to_aicheckword](F:/wzl/learning/python/aiword/webapp/company_routes.py)`）
- 容器以非 root 用户运行（Dockerfile `USER app`）

---

## 10. 后续可选增强（非首期必须）


| 项                        | 说明                                                        |
| ------------------------ | --------------------------------------------------------- |
| aicheckword Streamlit 容器 | 知识库训练 UI，按需 `docker compose --profile admin up streamlit` |
| 私有 Registry              | CI 构建 push，生产 `docker pull`                               |
| aiword 去硬编码默认 DB URI     | 改为必须从 `database_url.txt`/环境引导，降低误连风险                      |
| aiprintword Linux 签字容器   | 若将来 Windows 退役，可单独容器化签字模块（打印仍留 Windows）                   |


---

## 11. 预期交付物（编码阶段）

1. `[aiword/Dockerfile](F:/wzl/learning/python/aiword/Dockerfile)` + `docker-entrypoint.sh`（幂等、非 root）
2. `[aicheckword/Dockerfile](F:/wzl/learning/python/aicheckword/Dockerfile)` + entrypoint
3. `[aiword/deploy/docker-compose.yml](F:/wzl/learning/python/aiword/deploy/docker-compose.yml)` + `.env.example`（含 `AIWORD_IMAGE` / `AICHECKWORD_IMAGE` 版本变量）
4. `[aiword/deploy/upgrade.sh](F:/wzl/learning/python/aiword/deploy/upgrade.sh)` + `[backup.sh](F:/wzl/learning/python/aiword/deploy/backup.sh)`（升级与备份一键脚本）
5. `[aiword/deploy/README.md](F:/wzl/learning/python/aiword/deploy/README.md)`（部署、**升级**、回滚、禁止 `down -v` 警示）
6. 可选 `[aiword/deploy/nginx/nginx.conf](F:/wzl/learning/python/aiword/deploy/nginx/nginx.conf)`
7. 小改动：aiword 生产启动脚本或 `run_web.py` 支持 `AIWORD_USE_GUNICORN=1`；移除/弱化硬编码 DB 默认 URI

