#!/bin/bash
set -euo pipefail

mkdir -p /app/knowledge_store /app/uploads /app/training_docs

# 升级安全：不删除、不覆盖卷内已有数据
if [[ ! -f /app/.env ]] && [[ -f /app/.env.example ]]; then
  echo "[entrypoint] 提示: 运行时配置请通过 compose env_file / 环境变量注入（MySQL、LLM 等）。"
fi

exec "$@"
