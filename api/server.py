"""FastAPI 服务：将 Agent 能力暴露为 REST API，供其他项目调用"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from core.agent import ReviewAgent

app = FastAPI(
    title="注册文档审核 Agent API",
    description="基于 RAG 的注册文档审核服务，支持训练知识库和自动审核文档",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agents: Dict[str, ReviewAgent] = {}


def get_agent(collection: str = "regulations") -> ReviewAgent:
    if collection not in _agents:
        _agents[collection] = ReviewAgent(collection)
    return _agents[collection]


class TextReviewRequest(BaseModel):
    text: str
    file_name: str = "直接输入"
    collection: str = "regulations"


class KnowledgeQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection: str = "regulations"


class CollectionRequest(BaseModel):
    collection: str = "regulations"


# ============ 状态接口 ============

@app.get("/")
def root():
    return {"service": "注册文档审核 Agent", "version": "1.0.0", "status": "running"}


@app.get("/status")
def agent_status(collection: str = "regulations"):
    agent = get_agent(collection)
    return agent.get_status()


# ============ 训练接口 ============

@app.post("/train/upload")
async def train_upload(
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
):
    """上传文件进行训练"""
    agent = get_agent(collection)
    results = []

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = agent.train(tmp_path)
            result["original_filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "status": "error",
                "original_filename": file.filename,
                "message": str(e),
            })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    total_chunks = sum(r.get("chunks_added", 0) for r in results)
    return {
        "status": "success",
        "files_processed": len(results),
        "total_chunks_added": total_chunks,
        "details": results,
    }


@app.post("/train/directory")
def train_directory(dir_path: str = Form(...), collection: str = Form("regulations")):
    """从服务器目录训练"""
    if not Path(dir_path).exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{dir_path}")
    agent = get_agent(collection)
    result = agent.train(dir_path)
    return result


# ============ 审核接口 ============

@app.post("/review/upload")
async def review_upload(
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
):
    """上传文件进行审核"""
    agent = get_agent(collection)
    reports = []

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            report = agent.review(tmp_path)
            report["original_filename"] = file.filename
            reports.append(report)
        except Exception as e:
            reports.append({
                "file_name": file.filename,
                "error": str(e),
            })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return {"reports": reports, "total_files": len(reports)}


@app.post("/review/text")
def review_text(request: TextReviewRequest):
    """审核文本内容"""
    agent = get_agent(request.collection)
    report = agent.review_text(request.text, request.file_name)
    return report


# ============ 知识库接口 ============

@app.post("/knowledge/search")
def search_knowledge(request: KnowledgeQueryRequest):
    """查询知识库"""
    agent = get_agent(request.collection)
    results = agent.search_knowledge(request.query, request.top_k)
    return {"results": results}


@app.post("/knowledge/clear")
def clear_knowledge(request: CollectionRequest):
    """清空知识库"""
    agent = get_agent(request.collection)
    return agent.clear_knowledge()


@app.get("/knowledge/collections")
def list_collections():
    """列出所有知识库集合"""
    agent = get_agent()
    collections = agent.kb.list_collections()
    return {"collections": collections}


def start_server():
    """启动 API 服务"""
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    start_server()
