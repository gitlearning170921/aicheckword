"""文献检索 integration：供 aiword 调用，出站代理复用 llm_http_proxy。"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/integration/literature", tags=["integration-literature"])


class LiteratureSearchRequest(BaseModel):
    query: str = Field(..., description="检索式")
    sources: List[str] = Field(default_factory=lambda: ["pubmed"])
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    max_results_per_source: int = 200
    scholar_captcha_session_id: Optional[str] = None
    scholar_sort_by: Optional[str] = Field(
        default="relevance",
        description="Scholar 排序：relevance（相关性，默认）或 date（日期）",
    )
    scholar_start_offset: int = Field(
        default=0,
        description="Scholar 翻页起点（验证后继续检索时传入已抓条数，避免从 0 重抓）",
    )


@router.post("/search")
def literature_search(body: LiteratureSearchRequest) -> dict[str, Any]:
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    sources = [str(s or "").strip().lower() for s in (body.sources or []) if str(s or "").strip()]
    if not sources:
        raise HTTPException(status_code=400, detail="sources 至少选择 1 项")
    auto = [s for s in sources if s in ("pubmed", "scholar")]
    if not auto:
        raise HTTPException(status_code=400, detail="自动检索仅支持 pubmed / scholar")

    max_per = body.max_results_per_source or 200
    try:
        max_per = int(max_per)
    except (TypeError, ValueError):
        max_per = 200
    max_per = max(1, min(500, max_per))

    from src.core.literature_search import run_literature_search

    try:
        start_off = 0
        try:
            start_off = max(0, int(body.scholar_start_offset or 0))
        except (TypeError, ValueError):
            start_off = 0
        result = run_literature_search(
            query=query,
            sources=auto,
            start_year=body.start_year,
            end_year=body.end_year,
            max_results_per_source=max_per,
            scholar_captcha_session_id=(body.scholar_captcha_session_id or "").strip(),
            scholar_sort_by=(body.scholar_sort_by or "relevance").strip() or "relevance",
            scholar_start_offset=start_off,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"ok": True, **result}


@router.get("/scholar-captcha/{session_id}")
def scholar_captcha_entry(session_id: str, rewrite_base: str = ""):
    from src.core.scholar_captcha import proxy_scholar_captcha

    base = (rewrite_base or "").strip() or f"/api/integration/literature/scholar-captcha/{session_id}"
    status, headers, body, _ctype = proxy_scholar_captcha(
        session_id,
        method="GET",
        target_url="",
        rewrite_base=base,
    )
    return Response(content=body, status_code=status, media_type=headers.get("content-type"), headers={
        k: v for k, v in headers.items() if k.lower() != "content-type"
    })


@router.api_route("/scholar-captcha/{session_id}/nav", methods=["GET", "POST"])
async def scholar_captcha_nav(session_id: str, request: Request, u: str = "", rewrite_base: str = ""):
    from src.core.scholar_captcha import decode_nav_url, parse_form_body, proxy_scholar_captcha

    target = decode_nav_url(u)
    if not target:
        raise HTTPException(status_code=400, detail="missing u")
    base = (rewrite_base or "").strip() or f"/api/integration/literature/scholar-captcha/{session_id}"
    form_data = None
    if request.method.upper() == "POST":
        raw = await request.body()
        form_data = parse_form_body(raw, request.headers.get("content-type") or "")
        if not form_data:
            try:
                form = await request.form()
                form_data = {str(k): str(v) for k, v in form.items()}
            except Exception:
                form_data = {}
    status, headers, body, _ctype = proxy_scholar_captcha(
        session_id,
        method=request.method.upper(),
        target_url=target,
        form_data=form_data,
        rewrite_base=base,
    )
    return Response(
        content=body,
        status_code=status,
        media_type=headers.get("content-type"),
        headers={k: v for k, v in headers.items() if k.lower() != "content-type"},
    )
