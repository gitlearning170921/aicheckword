"""Agent 封装：三步流程
  第一步：训练法规/程序/案例 → 生成审核点清单
  第二步：训练审核点（将审核点清单向量化）+ 项目专属资料
  第三步：基于审核点知识库 + 可选项目专属 审核文档
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from .knowledge_base import KnowledgeBase
from .reviewer import DocumentReviewer, AuditReport
from .checklist_generator import ChecklistGenerator
from .db import (
    get_project,
    get_knowledge_stats,
    get_checkpoint_stats,
    get_project_knowledge_text,
    update_project_basic_info,
    update_project_system_functionality,
    get_review_extra_instructions,
    get_review_system_prompt,
    get_review_user_prompt,
)
from .document_loader import load_single_file, split_documents


# 固定的集合名前缀
COLLECTION_REGULATIONS = "regulations"
COLLECTION_CHECKPOINTS = "checkpoints"


class ReviewAgent:
    def __init__(self, collection_name: str = "regulations"):
        self.collection_name = collection_name
        self.checkpoint_collection = f"{collection_name}_checkpoints"

        self.kb = KnowledgeBase(collection_name)
        self.checkpoint_kb = KnowledgeBase(
            self.checkpoint_collection,
            base_collection=collection_name,
            is_checkpoint=True,
        )
        # 误报/人工纠正单独入库（knowledge_docs + 独立 Chroma），与第二步「审核点清单」向量库区分
        self.audit_feedback_collection = f"{collection_name}_audit_feedback"
        self.checkpoint_feedback_kb = KnowledgeBase(
            self.audit_feedback_collection,
            base_collection=collection_name,
            is_checkpoint=False,
        )

        self._reviewer: Optional[DocumentReviewer] = None
        self._checklist_gen: Optional[ChecklistGenerator] = None
        self._project_kbs: Dict[int, KnowledgeBase] = {}

    @property
    def reviewer(self) -> DocumentReviewer:
        """审核器使用审核点知识库（第二步训练的结果）"""
        if self._reviewer is None:
            self._reviewer = DocumentReviewer(
                knowledge_base=self.checkpoint_kb,
                feedback_knowledge_base=self.checkpoint_feedback_kb,
                collection_name=self.collection_name,
            )
        return self._reviewer

    @property
    def checklist_generator(self) -> ChecklistGenerator:
        """审核点生成器使用法规知识库（第一步训练的结果）"""
        if self._checklist_gen is None:
            self._checklist_gen = ChecklistGenerator(knowledge_base=self.kb)
        return self._checklist_gen

    def reset_clients(self):
        self.kb.reset_clients()
        self.checkpoint_kb.reset_clients()
        self.checkpoint_feedback_kb.reset_clients()
        for pkb in self._project_kbs.values():
            pkb.reset_clients()
        self._project_kbs.clear()
        if self._reviewer is not None:
            self._reviewer.reset_client()
        self._reviewer = None
        if self._checklist_gen is not None:
            self._checklist_gen.reset_client()
        self._checklist_gen = None

    def get_project_kb(self, project_id: int) -> KnowledgeBase:
        """获取项目专属向量库（每个项目独立集合）"""
        if project_id not in self._project_kbs:
            coll = f"{self.collection_name}_project_{project_id}"
            self._project_kbs[project_id] = KnowledgeBase(
                collection_name=coll,
                project_id=project_id,
                base_collection=self.collection_name,
            )
        return self._project_kbs[project_id]

    # ─── 第一步：训练法规 / 程序 / 案例 ────────────────────

    def train(self, file_path: str, category: str = "regulation") -> dict:
        path = Path(file_path)
        if path.is_dir():
            count = self.kb.train_from_directory(path, category=category)
            return {"status": "success", "chunks_added": count, "source": str(path), "category": category}
        elif path.is_file():
            count = self.kb.train_from_file(path, category=category)
            return {"status": "success", "chunks_added": count, "source": path.name, "category": category}
        else:
            return {"status": "error", "message": f"路径不存在：{file_path}"}

    def train_batch(self, file_paths: List[str], category: str = "regulation") -> List[dict]:
        results = []
        for fp in file_paths:
            result = self.train(fp, category=category)
            results.append(result)
        return results

    # ─── 第一步 → 生成审核点清单 ────────────────────────

    def generate_checklist(
        self,
        base_checklist: Optional[str] = None,
        query_hints: List[str] = None,
        provider: Optional[str] = None,
        generate_prompt_override: Optional[str] = None,
        optimize_prompt_override: Optional[str] = None,
        document_language: Optional[str] = None,
        kb_stats: Optional[Dict[str, Any]] = None,
        registration_countries: Optional[List[str]] = None,
        registration_type: Optional[str] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """registration_countries 可选；选中国家时按「国家→额外关键词」扩展检索。registration_type 可选；传入时按注册类别注入审核尺度（Ⅲ>Ⅱb>Ⅱa>Ⅱ>Ι）。"""
        if kb_stats is None:
            kb_stats = get_knowledge_stats(self.collection_name)
        return self.checklist_generator.generate_checklist(
            base_checklist, query_hints, provider=provider,
            generate_prompt_override=generate_prompt_override,
            optimize_prompt_override=optimize_prompt_override,
            document_language=document_language,
            kb_stats=kb_stats,
            registration_countries=registration_countries,
            registration_type=registration_type,
            progress_callback=progress_callback,
        )

    # ─── 第二步：将审核点清单训练入审核点知识库 ────────────

    def train_checklist(self, checklist: List[Dict[str, Any]], callback=None, file_name: str = "审核点清单") -> int:
        docs = self.checklist_generator.checklist_to_documents(checklist)
        name = (file_name or "审核点清单").strip() or "审核点清单"
        if callback:
            return self.checkpoint_kb.add_documents_with_progress(
                docs, batch_size=12, callback=callback, file_name=name
            )
        return self.checkpoint_kb.add_documents(docs, file_name=name)

    # ─── 第二步（续）：项目专属资料训练 ────────────────────

    def train_project_docs(
        self,
        project_id: int,
        file_path: str,
        file_name: str = "",
        callback=None,
        on_loading=None,
    ) -> int:
        """将单个文件训练到项目专属向量库。on_loading(msg) 在加载/分块阶段回调，便于 UI 显示进度。"""
        from .document_loader import load_and_split
        if on_loading:
            on_loading("加载文件中...")
        chunks = load_and_split(file_path)
        if on_loading:
            on_loading(f"已分块 {len(chunks)} 块，正在向量化...")
        name = file_name or Path(file_path).name
        pkb = self.get_project_kb(project_id)
        if callback:
            return pkb.add_documents_with_progress(chunks, batch_size=12, callback=callback, file_name=name)
        return pkb.add_documents(chunks, file_name=name)

    def extract_and_save_project_basic_info(
        self, project_id: int, provider: Optional[str] = None
    ) -> str:
        """从项目知识库中已入库的文本提取「项目基本信息」，若知识库中无此条则训练后写入 projects.basic_info_text，供审核时与待审文档一致性核对。返回提取出的文本。"""
        from config import settings
        text = get_project_knowledge_text(project_id, max_chars=15000)
        if not text or len(text.strip()) < 50:
            update_project_basic_info(project_id, "")
            return ""
        use_cursor = (provider or getattr(settings, "provider", None) or "").lower() == "cursor"
        from .db import get_prompt_by_key
        _default_basic = """从以下项目资料（如技术要求、说明书等）中，提取用于审核时与待审文档做一致性核对的基本信息。

## 项目资料摘要

{text}

## 要求

请仅输出以下项目的关键信息，每行一项，格式示例：
项目名称：xxx
产品名称：xxx
型号规格：xxx
注册单元名称：xxx
（若某类信息在资料中未出现可省略该行）

不要输出其他说明，仅输出上述格式的若干行。"""
        tpl = (get_prompt_by_key("project_basic_info_prompt") or "").strip() or _default_basic
        prompt = tpl.format(text=text[:12000])

        try:
            if use_cursor:
                from .cursor_agent import complete_task
                out = complete_task(prompt)
            else:
                msg = self.reviewer.llm.invoke(prompt)
                out = getattr(msg, "content", str(msg))
            out = (out or "").strip()
            if out:
                update_project_basic_info(project_id, out)
            return out
        except Exception:
            update_project_basic_info(project_id, "")
            return ""

    def identify_system_functionality_from_package(
        self, project_id: int, file_path: str, provider: Optional[str] = None
    ) -> str:
        """从安装包/压缩包识别系统功能并写入项目，供审核时与文档一致性核对。返回识别出的功能描述。"""
        from .system_functionality import extract_package_info, identify_system_functionality_with_llm
        raw = extract_package_info(file_path)
        if not raw or raw.startswith("文件不存在") or raw.startswith("请求失败"):
            return raw
        source_hint = "安装包/压缩包（文件列表与包内可读文本）"
        text = identify_system_functionality_with_llm(raw, source_hint, provider=provider)
        if text:
            update_project_system_functionality(project_id, text, "package")
        return text or "未能识别出系统功能描述。"

    def identify_system_functionality_from_url(
        self,
        project_id: int,
        url: str,
        username: str = "",
        password: str = "",
        provider: Optional[str] = None,
        captcha: str = "",
    ) -> str:
        """从 URL（可选账号密码、可选验证码）抓取页面并识别系统功能，写入项目。返回识别出的功能描述。"""
        from .system_functionality import fetch_url_content, identify_system_functionality_with_llm
        raw = fetch_url_content(url, username, password, captcha=captcha)
        if not raw or raw.startswith("未填写") or raw.startswith("请求失败"):
            return raw
        if raw.startswith("CAPTCHA_REQUIRED"):
            return raw
        source_hint = "系统/网页 URL 抓取内容"
        text = identify_system_functionality_with_llm(raw, source_hint, provider=provider)
        if text:
            update_project_system_functionality(project_id, text, "url")
        return text or "未能识别出系统功能描述。"

    # ─── 第三步：基于审核点知识库（+ 可选项目专属）审核文档 ────────────────

    def _get_project_context(self, project_id: int, document_text: str, top_k: int = 10) -> str:
        pkb = self.get_project_kb(project_id)
        docs = pkb.search(document_text[:2000], top_k=top_k)
        if not docs:
            return ""
        return "\n\n".join(
            f"【项目资料】来源：{d.metadata.get('source_file', '未知')}\n{d.page_content}" for d in docs
        )

    def _build_review_context(
        self,
        project_id: Optional[int],
        override: Optional[Dict] = None,
        extra_instructions_override: Optional[str] = None,
    ) -> Optional[dict]:
        """构建审核维度：优先用 override，否则从项目带出。含项目名称、自定义审核提示词，供审核时与项目资料中的基本信息一致性检查。
        extra_instructions_override: 若传入则用其作为自定义审核要求，否则用 get_review_extra_instructions()。"""
        extra = extra_instructions_override if extra_instructions_override is not None else get_review_extra_instructions()
        if override:
            override = dict(override)
            if extra and "extra_instructions" not in override:
                override["extra_instructions"] = extra
            # 按项目审核时 override 仅含维度，需合并项目的名称、基本信息、适用范围、系统功能等
            if project_id:
                proj = get_project(project_id)
                if proj:
                    if "project_name" not in override:
                        override["project_name"] = proj.get("name") or ""
                    if "basic_info_text" not in override:
                        override["basic_info_text"] = proj.get("basic_info_text") or ""
                    if "system_functionality_text" not in override:
                        override["system_functionality_text"] = proj.get("system_functionality_text") or ""
                    if "scope_of_application" not in override:
                        override["scope_of_application"] = proj.get("scope_of_application") or ""
                    if "model" not in override:
                        override["model"] = proj.get("model") or ""
                    if "model_en" not in override:
                        override["model_en"] = proj.get("model_en") or ""
            return override
        if not project_id:
            if extra:
                return {"extra_instructions": extra}
            return None
        proj = get_project(project_id)
        if not proj:
            return None
        ctx = {
            "project_name": proj.get("name") or "",
            "project_name_en": proj.get("name_en") or "",
            "product_name": proj.get("product_name") or "",
            "product_name_en": proj.get("product_name_en") or "",
            "model": proj.get("model") or "",
            "model_en": proj.get("model_en") or "",
            "basic_info_text": proj.get("basic_info_text") or "",
            "system_functionality_text": proj.get("system_functionality_text") or "",
            "scope_of_application": proj.get("scope_of_application") or "",
            "registration_country": proj.get("registration_country") or "",
            "registration_country_en": proj.get("registration_country_en") or "",
            "registration_type": proj.get("registration_type") or "",
            "registration_component": proj.get("registration_component") or "",
            "project_form": proj.get("project_form") or "",
        }
        if extra:
            ctx["extra_instructions"] = extra
        return ctx

    def review(
        self,
        file_path: str,
        project_id: Optional[int] = None,
        review_context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        display_file_name: Optional[str] = None,
    ) -> dict:
        """system_prompt / user_prompt / extra_instructions 为本次覆盖，不传则用数据库或内置默认。
        display_file_name：上传/展示用文件名；批量审核时传入可避免历史报告里出现临时文件路径名。"""
        project_context_text = ""
        if project_id:
            try:
                docs = load_single_file(file_path)
                doc_text = "\n\n".join(d.page_content for doc in docs)
            except Exception:
                doc_text = ""
            project_context_text = self._get_project_context(project_id, doc_text)
        ctx = self._build_review_context(
            project_id, review_context, extra_instructions_override=extra_instructions
        )
        sys_p = system_prompt if (system_prompt is not None and system_prompt.strip()) else get_review_system_prompt()
        usr_p = user_prompt if (user_prompt is not None and user_prompt.strip()) else get_review_user_prompt()
        report = self.reviewer.review_file(
            file_path,
            review_context=ctx,
            project_context_text=project_context_text or None,
            system_prompt=sys_p,
            user_prompt=usr_p,
        )
        out = report.to_dict()
        disp = (display_file_name or "").strip()
        if disp:
            out["file_name"] = disp
            out["original_filename"] = disp
        return out

    def review_text(
        self,
        text: str,
        file_name: str = "直接输入",
        project_id: Optional[int] = None,
        review_context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> dict:
        """system_prompt / user_prompt / extra_instructions 为本次覆盖，不传则用数据库或内置默认。"""
        project_context_text = ""
        if project_id:
            project_context_text = self._get_project_context(project_id, text)
        ctx = self._build_review_context(
            project_id, review_context, extra_instructions_override=extra_instructions
        )
        sys_p = system_prompt if (system_prompt is not None and system_prompt.strip()) else get_review_system_prompt()
        usr_p = user_prompt if (user_prompt is not None and user_prompt.strip()) else get_review_user_prompt()
        report = self.reviewer.review_text(
            text,
            file_name,
            review_context=ctx,
            project_context_text=project_context_text or None,
            system_prompt=sys_p,
            user_prompt=usr_p,
        )
        return report.to_dict()

    def review_batch(
        self,
        file_paths: List[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> List[dict]:
        """system_prompt / user_prompt / extra_instructions 为本次覆盖，不传则用数据库或内置默认。"""
        sys_p = system_prompt if (system_prompt is not None and system_prompt.strip()) else get_review_system_prompt()
        usr_p = user_prompt if (user_prompt is not None and user_prompt.strip()) else get_review_user_prompt()
        reports = self.reviewer.review_multiple_files(
            file_paths, system_prompt=sys_p, user_prompt=usr_p
        )
        return [r.to_dict() for r in reports]

    def review_multi_document_consistency(
        self,
        items: List[tuple],
        project_id: Optional[int] = None,
        review_context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        多文档一致性与模板风格审核。items = [(path, display_name), ...]，path 为文件路径，会读取内容。
        返回与单文档审核相同结构的 report 字典，file_name 为「多文档一致性与模板风格审核」。
        """
        from .document_loader import load_single_file
        doc_list = []
        for path, display_name in items:
            try:
                docs = load_single_file(path)
                text = "\n\n".join(d.page_content for d in docs)
            except Exception:
                text = ""
            doc_list.append((display_name, text))
        ctx = self._build_review_context(project_id, review_context) if (project_id or review_context) else None
        report = self.reviewer.review_multi_document_consistency(doc_list, review_context=ctx)
        return report.to_dict()

    # ─── 通用查询 ────────────────────────────────────

    def search_knowledge(self, query: str, top_k: int = 5, use_checkpoints: bool = False) -> List[dict]:
        kb = self.checkpoint_kb if use_checkpoints else self.kb
        docs = kb.search(query, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "未知"),
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def get_status(self) -> dict:
        """统计一律以数据库为准（knowledge_docs / checkpoint_docs）"""
        reg_stats = get_knowledge_stats(self.collection_name)
        cp_stats = get_checkpoint_stats(self.collection_name)
        fb_stats = get_knowledge_stats(self.audit_feedback_collection)
        return {
            "agent_name": "注册文档审核Agent",
            "collection_name": self.collection_name,
            "regulations_kb": {
                "collection_name": self.collection_name,
                "document_count": reg_stats.get("total_chunks", 0),
            },
            "checkpoints_kb": {
                "collection_name": self.checkpoint_collection,
                "document_count": cp_stats.get("total_chunks", 0),
            },
            "audit_feedback_kb": {
                "collection_name": self.audit_feedback_collection,
                "document_count": fb_stats.get("total_chunks", 0),
            },
            "capabilities": [
                "第一步：法规/程序/案例训练 + 生成审核点",
                "第二步：审核点训练（向量化）",
                "第三步：文档审核（单文件/批量/文本）",
                "知识库查询",
            ],
        }

    def clear_knowledge(self, which: str = "all") -> dict:
        """清空知识库。which: 'regulations' / 'checkpoints' / 'all'"""
        if which in ("regulations", "all"):
            self.kb.clear()
        if which in ("checkpoints", "all"):
            self.checkpoint_kb.clear()
        if which == "all":
            self.checkpoint_feedback_kb.clear()
        return {"status": "success", "message": f"已清空：{which}"}

    def export_report(self, report: dict, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return str(path)
