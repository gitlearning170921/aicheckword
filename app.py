"""Streamlit Web UI：注册文档审核工具"""

import json
import time
import tempfile
import shutil
import traceback
from pathlib import Path

import streamlit as st

from config import settings
from core.agent import ReviewAgent
from core.document_loader import (
    load_single_file,
    split_documents,
    LOADER_MAP,
    is_archive,
    extract_archive,
)
from core.db import (
    load_app_settings,
    save_app_settings,
    add_operation_log,
    get_operation_logs,
    get_operation_summary,
    OP_TYPE_TRAIN_BATCH,
    OP_TYPE_REVIEW_BATCH,
)


def _provider_ready() -> bool:
    """当前 provider 是否就绪"""
    if settings.is_ollama:
        return True
    if settings.is_cursor:
        return bool(settings.cursor_api_key and settings.cursor_repository)
    return bool(settings.openai_api_key)


def init_agent():
    """初始化或获取 Agent（不触发 OpenAI 连接）"""
    collection = st.session_state.get("collection_name", "regulations")
    if "agent" not in st.session_state or st.session_state.get("_col") != collection:
        st.session_state.agent = ReviewAgent(collection)
        st.session_state._col = collection
    return st.session_state.agent


def _require_provider() -> bool:
    """检查 AI 服务是否就绪"""
    if _provider_ready():
        return True
    if settings.is_ollama:
        st.warning("⚠️ Ollama 服务未启动，请确保已安装并运行 Ollama（ollama serve）。")
    elif settings.is_cursor:
        st.warning("⚠️ Cursor 模式下请填写 API Key 和 GitHub 仓库地址（Cursor Dashboard → Integrations）。")
    else:
        st.warning("⚠️ 请先在左侧边栏填写 OpenAI API Key。")
    return False


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 设置")

        # 首次从 SQLite 载入配置到 settings（仅一次）
        if not st.session_state.get("db_settings_loaded"):
            db_conf = load_app_settings()
            if db_conf:
                settings.provider = db_conf.get("provider") or settings.provider
                settings.openai_api_key = db_conf.get("openai_api_key") or settings.openai_api_key
                settings.openai_base_url = db_conf.get("openai_base_url") or settings.openai_base_url
                settings.ollama_base_url = db_conf.get("ollama_base_url") or settings.ollama_base_url
                settings.cursor_api_key = db_conf.get("cursor_api_key") or settings.cursor_api_key
                settings.cursor_api_base = db_conf.get("cursor_api_base") or settings.cursor_api_base
                settings.cursor_repository = db_conf.get("cursor_repository") or settings.cursor_repository
                settings.cursor_ref = db_conf.get("cursor_ref") or settings.cursor_ref
                settings.cursor_embedding = db_conf.get("cursor_embedding") or settings.cursor_embedding
                settings.llm_model = db_conf.get("llm_model") or settings.llm_model
                settings.embedding_model = db_conf.get("embedding_model") or settings.embedding_model
            st.session_state["db_settings_loaded"] = True

        # --- AI 服务配置 ---
        st.subheader("AI 服务")

        provider_options = ["Ollama (本地免费)", "OpenAI (需 API Key)", "Cursor Agent (Cloud API)"]
        if settings.is_ollama:
            current_idx = 0
        elif settings.is_cursor:
            current_idx = 2
        else:
            current_idx = 1
        provider_choice = st.selectbox(
            "服务提供方",
            provider_options,
            index=current_idx,
            help="Ollama: 本地免费\nOpenAI: 需 API Key\nCursor: 调用 Cursor Cloud Agents API，需 API Key + GitHub 仓库",
        )
        is_ollama = provider_choice.startswith("Ollama") and "Cursor" not in provider_choice
        is_cursor = provider_choice.startswith("Cursor")

        if is_cursor:
            cursor_api_key = st.text_input(
                "Cursor API Key",
                value=settings.cursor_api_key,
                type="password",
                help="Cursor Dashboard → Integrations 创建",
            )
            cursor_repo = st.text_input(
                "GitHub 仓库地址",
                value=settings.cursor_repository,
                help="必填，如 https://github.com/your-org/your-repo（Agent 会基于该仓库运行）",
            )
            cursor_ref = st.text_input(
                "分支/标签",
                value=settings.cursor_ref,
                help="默认 main",
            )
            cursor_embed = st.selectbox(
                "向量化使用",
                ["ollama", "openai"],
                index=0 if (settings.cursor_embedding or "ollama").lower() == "ollama" else 1,
                help="知识库向量化仍需要 Ollama 或 OpenAI",
            )
            llm_model = settings.llm_model
            embed_model = settings.embedding_model
        elif is_ollama:
            ollama_url = st.text_input(
                "Ollama 地址",
                value=settings.ollama_base_url,
                help="默认 http://localhost:11434，通常不用改",
            )
            llm_model = st.text_input(
                "审核模型",
                value=settings.llm_model,
                help="推荐 qwen2.5（中文好）、llama3.1、mistral 等",
            )
            embed_model = st.text_input(
                "向量化模型",
                value=settings.embedding_model,
                help="推荐 nomic-embed-text、bge-m3 等",
            )
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                value=settings.openai_api_key,
                type="password",
                help="也支持兼容 OpenAI 接口的国内服务",
            )
            base_url = st.text_input(
                "API Base URL",
                value=settings.openai_base_url,
                help="如使用国内代理或兼容服务，修改为对应地址",
            )
            llm_model = st.text_input(
                "审核模型",
                value=settings.llm_model,
                help="如 gpt-4o, gpt-4o-mini, gpt-3.5-turbo 等",
            )
            embed_model = st.text_input(
                "向量化模型",
                value=settings.embedding_model,
                help="如 text-embedding-3-small 等",
            )

        if st.button("💾 保存配置"):
            settings.provider = "cursor" if is_cursor else ("ollama" if is_ollama else "openai")
            settings.llm_model = llm_model
            settings.embedding_model = embed_model
            if is_cursor:
                settings.cursor_api_key = cursor_api_key
                settings.cursor_repository = cursor_repo
                settings.cursor_ref = cursor_ref
                settings.cursor_embedding = cursor_embed
            elif is_ollama:
                settings.ollama_base_url = ollama_url
            else:
                settings.openai_api_key = api_key
                settings.openai_base_url = base_url
            save_app_settings(
                provider=settings.provider,
                openai_api_key=settings.openai_api_key,
                openai_base_url=settings.openai_base_url,
                ollama_base_url=settings.ollama_base_url,
                cursor_api_key=settings.cursor_api_key,
                cursor_api_base=settings.cursor_api_base,
                cursor_repository=settings.cursor_repository,
                cursor_ref=settings.cursor_ref,
                cursor_embedding=settings.cursor_embedding,
                llm_model=settings.llm_model,
                embedding_model=settings.embedding_model,
            )
            if "agent" in st.session_state:
                st.session_state.agent.reset_clients()
            st.success("配置已保存，下次打开自动生效。")

        if _provider_ready():
            if settings.is_cursor:
                label = "Cursor Agent 模式 ✓"
            elif settings.is_ollama:
                label = "Ollama 本地模式 ✓"
            else:
                label = "OpenAI 模式 ✓"
            st.success(label)
        else:
            st.error("AI 服务未就绪")

        st.markdown("---")

        # --- 知识库 ---
        st.subheader("知识库")
        collection = st.text_input(
            "知识库名称",
            value=st.session_state.get("collection_name", "regulations"),
            help="不同项目可使用不同的知识库名称",
        )
        st.session_state.collection_name = collection

        agent = init_agent()
        status = agent.get_status()
        kb_info = status["knowledge_base"]
        st.metric("已入库文档块数", kb_info["document_count"])

        if st.button("🗑️ 清空当前知识库"):
            agent.clear_knowledge()
            st.success("知识库已清空")
            st.experimental_rerun()

        st.markdown("---")
        st.subheader("API 服务")
        st.code(f"http://localhost:{settings.api_port}", language="text")
        st.caption("启动命令：`python -m api.server`")

        st.markdown("---")
        st.caption("注册文档审核工具 v1.0")


def _save_uploaded_file(file) -> str:
    """将上传文件保存到临时目录，返回路径"""
    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        return tmp.name


def _scan_directory_files(dir_path):
    """扫描目录中所有支持格式的文件"""
    path = Path(dir_path)
    files = []
    for ext in LOADER_MAP:
        for fp in path.rglob("*" + ext):
            files.append(fp)
    return files


def _expand_uploads(uploaded_files):
    """
    将上传列表展开：压缩包解压后加入其内文档，普通文件直接加入。
    返回 (items, temp_dirs)。items = [(path, display_name, is_from_archive), ...]
    """
    items = []
    temp_dirs = []

    for file in uploaded_files:
        tmp_path = _save_uploaded_file(file)
        if is_archive(Path(tmp_path)):
            try:
                temp_dir, doc_paths = extract_archive(tmp_path)
                temp_dirs.append(temp_dir)
                Path(tmp_path).unlink(missing_ok=True)
                archive_name = Path(file.name).stem
                for fp in doc_paths:
                    try:
                        rel = fp.relative_to(temp_dir)
                        display = f"{archive_name}/{rel}"
                    except ValueError:
                        display = fp.name
                    items.append((str(fp), display, True))
            except Exception as e:
                items.append((tmp_path, file.name + f" (解压失败: {e})", False))
        else:
            items.append((tmp_path, file.name, False))

    return items, temp_dirs


def _format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"


def _train_single_file(agent, file_path, file_name, log_container, embed_status):
    """训练单个文件，分步展示进度，返回 (成功?, 块数, 耗时)"""
    t0 = time.time()

    log_container.info(f"📂 **[{file_name}]** 正在加载文件...")
    try:
        docs = load_single_file(file_path)
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e)
        log_container.error(f"❌ **[{file_name}]** 加载失败：{err_msg}")
        log_container.caption("详细原因与堆栈已记入「操作记录」，可到该页查看。")
        add_operation_log(
            op_type="train_error",
            collection=agent.collection_name,
            file_name=file_name,
            source=str(file_path),
            extra={"error": err_msg, "traceback": tb, "stage": "load"},
        )
        return False, 0, time.time() - t0
    log_container.info(f"📂 **[{file_name}]** 文件加载完成，共 {len(docs)} 页/段")

    log_container.info(f"✂️ **[{file_name}]** 正在分块...")
    chunks = split_documents(docs)
    log_container.info(f"✂️ **[{file_name}]** 分块完成，共 {len(chunks)} 个文档块")

    if not chunks:
        log_container.warning(f"⚠️ **[{file_name}]** 文件内容为空，跳过")
        return True, 0, time.time() - t0

    log_container.info(f"🔄 **[{file_name}]** 正在向量化并入库（{len(chunks)} 块）...")

    def on_batch_done(done, total):
        pct = done / total
        embed_status.progress(pct)

    embed_status.progress(0.0)
    try:
        count = agent.kb.add_documents_with_progress(chunks, batch_size=20, callback=on_batch_done)
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e)
        log_container.error(f"❌ **[{file_name}]** 入库失败：{err_msg}")
        log_container.caption("详细原因与堆栈已记入「操作记录」，可到该页查看。")
        add_operation_log(
            op_type="train_error",
            collection=agent.collection_name,
            file_name=file_name,
            source=str(file_path),
            extra={"error": err_msg, "traceback": tb, "stage": "embed"},
        )
        return False, 0, time.time() - t0

    elapsed = time.time() - t0
    embed_status.progress(1.0)
    log_container.success(f"✅ **[{file_name}]** 完成！入库 {count} 块，耗时 {_format_time(elapsed)}")
    # 记录成功训练日志
    add_operation_log(
        op_type="train",
        collection=agent.collection_name,
        file_name=file_name,
        source=str(file_path),
        extra={"chunks": count, "duration_sec": elapsed},
    )
    return True, count, elapsed


def render_training_page():
    """训练页面"""
    st.header("📚 知识库训练")
    st.markdown("上传法规、标准、程序文件等，构建审核知识库。支持 **PDF / Word / Excel / TXT / Markdown** 格式。")

    if not _require_provider():
        return

    agent = init_agent()

    tab1, tab2 = st.tabs(["📤 上传文件训练", "📂 从目录训练"])

    with tab1:
        uploaded_files = st.file_uploader(
            "选择训练文件（支持单个文档或 .zip / .tar / .tar.gz 压缩包，压缩包将自动解压后导入）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="train_uploader",
        )

        if uploaded_files and st.button("🚀 开始训练", key="train_btn"):
            items, temp_dirs = _expand_uploads(uploaded_files)

            if not items:
                st.warning("没有可训练的文件：若上传的是压缩包，请确保包内有 PDF/Word/Excel/TXT/Markdown 等格式的文档。")
                return

            total_files = len(items)
            st.markdown(f"**共 {total_files} 个文件待训练**（含压缩包内解压出的文档）")

            overall_bar = st.progress(0.0)
            overall_text = st.empty()
            st.markdown("---")

            embed_status = st.empty()
            log_area = st.container()

            total_chunks = 0
            success_count = 0
            fail_count = 0
            t_start = time.time()

            try:
                for idx, (path, display_name, is_from_archive) in enumerate(items):
                    overall_text.text(
                        f"总进度：{idx}/{total_files} | "
                        f"已成功 {success_count} 个 | 已失败 {fail_count} 个 | "
                        f"已入库 {total_chunks} 块"
                    )
                    overall_bar.progress(idx / total_files)

                    try:
                        ok, chunks, elapsed = _train_single_file(
                            agent, path, display_name, log_area, embed_status
                        )
                        if ok:
                            success_count += 1
                            total_chunks += chunks
                        else:
                            fail_count += 1
                    except Exception as e:
                        log_area.error(f"❌ **[{display_name}]** 异常：{e}")
                        fail_count += 1
                    finally:
                        if not is_from_archive:
                            Path(path).unlink(missing_ok=True)

                overall_bar.progress(1.0)
                total_time = time.time() - t_start
                overall_text.text(
                    f"全部完成！成功 {success_count}/{total_files} 个文件，"
                    f"共入库 {total_chunks} 块，总耗时 {_format_time(total_time)}"
                )
                embed_status.empty()
                if fail_count == 0:
                    st.balloons()
                # 记录本批次训练汇总
                add_operation_log(
                    op_type=OP_TYPE_TRAIN_BATCH,
                    collection=agent.collection_name,
                    file_name="",
                    source="upload",
                    extra={
                        "total_files": total_files,
                        "success_count": success_count,
                        "fail_count": fail_count,
                        "total_chunks": total_chunks,
                        "duration_sec": round(total_time, 2),
                    },
                )
            finally:
                for d in temp_dirs:
                    shutil.rmtree(d, ignore_errors=True)

    with tab2:
        dir_path = st.text_input(
            "输入目录路径",
            value=str(settings.training_path),
            help="服务器上的目录路径，将递归加载所有支持格式的文件",
        )
        if st.button("🚀 从目录训练", key="train_dir_btn"):
            if not Path(dir_path).exists():
                st.error(f"目录不存在：{dir_path}")
                return

            scan_status = st.empty()
            scan_status.info("🔍 正在扫描目录中的文件...")
            files = _scan_directory_files(dir_path)

            if not files:
                scan_status.warning("⚠️ 目录中没有找到支持格式的文件")
                return

            scan_status.success(f"🔍 扫描完成，发现 {len(files)} 个文件")

            overall_bar = st.progress(0.0)
            overall_text = st.empty()
            st.markdown("---")

            embed_status = st.empty()
            log_area = st.container()

            total_chunks = 0
            success_count = 0
            fail_count = 0
            t_start = time.time()

            for idx, fp in enumerate(files):
                overall_text.text(
                    f"总进度：{idx}/{len(files)} | "
                    f"已成功 {success_count} 个 | 已失败 {fail_count} 个 | "
                    f"已入库 {total_chunks} 块"
                )
                overall_bar.progress(idx / len(files))

                try:
                    ok, chunks, elapsed = _train_single_file(
                        agent, str(fp), fp.name, log_area, embed_status
                    )
                    if ok:
                        success_count += 1
                        total_chunks += chunks
                    else:
                        fail_count += 1
                except Exception as e:
                    log_area.error(f"❌ **[{fp.name}]** 异常：{e}")
                    fail_count += 1

            overall_bar.progress(1.0)
            total_time = time.time() - t_start
            overall_text.text(
                f"全部完成！成功 {success_count}/{len(files)} 个文件，"
                f"共入库 {total_chunks} 块，总耗时 {_format_time(total_time)}"
            )
            embed_status.empty()
            if fail_count == 0:
                st.balloons()
            # 记录本批次训练汇总
            add_operation_log(
                op_type=OP_TYPE_TRAIN_BATCH,
                collection=agent.collection_name,
                file_name="",
                source="directory",
                extra={
                    "total_files": len(files),
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "total_chunks": total_chunks,
                    "duration_sec": round(total_time, 2),
                    "dir_path": dir_path,
                },
            )


def render_review_page():
    """审核页面"""
    st.header("🔍 文档审核")
    st.markdown("上传待审核的注册文档，AI 将根据知识库中的法规和标准自动识别审核点。")

    if not _require_provider():
        return

    agent = init_agent()

    status = agent.get_status()
    if status["knowledge_base"]["document_count"] == 0:
        st.warning("⚠️ 知识库为空，请先到「知识库训练」页面上传法规文件进行训练。")

    tab1, tab2 = st.tabs(["📤 上传文件审核", "📝 文本审核"])

    with tab1:
        review_files = st.file_uploader(
            "选择待审核文件（支持单个文档或 .zip / .tar / .tar.gz 压缩包，压缩包将自动解压后逐个审核）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="review_uploader",
        )

        if review_files and st.button("🔍 开始审核", key="review_btn"):
            items, temp_dirs = _expand_uploads(review_files)

            if not items:
                st.warning("没有可审核的文件：若上传的是压缩包，请确保包内有 PDF/Word/Excel/TXT/Markdown 等格式的文档。")
                return

            total_files = len(items)
            st.markdown(f"**共 {total_files} 个文件待审核**（含压缩包内解压出的文档）")

            review_bar = st.progress(0.0)
            review_status = st.empty()
            review_log = st.container()

            all_reports = []
            t_start = time.time()

            try:
                for idx, (path, display_name, is_from_archive) in enumerate(items):
                    review_bar.progress(idx / total_files)
                    review_status.text(
                        f"审核进度：{idx}/{total_files} | "
                        f"已完成 {len(all_reports)} 个 | "
                        f"耗时 {_format_time(time.time() - t_start)}"
                    )
                    review_log.info(f"🔍 **[{display_name}]** 正在审核...")

                    try:
                        t0 = time.time()
                        report = agent.review(path)
                        elapsed = time.time() - t0
                        report["original_filename"] = display_name
                        all_reports.append(report)
                        n_points = report.get("total_points", 0)
                        review_log.success(
                            f"✅ **[{display_name}]** 审核完成，"
                            f"发现 {n_points} 个审核点，耗时 {_format_time(elapsed)}"
                        )
                    except Exception as e:
                        review_log.error(f"❌ **[{display_name}]** 审核失败：{e}")
                        add_operation_log(
                            op_type="review_error",
                            collection=agent.collection_name,
                            file_name=display_name,
                            source=str(path),
                            extra={"error": str(e)},
                        )
                    finally:
                        if not is_from_archive:
                            Path(path).unlink(missing_ok=True)

                review_bar.progress(1.0)
                total_time = time.time() - t_start
                review_status.text(
                    f"全部完成！共审核 {len(all_reports)}/{total_files} 个文件，"
                    f"总耗时 {_format_time(total_time)}"
                )

                if all_reports:
                    st.session_state.review_reports = all_reports
                # 记录本批次审核汇总
                total_points = sum(r.get("total_points", 0) for r in all_reports)
                add_operation_log(
                    op_type=OP_TYPE_REVIEW_BATCH,
                    collection=agent.collection_name,
                    file_name="",
                    source="upload",
                    extra={
                        "total_files": total_files,
                        "success_count": len(all_reports),
                        "fail_count": total_files - len(all_reports),
                        "total_audit_points": total_points,
                        "duration_sec": round(total_time, 2),
                    },
                )
            finally:
                for d in temp_dirs:
                    shutil.rmtree(d, ignore_errors=True)

    with tab2:
        review_text = st.text_area(
            "输入待审核文本",
            height=300,
            placeholder="粘贴文档内容到这里...",
        )
        text_file_name = st.text_input("文件名（可选）", value="直接输入")

        if review_text and st.button("🔍 审核文本", key="review_text_btn"):
            text_status = st.empty()
            text_status.info(f"🔍 正在审核文本（{len(review_text)} 字）...")
            t0 = time.time()
            try:
                report = agent.review_text(review_text, text_file_name)
                elapsed = time.time() - t0
                st.session_state.review_reports = [report]
                n_points = report.get("total_points", 0)
                text_status.success(
                    f"✅ 审核完成！发现 {n_points} 个审核点，耗时 {_format_time(elapsed)}"
                )
            except Exception as e:
                text_status.error(f"❌ 审核失败：{e}")
                add_operation_log(
                    op_type="review_text_error",
                    collection=agent.collection_name,
                    file_name=text_file_name,
                    source="text_input",
                    extra={"error": str(e)},
                )

    if "review_reports" in st.session_state and st.session_state.review_reports:
        st.markdown("---")
        render_reports(st.session_state.review_reports)


def render_reports(reports: list):
    """渲染审核报告"""
    st.subheader("📋 审核报告")

    for report in reports:
        file_name = report.get("original_filename", report.get("file_name", "未知"))

        with st.expander(f"📄 {file_name}", expanded=True):
            cols = st.columns(4)
            cols[0].metric("🔴 高风险", report.get("high_count", 0))
            cols[1].metric("🟡 中风险", report.get("medium_count", 0))
            cols[2].metric("🔵 低风险", report.get("low_count", 0))
            cols[3].metric("ℹ️ 提示", report.get("info_count", 0))

            if report.get("summary"):
                st.markdown(f"**总结：** {report['summary']}")

            st.markdown("---")

            severity_colors = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🔵",
                "info": "ℹ️",
            }

            for i, point in enumerate(report.get("audit_points", []), 1):
                severity = point.get("severity", "info")
                icon = severity_colors.get(severity, "ℹ️")

                st.markdown(f"### {icon} 审核点 {i}：{point.get('category', '未分类')}")

                col1, col2 = st.columns(2)
                col1.markdown(f"**严重程度：** `{severity}`")
                col2.markdown(f"**位置：** {point.get('location', '未知')}")

                st.markdown(f"**问题描述：** {point.get('description', '')}")
                st.markdown(f"**法规依据：** {point.get('regulation_ref', '')}")
                st.markdown(f"**修改建议：** {point.get('suggestion', '')}")
                st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        json_data = json.dumps(reports, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 下载审核报告 (JSON)",
            data=json_data,
            file_name="audit_report.json",
            mime="application/json",
        )

    with col2:
        md_lines = []
        for report in reports:
            file_name = report.get("original_filename", report.get("file_name", ""))
            md_lines.append(f"# 审核报告：{file_name}\n")
            md_lines.append(f"**总结：** {report.get('summary', '')}\n")
            md_lines.append(f"| 高风险 | 中风险 | 低风险 | 提示 |")
            md_lines.append(f"|--------|--------|--------|------|")
            md_lines.append(
                f"| {report.get('high_count', 0)} | {report.get('medium_count', 0)} "
                f"| {report.get('low_count', 0)} | {report.get('info_count', 0)} |\n"
            )
            for i, point in enumerate(report.get("audit_points", []), 1):
                md_lines.append(f"## 审核点 {i}：{point.get('category', '')}")
                md_lines.append(f"- **严重程度：** {point.get('severity', '')}")
                md_lines.append(f"- **位置：** {point.get('location', '')}")
                md_lines.append(f"- **描述：** {point.get('description', '')}")
                md_lines.append(f"- **法规依据：** {point.get('regulation_ref', '')}")
                md_lines.append(f"- **建议：** {point.get('suggestion', '')}\n")

        st.download_button(
            "📥 下载审核报告 (Markdown)",
            data="\n".join(md_lines),
            file_name="audit_report.md",
            mime="text/markdown",
        )


def render_knowledge_page():
    """知识库查询页面"""
    st.header("🔎 知识库查询")
    st.markdown("查询已训练的知识库内容，验证法规和标准是否已正确入库。")

    if not _require_provider():
        return

    agent = init_agent()

    query = st.text_input("输入查询内容", placeholder="例如：产品注册需要哪些资料？")
    top_k = st.slider("返回结果数", 1, 20, 5)

    if query and st.button("🔍 查询", key="search_btn"):
        with st.spinner("正在检索..."):
            results = agent.search_knowledge(query, top_k=top_k)

        if not results:
            st.warning("未找到相关内容，请确认知识库已训练。")
        else:
            for i, result in enumerate(results, 1):
                with st.expander(f"📄 结果 {i} — 来源：{result['source']}", expanded=(i <= 3)):
                    st.markdown(result["content"])


def _op_type_label(op_type):
    """操作类型中文标签"""
    labels = {
        OP_TYPE_TRAIN_BATCH: "📚 训练批次",
        "train": "📄 单文件训练",
        "train_error": "❌ 训练失败",
        OP_TYPE_REVIEW_BATCH: "🔍 审核批次",
        "review_error": "❌ 审核失败",
        "review_text_error": "❌ 文本审核失败",
    }
    return labels.get(op_type, op_type)


def render_operations_page():
    """操作记录页面：查看导入/训练/审核等日志"""
    st.header("📋 操作记录")
    st.markdown("查看每次导入、训练、审核的批次汇总与明细，支持按类型和知识库筛选。")

    summary = get_operation_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("训练批次数", summary["total_train_batches"])
    c2.metric("审核批次数", summary["total_review_batches"])
    c3.metric("今日操作数", summary["today_operations"])

    st.markdown("---")

    col_filter, col_limit = st.columns(2)
    with col_filter:
        op_type_filter = st.selectbox(
            "操作类型",
            ["全部", "训练批次", "审核批次", "单文件训练", "训练失败", "审核失败"],
            key="op_type_filter",
        )
    with col_limit:
        limit = st.selectbox("显示条数", [50, 100, 200, 500], index=1, key="op_limit")

    type_map = {
        "全部": None,
        "训练批次": OP_TYPE_TRAIN_BATCH,
        "审核批次": OP_TYPE_REVIEW_BATCH,
        "单文件训练": "train",
        "训练失败": "train_error",
        "审核失败": "review_error",
    }
    op_type = type_map.get(op_type_filter, None)

    only_current = st.checkbox("仅当前知识库", value=False, key="op_only_collection")
    collection_filter = st.session_state.get("collection_name", "regulations") if only_current else None

    logs = get_operation_logs(op_type=op_type, collection=collection_filter, limit=limit)

    if not logs:
        st.info("暂无操作记录，完成一次训练或审核后会自动记录。")
        return

    for rec in logs:
        extra = rec.get("extra") or {}
        op_label = _op_type_label(rec["op_type"])

        if rec["op_type"] == OP_TYPE_TRAIN_BATCH:
            title = (
                f"{op_label} | 导入 {extra.get('total_files', 0)} 个文件，"
                f"成功 {extra.get('success_count', 0)}，失败 {extra.get('fail_count', 0)}，"
                f"入库 {extra.get('total_chunks', 0)} 块，耗时 {extra.get('duration_sec', 0):.1f}s"
            )
            detail = f"来源：{rec.get('source', '')} | 知识库：{rec.get('collection', '')}"
        elif rec["op_type"] == OP_TYPE_REVIEW_BATCH:
            title = (
                f"{op_label} | 审核 {extra.get('total_files', 0)} 个文件，"
                f"完成 {extra.get('success_count', 0)} 个，共 {extra.get('total_audit_points', 0)} 个审核点，"
                f"耗时 {extra.get('duration_sec', 0):.1f}s"
            )
            detail = f"来源：{rec.get('source', '')} | 知识库：{rec.get('collection', '')}"
        elif rec["op_type"] == "train":
            title = f"{op_label} | {rec.get('file_name', '')} | 入库 {extra.get('chunks', 0)} 块"
            detail = f"知识库：{rec.get('collection', '')} | 耗时 {extra.get('duration_sec', 0):.1f}s"
        elif rec["op_type"] in ("train_error", "review_error", "review_text_error"):
            title = f"{op_label} | {rec.get('file_name', '')}"
            detail = f"错误：{extra.get('error', '')} | 知识库：{rec.get('collection', '')}"
        else:
            title = f"{op_label} | {rec.get('file_name', '')}"
            detail = rec.get("source", "")

        with st.expander(f"**{rec.get('created_at', '')}** — {title}", expanded=False):
            st.caption(detail)
            if rec["op_type"] in ("train_error", "review_error", "review_text_error") and extra.get("traceback"):
                st.markdown("**堆栈日志：**")
                st.code(extra.get("traceback", ""), language="text")
            if extra:
                st.json(extra)

    st.markdown("---")
    st.caption("仅展示批次汇总与单条明细，按时间倒序。")


def main():
    st.set_page_config(
        page_title="注册文档审核工具",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.title("📋 注册文档审核工具")

    page = st.radio(
        "选择功能",
        ["📚 知识库训练", "🔍 文档审核", "🔎 知识库查询", "📋 操作记录"],
        horizontal=True,
    )

    if page == "📚 知识库训练":
        render_training_page()
    elif page == "🔍 文档审核":
        render_review_page()
    elif page == "🔎 知识库查询":
        render_knowledge_page()
    elif page == "📋 操作记录":
        render_operations_page()


if __name__ == "__main__":
    main()
