"""初稿生成：输入/参考文件向量化 — 按项目检测重名与覆盖/跳过策略。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.core.db import get_existing_project_file_names

ProgressFn = Optional[Callable[[str, float], None]]
LoadingFn = Optional[Callable[[str], None]]
EmbedCb = Optional[Callable[[int, int], None]]


def find_duplicate_input_file_names(project_id: int, display_names: List[str]) -> List[str]:
    """返回在 project_knowledge_docs 中已存在同名 file_name 的展示名列表。"""
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return []
    if pid <= 0:
        return []
    names = [(n or "").strip() for n in (display_names or [])]
    names = [n for n in names if n]
    if not names:
        return []
    existing = set(get_existing_project_file_names(pid))
    out: List[str] = []
    seen: Set[str] = set()
    for dn in names:
        if dn in seen:
            continue
        seen.add(dn)
        if dn in existing:
            out.append(dn)
    return out


def train_input_files_to_project(
    agent: Any,
    project_id: int,
    saved_inputs: List[Tuple[str, str]],
    *,
    on_duplicate: str = "skip",
    progress: ProgressFn = None,
    on_loading: LoadingFn = None,
    embed_callback_factory: Optional[
        Callable[[int, int, str], EmbedCb]
    ] = None,
) -> Dict[str, Any]:
    """
    将输入/参考文件训练到项目向量库。

    on_duplicate:
      - skip: 重名文件不向量化，优先使用库内已有块；
      - overwrite: 先删除库内同名块再重新向量化。
    """
    policy = (on_duplicate or "skip").strip().lower()
    if policy not in ("skip", "overwrite"):
        policy = "skip"

    pid = int(project_id)
    existing = set(get_existing_project_file_names(pid))
    trained: List[str] = []
    skipped: List[str] = []
    overwritten: List[str] = []

    n_in = len(saved_inputs or [])
    for i_in, (fp, fn) in enumerate(saved_inputs or [], start=1):
        dn = (fn or "").strip() or str(fp)
        is_dup = dn in existing

        if is_dup and policy == "skip":
            skipped.append(dn)
            if progress:
                progress(
                    f"参考文件 ({i_in}/{n_in})：{dn} 已在项目向量库中，跳过向量化（使用已有数据）",
                    0.10 + (0.12 * i_in / max(1, n_in)),
                )
            continue

        if is_dup and policy == "overwrite":
            try:
                agent.get_project_kb(pid).delete_documents_by_file_name(dn)
            except Exception:
                pass
            overwritten.append(dn)

        frac_lo, frac_hi = 0.10, 0.22
        base_frac = frac_lo + (frac_hi - frac_lo) * ((i_in - 1) / max(1, n_in))
        span = (frac_hi - frac_lo) / max(1, n_in)

        def _on_loading(msg: str, *, _i=i_in, _fn=dn) -> None:
            if on_loading:
                on_loading(msg)
            elif progress:
                progress(
                    f"训练输入文件 ({_i}/{n_in})：{_fn} · {msg}",
                    base_frac + span * 0.08,
                )

        _embed_cb = None
        if embed_callback_factory:
            _embed_cb = embed_callback_factory(i_in, n_in, dn)

        if progress and not on_loading:
            progress(f"训练输入文件 ({i_in}/{n_in})：{dn}…", base_frac)

        agent.train_project_docs(
            pid,
            fp,
            file_name=dn,
            on_loading=_on_loading,
            callback=_embed_cb,
        )
        trained.append(dn)
        existing.add(dn)

    return {
        "trained": trained,
        "skipped": skipped,
        "overwritten": overwritten,
        "on_duplicate": policy,
    }
