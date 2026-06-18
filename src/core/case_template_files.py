"""项目案例模板原件：训练时落盘，初稿未上传 Base 时按同名解析为基底文件。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from config import settings


def _safe_collection_segment(collection: str) -> str:
    s = (collection or "").strip() or "default"
    return re.sub(r"[^\w\-.]+", "_", s)[:128]


def case_template_store_dir(collection: str, case_id: int) -> Path:
    return settings.uploads_path / "case_templates" / _safe_collection_segment(collection) / str(int(case_id))


def persist_case_template_file(
    *,
    source_path: str,
    collection: str,
    case_id: int,
    file_name: str,
) -> Optional[str]:
    """训练 project_case 时复制原件，供后续初稿 export_like_base / inplace_patch 使用。"""
    src = Path(source_path)
    if not src.is_file():
        return None
    name = (file_name or src.name).strip() or src.name
    dest_dir = case_template_store_dir(collection, case_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(name).name
    try:
        shutil.copy2(str(src), str(dest))
        return str(dest.resolve())
    except OSError:
        return None


def _basename_matches(a: str, b: str) -> bool:
    return Path(a).name.casefold() == Path(b).name.casefold()


def resolve_case_template_file_path(
    *,
    collection: str,
    case_id: int,
    file_name: str,
) -> Optional[str]:
    """按案例 ID + 模板文件名查找磁盘上的案例原件（docx/xlsx/…）。"""
    fn = (file_name or "").strip()
    if not fn or int(case_id or 0) <= 0:
        return None
    cid = int(case_id)

    store = case_template_store_dir(collection, cid)
    candidates: List[Path] = []
    for cand in (store / Path(fn).name, store / fn):
        if cand.is_file():
            candidates.append(cand)
    if store.is_dir():
        for p in store.iterdir():
            if p.is_file() and _basename_matches(p.name, fn):
                candidates.append(p)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0].resolve())

    td = settings.training_docs_path
    if td.is_dir():
        hits: List[Path] = []
        for p in td.rglob("*"):
            if p.is_file() and _basename_matches(p.name, fn):
                hits.append(p)
        if hits:
            coll_seg = _safe_collection_segment(collection).casefold()
            scoped = [h for h in hits if coll_seg and coll_seg in str(h).casefold()]
            pool = scoped if scoped else hits
            pool.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(pool[0].resolve())
    return None


def enrich_base_paths_from_case_templates(
    *,
    collection: str,
    base_case_id: int,
    template_file_names: Optional[List[str]],
    base_paths_map: Dict[str, str],
    base_name_to_path: Dict[str, str],
    dest_dir: Optional[Path] = None,
) -> None:
    """未上传 Base 时，用案例库模板原件补全 base_paths_map（格式与模板一致）。"""
    cid = int(base_case_id or 0)
    if cid <= 0:
        return
    for tf in template_file_names or []:
        tgs = (tf or "").strip()
        if not tgs or tgs in base_paths_map:
            continue
        src = resolve_case_template_file_path(collection=collection, case_id=cid, file_name=tgs)
        if not src:
            continue
        pth = src
        if dest_dir is not None:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / Path(tgs).name
                if (not dest.is_file()) or dest.stat().st_mtime < Path(src).stat().st_mtime:
                    shutil.copy2(src, dest)
                pth = str(dest.resolve())
            except OSError:
                pth = src
        base_paths_map[tgs] = pth
        base_name_to_path[Path(tgs).name] = pth
