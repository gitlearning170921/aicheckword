"""文档加载器：支持 PDF、Word、Excel、TXT、Markdown 等格式，以及压缩包自动解压"""

import os
import shutil
import subprocess
import zipfile
import tempfile
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from config import settings

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": UnstructuredWordDocumentLoader,  # 旧版 .doc 用 Unstructured（需系统安装 LibreOffice 时效果最佳）
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}

# 支持的压缩格式（.rar 需安装 rarfile 及系统 UnRAR 可执行文件）
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tgz", ".gz", ".rar")
SUPPORTED_DOC_EXTENSIONS = tuple(LOADER_MAP.keys())

# 标了「废弃」的文件或文件夹在训练、审核时跳过（路径或文件名中含此标记即视为废弃）
DEPRECATED_MARKER = "废弃"


def is_deprecated_path(path_or_name) -> bool:
    """路径或文件名中若包含「废弃」则视为废弃，跳过训练/审核。"""
    s = str(path_or_name) if path_or_name is not None else ""
    return DEPRECATED_MARKER in s


def extract_section_outline_from_texts(texts: List[str], max_sections: int = 80) -> str:
    """从多段文档内容中提取章节标题，形成「应有章节」参考列表，用于文档内容完整性审核。
    识别常见格式：第X章、1. 1.1、一、二、# 标题 等。去重并按出现顺序保留。"""
    import re
    seen = set()
    sections = []
    # 常见章节模式（一行视为一个候选标题）
    patterns = [
        re.compile(r"^第[一二三四五六七八九十百零\d]+章\s*.+$", re.MULTILINE),
        re.compile(r"^\d+(\.\d+)*[\.\s、]\s*.+$", re.MULTILINE),  # 1. 1.1 1.1.1
        re.compile(r"^[一二三四五六七八九十]+[、．.]\s*.+$", re.MULTILINE),
        re.compile(r"^#+\s*.+$", re.MULTILINE),  # Markdown
        re.compile(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[)）]\s*.+$", re.MULTILINE),
    ]
    for text in (texts or []):
        if not (text and isinstance(text, str)):
            continue
        for line in text.splitlines():
            line = line.strip()
            if len(line) < 2 or len(line) > 200:
                continue
            for pat in patterns:
                if pat.match(line):
                    key = re.sub(r"\s+", " ", line).strip()
                    if key and key not in seen:
                        seen.add(key)
                        sections.append(line if len(line) <= 100 else line[:97] + "...")
                    break
            if len(sections) >= max_sections:
                break
        if len(sections) >= max_sections:
            break
    if not sections:
        return ""
    return "\n".join(sections)


# PDF：仅支持文本型；含图片/扫描版无法提取文字。限制最大页数避免大文件卡死
MAX_PDF_PAGES = 500
MIN_PDF_TEXT_LEN = 20  # 提取文字少于此长度视为“图片版/扫描版”


def _load_xlsx_with_openpyxl(path: Path) -> List[Document]:
    """使用 openpyxl 读取 xlsx（当 UnstructuredExcelLoader 失败时备用），避免路径/格式兼容问题。"""
    import openpyxl
    path = Path(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    docs = []
    try:
        for sheet in wb.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append("\t".join(str(c) if c is not None else "" for c in row))
            text = "\n".join(rows).strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source_file": path.name, "file_type": ".xlsx", "sheet": sheet.title},
                ))
    finally:
        wb.close()
    return docs if docs else [Document(page_content="(空表)", metadata={"source_file": path.name, "file_type": ".xlsx"})]


def _find_soffice() -> Optional[str]:
    """查找系统已安装的 LibreOffice soffice 可执行路径（Windows 常见路径或 PATH）。"""
    # 环境变量优先，便于用户自定义安装路径
    env_path = os.environ.get("LIBREOFFICE_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_file() and p.exists():
            return str(p.resolve())
        if p.is_dir():
            for exe in ("soffice.exe", "soffice"):
                candidate = p / exe
                if candidate.exists():
                    return str(candidate.resolve())
    if os.name == "nt":
        for base in (
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ):
            if Path(base).exists():
                return base
        for name in ("soffice.exe", "soffice", "libreoffice.exe", "libreoffice"):
            found = shutil.which(name)
            if found:
                return found
    else:
        for name in ("soffice", "libreoffice"):
            found = shutil.which(name)
            if found:
                return found
    return None


def _find_wps() -> Optional[str]:
    """查找 Windows 下已安装的 WPS 文字（wps.exe）。可通过环境变量 WPS_PATH 指定。"""
    if os.name != "nt":
        return None
    env_path = os.environ.get("WPS_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_file() and p.suffix.lower() in (".exe", ""):
            return str(p.resolve())
        if p.is_dir():
            for name in ("wps.exe", "et.exe"):
                candidate = p / name
                if candidate.exists():
                    return str(candidate.resolve())
    # 常见安装路径：Kingsoft WPS Office
    for base in (
        Path(os.environ.get("ProgramFiles(X86)", r"C:\Program Files (x86)")) / "Kingsoft" / "WPS Office",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Kingsoft" / "WPS Office",
    ):
        if not base.exists():
            continue
        for sub in base.iterdir():
            if sub.is_dir():
                wps_exe = sub / "wps.exe"
                if wps_exe.exists():
                    return str(wps_exe.resolve())
        return None


def _convert_doc_to_docx_with_pandoc(doc_path: Path) -> Optional[Path]:
    """使用 pandoc 将 .doc 转为 .docx（需系统已安装 pandoc 且支持 doc 格式）。"""
    doc_path = doc_path.resolve()
    if not doc_path.exists():
        return None
    pandoc_exe = shutil.which("pandoc")
    if not pandoc_exe:
        return None
    fd, out_file = tempfile.mkstemp(suffix=".docx", prefix="aicheckword_pandoc_")
    os.close(fd)
    try:
        result = subprocess.run(
            [pandoc_exe, "-f", "doc", "-t", "docx", "-o", out_file, str(doc_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and Path(out_file).exists() and Path(out_file).stat().st_size > 0:
            return Path(out_file)
    except Exception:
        pass
    try:
        Path(out_file).unlink(missing_ok=True)
    except Exception:
        pass
    return None


def _convert_doc_to_docx_with_wps(doc_path: Path) -> Optional[Path]:
    """使用 WPS Office COM 接口将 .doc 转为 .docx（仅 Windows，需已安装 WPS 与 pywin32）。
    只尝试 Kwps.Application；用完显式释放 COM。"""
    if os.name != "nt":
        return None
    doc_path = doc_path.resolve()
    if not doc_path.exists():
        return None
    try:
        import win32com.client
    except ImportError:
        return None
    prog_id = "Kwps.Application"
    wps = None
    doc = None
    out_file = None
    try:
        wps = win32com.client.Dispatch(prog_id)
        if not wps:
            return None
        # WPS 为兼容性会将 Name 报告为 "Microsoft Word"，不能以此判断是否真正的 MS Word。
        # 改为检查可执行路径：真正的 WPS 进程路径中包含 "Kingsoft" 或 "WPS Office"。
        try:
            exe_path = (getattr(wps, "Path", None) or "").strip()
            is_genuine_msword = exe_path and "Microsoft" in exe_path and "Kingsoft" not in exe_path and "WPS" not in exe_path
            if is_genuine_msword:
                wps.Quit()
                return None
        except Exception:
            pass
        wps.Visible = False
        doc_full = str(doc_path.resolve())
        doc = wps.Documents.Open(doc_full)
        fd, out_file = tempfile.mkstemp(suffix=".docx", prefix="aicheckword_wps_")
        os.close(fd)
        doc.SaveAs2(out_file, FileFormat=12)
        doc.Close(False)
        doc = None
        wps.Quit()
        wps = None
        return Path(out_file)
    except Exception:
        if doc is not None:
            try:
                doc.Close(False)
            except Exception:
                pass
        if wps is not None:
            try:
                wps.Quit()
            except Exception:
                pass
        if out_file and Path(out_file).exists():
            try:
                Path(out_file).unlink(missing_ok=True)
            except Exception:
                pass
        return None


def _convert_doc_to_docx_with_libreoffice(doc_path: Path) -> Optional[Path]:
    """使用 LibreOffice 将 .doc 转为 .docx，返回生成的 .docx 路径；失败返回 None。"""
    soffice = _find_soffice()
    if not soffice:
        return None
    doc_path = doc_path.resolve()
    if not doc_path.exists():
        return None
    out_dir = tempfile.mkdtemp(prefix="aicheckword_doc_")
    result = None
    try:
        env = os.environ.copy()
        env["SAL_USE_VCLPLUGIN"] = "headless"
        # 独立 UserInstallation 避免与已打开的 LibreOffice 冲突
        user_inst = tempfile.mkdtemp(prefix="LibreOffice_Conversion_")
        try:
            user_inst_uri = Path(user_inst).as_uri()
            cmd = [
                soffice,
                "--headless",
                f"-env:UserInstallation={user_inst_uri}",
                "--convert-to",
                "docx",
                "--outdir",
                out_dir,
                str(doc_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
                cwd=out_dir,
            )
        finally:
            shutil.rmtree(user_inst, ignore_errors=True)
        if result is None or result.returncode != 0:
            return None
        # 输出文件名与输入相同，扩展名改为 .docx
        docx_name = doc_path.stem + ".docx"
        docx_path = Path(out_dir) / docx_name
        if docx_path.exists():
            # 移到临时文件供调用方使用，避免 out_dir 被删后无法读
            fd, tmp = tempfile.mkstemp(suffix=".docx", prefix="aicheckword_")
            os.close(fd)
            shutil.copy2(docx_path, tmp)
            shutil.rmtree(out_dir, ignore_errors=True)
            return Path(tmp)
        return None
    except Exception:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        return None


def _load_word_with_unstructured(path: Path, suffix: str) -> List[Document]:
    """使用 UnstructuredWordDocumentLoader 加载 .doc 或 .docx（兼容旧版 .doc，需 unstructured 且推荐安装 LibreOffice）。"""
    loader = UnstructuredWordDocumentLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source_file", path.name)
        d.metadata.setdefault("file_type", suffix)
    return docs if docs else [Document(page_content="(空)", metadata={"source_file": path.name, "file_type": suffix})]


def load_single_file(file_path) -> List[Document]:
    """加载单个文件，返回 Document 列表。失败时抛出带完整原因的异常。"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in LOADER_MAP:
        raise ValueError(f"不支持的文件格式: {suffix}，支持的格式: {list(LOADER_MAP.keys())}")

    loader_cls = LOADER_MAP[suffix]
    last_error = None

    # Excel：先尝试 UnstructuredExcelLoader，失败则用 openpyxl 备用（兼容中文路径、复杂表等）
    if suffix in (".xlsx", ".xls"):
        if suffix == ".xlsx":
            try:
                loader = loader_cls(str(path))
                docs = loader.load()
            except Exception as e:
                last_error = e
                try:
                    docs = _load_xlsx_with_openpyxl(path)
                except Exception as e2:
                    err_msg = f"Excel 加载失败: {last_error!s}\n备用解析(openpyxl)也失败: {e2!s}"
                    raise RuntimeError(err_msg) from last_error
        else:
            # .xls 仅用原 loader，无 openpyxl 备用
            try:
                loader = loader_cls(str(path))
                docs = loader.load()
            except Exception as e:
                raise RuntimeError(f"xls 文件加载失败: {e!s}") from e
    elif suffix == ".pdf":
        # PDF：用 lazy_load + 页数上限，避免大文件或复杂版式卡死；检测图片版/扫描版
        try:
            loader = loader_cls(str(path))
            docs = []
            for i, doc in enumerate(loader.lazy_load()):
                if i >= MAX_PDF_PAGES:
                    break
                doc.metadata.setdefault("source_file", path.name)
                doc.metadata.setdefault("file_type", ".pdf")
                doc.metadata.setdefault("page", i + 1)
                docs.append(doc)
            if not docs:
                raise RuntimeError(
                    "该 PDF 无法解析出任何页面。若为扫描件或纯图片 PDF，当前不支持（需先做 OCR 或使用文本型 PDF）。"
                )
            total_text = "".join(d.page_content for d in docs)
            if len(total_text.strip()) < MIN_PDF_TEXT_LEN:
                raise RuntimeError(
                    f"该 PDF 几乎未提取到文字（仅 {len(total_text.strip())} 字），可能为扫描件/图片版。"
                    "当前仅支持文本型 PDF，建议先 OCR 转成文本或使用可复制文字的 PDF。"
                )
        except Exception as e:
            if "无法解析" in str(e) or "几乎未提取" in str(e):
                raise
            raise RuntimeError(f"PDF 加载失败: {e!s}") from e
    elif suffix in (".docx", ".doc"):
        # Word：.docx 优先用 Docx2txtLoader；遇 BadZipFile（实为 .doc 或损坏）或 .doc 则用 Unstructured 兼容旧版 .doc
        if suffix == ".docx":
            try:
                loader = Docx2txtLoader(str(path))
                docs = loader.load()
            except zipfile.BadZipFile:
                try:
                    docs = _load_word_with_unstructured(path, suffix)
                except Exception as e2:
                    raise RuntimeError(
                        "该文件不是有效的 .docx 格式（可能为旧版 .doc 或损坏）。"
                        "已尝试用 Unstructured 解析仍失败，请确认文件可被 Word 打开，或安装 LibreOffice 后重试。"
                        f" 详情: {e2!s}"
                    ) from e2
            except Exception as e:
                if "zip" in str(e).lower() or "not a zip" in str(e).lower():
                    try:
                        docs = _load_word_with_unstructured(path, suffix)
                    except Exception as e2:
                        raise RuntimeError(
                            "该 Word 文件无法按 .docx 解析，已尝试 Unstructured 仍失败。"
                            "若为旧版 .doc 或损坏，请安装 LibreOffice 或改用 .docx。"
                            f" 详情: {e2!s}"
                        ) from e2
                else:
                    raise RuntimeError(f"Word 加载失败: {e!s}") from e
        else:
            # .doc 旧版格式：先统一尝试转为 .docx 再训练（WPS 仅 Windows；LibreOffice / Pandoc 全平台），避免 Unstructured 不稳定
            docx_path = None
            if os.name == "nt":
                docx_path = _convert_doc_to_docx_with_wps(path)
            if not (docx_path and docx_path.exists()):
                docx_path = _convert_doc_to_docx_with_libreoffice(path)
            if not (docx_path and docx_path.exists()):
                docx_path = _convert_doc_to_docx_with_pandoc(path)
            if docx_path and docx_path.exists():
                try:
                    loader = Docx2txtLoader(str(docx_path))
                    docs = loader.load()
                    for d in docs:
                        d.metadata.setdefault("source_file", path.name)
                        d.metadata.setdefault("file_type", ".doc")
                    if not docs:
                        docs = [Document(page_content="(空)", metadata={"source_file": path.name, "file_type": ".doc"})]
                finally:
                    try:
                        Path(docx_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                # 转换均失败时再试 Unstructured；若仍失败则报错并提示安装 LibreOffice
                err_unstruct = None
                try:
                    docs = _load_word_with_unstructured(path, suffix)
                except Exception as e:
                    err_unstruct = e
                    docs = None
                if docs is None or (docs and all((d.page_content or "").strip() in ("", "(空)") for d in docs)):
                    hint = (
                        "请安装 LibreOffice（推荐）或 WPS Office（仅 Windows，需 pywin32），或安装 pandoc；"
                        "或将 .doc 用 Word/WPS 另存为 .docx 后上传。"
                    )
                    msg = str(err_unstruct) if err_unstruct else "未解析出内容"
                    raise RuntimeError("旧版 .doc 无法转为 .docx 且解析失败: %s。%s" % (msg, hint)) from err_unstruct
        for doc in docs:
            doc.metadata.setdefault("source_file", path.name)
            doc.metadata.setdefault("file_type", suffix)
    else:
        try:
            loader = loader_cls(str(path))
            docs = loader.load()
        except Exception as e:
            # 仅对纯文本类用 TextLoader 兜底
            if suffix == ".txt":
                try:
                    loader = TextLoader(str(path), encoding="utf-8")
                    docs = loader.load()
                except Exception as e2:
                    raise RuntimeError(f"文本加载失败: {e!s}; 备用编码失败: {e2!s}") from e
            else:
                raise RuntimeError(f"加载失败: {e!s}") from e

    for doc in docs:
        doc.metadata.setdefault("source_file", path.name)
        doc.metadata.setdefault("file_type", suffix)

    return docs


def is_archive(path) -> bool:
    """判断是否为支持的压缩文件"""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return True
    return path.suffix.lower() in (".zip", ".tar", ".gz", ".rar")


def _open_zip_with_encoding(archive_path) -> zipfile.ZipFile:
    """用正确编码打开 ZIP，避免中文等文件名乱码。先尝试 gbk（常见于含文件+文件夹的 Windows 压缩包），再 utf-8。"""
    path = Path(archive_path)
    try:
        _ = zipfile.ZipFile(path, "r", metadata_encoding="utf-8")
        _.close()
    except TypeError:
        return zipfile.ZipFile(path, "r")
    for enc in ("gbk", "utf-8", "cp437"):
        try:
            zf = zipfile.ZipFile(path, "r", metadata_encoding=enc)
            names = zf.namelist()
            if names and any("\ufffd" in n for n in names):
                zf.close()
                continue
            return zf
        except (ValueError, UnicodeDecodeError):
            continue
    return zipfile.ZipFile(path, "r", metadata_encoding="utf-8")


def extract_archive(archive_path) -> Tuple[str, List[Path]]:
    """解压到临时目录，返回 (临时目录路径, 文档路径列表)。调用方需在完成后删除临时目录。文件名编码兼容 utf-8/gbk，避免乱码。"""
    archive_path = Path(archive_path)
    temp_dir = tempfile.mkdtemp(prefix="aicheckword_")
    temp_path = Path(temp_dir)
    doc_files = []

    try:
        if archive_path.suffix.lower() == ".zip":
            zf = _open_zip_with_encoding(archive_path)
            try:
                for name in zf.namelist():
                    if name.endswith("/") or "/__MACOSX" in name:
                        continue
                    base = Path(name).name
                    if not base or base.startswith("."):
                        continue
                    zf.extract(name, temp_dir)
                    extracted = temp_path / name
                    if extracted.is_file():
                        suf = extracted.suffix.lower()
                        if suf in SUPPORTED_DOC_EXTENSIONS and not is_deprecated_path(name):
                            doc_files.append(extracted)
            finally:
                zf.close()

        elif archive_path.name.lower().endswith((".tar.gz", ".tgz")) or archive_path.suffix.lower() in (".tar", ".gz"):
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if "/__MACOSX" in name or name.startswith("."):
                        continue
                    tf.extract(member, temp_dir)
                    extracted = temp_path / member.name
                    if extracted.is_file():
                        suf = extracted.suffix.lower()
                        if suf in SUPPORTED_DOC_EXTENSIONS and not is_deprecated_path(member.name):
                            doc_files.append(extracted)
        elif archive_path.suffix.lower() == ".rar":
            try:
                import rarfile
            except ImportError:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(
                    "解压 .rar 需要安装 rarfile：pip install rarfile；且系统需安装 UnRAR（如 Windows 下将 UnRAR 加入 PATH，或从 https://www.rarlab.com/rar_add.htm 下载）。也可先将文件改为 .zip 后上传。"
                ) from None
            try:
                with rarfile.RarFile(archive_path) as rf:
                    for name in rf.namelist():
                        if name.endswith("/") or "/__MACOSX" in name or not name.strip():
                            continue
                        base = Path(name).name
                        if not base or base.startswith("."):
                            continue
                        rf.extract(name, temp_dir)
                        extracted = temp_path / name
                        if extracted.is_file():
                            suf = extracted.suffix.lower()
                            if suf in SUPPORTED_DOC_EXTENSIONS and not is_deprecated_path(name):
                                doc_files.append(extracted)
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(
                    f"解压 .rar 失败: {e}。请确认已安装 rarfile（pip install rarfile）且系统已安装 UnRAR，或将文件转为 .zip 后上传。"
                ) from e
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError("不支持的压缩格式: " + str(archive_path.suffix))

        return temp_dir, doc_files

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_directory(dir_path) -> List[Document]:
    """加载目录下所有支持格式的文件；路径中含「废弃」的文件或目录跳过。"""
    path = Path(dir_path)
    all_docs = []

    for ext in LOADER_MAP:
        for file_path in path.rglob(f"*{ext}"):
            if is_deprecated_path(file_path):
                continue
            try:
                docs = load_single_file(file_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")

    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """将文档分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(file_path) -> List[Document]:
    """加载并分块单个文件"""
    docs = load_single_file(file_path)
    return split_documents(docs)


def load_and_split_directory(dir_path) -> List[Document]:
    """加载并分块目录下所有文件"""
    docs = load_directory(dir_path)
    return split_documents(docs)
