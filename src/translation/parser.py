"""
带结构的文档解析：提取段落、表格单元格、标题等，供分句与回填使用。
支持 .docx（python-docx）、.txt、.xlsx（openpyxl）。
"""
from pathlib import Path
from typing import List

from .models import TextBlock, SUPPORTED_EXTENSIONS


def parse_document(file_path: str) -> List[TextBlock]:
    """
    按后缀解析单个文件，返回文档顺序的文本块列表。
    不支持格式抛出 ValueError。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的格式: {suffix}，首期支持: {list(SUPPORTED_EXTENSIONS)}"
        )
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".txt":
        return _parse_txt(path)
    if suffix == ".xlsx":
        return _parse_xlsx(path)
    raise ValueError(f"未实现解析: {suffix}")


def _parse_docx(path: Path) -> List[TextBlock]:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import _Cell, Table
    from docx.text.paragraph import Paragraph

    doc = Document(path)
    blocks: List[TextBlock] = []
    body = doc.element.body
    table_idx = 0
    for child in body:
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)
            text = (para.text or "").strip()
            blocks.append(TextBlock(
                block_type="paragraph",
                path=(len(blocks),),
                original_text=text,
            ))
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            for row_idx, row in enumerate(table.rows):
                for col_idx, cell in enumerate(row.cells):
                    if isinstance(cell, _Cell):
                        text = (cell.text or "").strip()  # 保留换行，便于翻译后回填
                    else:
                        text = ""
                    blocks.append(TextBlock(
                        block_type="table_cell",
                        path=(table_idx, row_idx, col_idx),
                        original_text=text,
                    ))
            table_idx += 1
    # 页眉、页脚（按 section 顺序）
    for section_idx, section in enumerate(doc.sections):
        try:
            for p_idx, para in enumerate(section.header.paragraphs):
                blocks.append(TextBlock(
                    block_type="header",
                    path=(section_idx, "h", p_idx),
                    original_text=(para.text or "").strip(),
                ))
            for p_idx, para in enumerate(section.footer.paragraphs):
                blocks.append(TextBlock(
                    block_type="footer",
                    path=(section_idx, "f", p_idx),
                    original_text=(para.text or "").strip(),
                ))
        except Exception:
            pass
    return blocks


def _parse_txt(path: Path) -> List[TextBlock]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gbk")
    lines = content.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        blocks.append(TextBlock(
            block_type="line",
            path=(i,),
            original_text=line,
        ))
    return blocks


def _xlsx_header_footer_keys():
    """Excel 页眉页脚键：(属性名, 区域)"""
    return [
        ("oddHeader", "left"), ("oddHeader", "center"), ("oddHeader", "right"),
        ("oddFooter", "left"), ("oddFooter", "center"), ("oddFooter", "right"),
        ("evenHeader", "left"), ("evenHeader", "center"), ("evenHeader", "right"),
        ("evenFooter", "left"), ("evenFooter", "center"), ("evenFooter", "right"),
        ("firstHeader", "left"), ("firstHeader", "center"), ("firstHeader", "right"),
        ("firstFooter", "left"), ("firstFooter", "center"), ("firstFooter", "right"),
    ]


def _xlsx_is_merged_slave(sheet, row_idx: int, col_idx: int) -> bool:
    """若为合并区域内非左上角单元格，返回 True（解析时跳过，避免与主格重复导致写入错位）。"""
    try:
        for rng in sheet.merged_cells.ranges:
            if rng.min_row <= row_idx <= rng.max_row and rng.min_col <= col_idx <= rng.max_col:
                return (row_idx, col_idx) != (rng.min_row, rng.min_col)
    except Exception:
        pass
    return False


def _parse_xlsx(path: Path) -> List[TextBlock]:
    import openpyxl
    # read_only=False 以便读取页眉页脚（openpyxl 在 read_only 下可能不加载）
    wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
    blocks: List[TextBlock] = []
    try:
        for sheet in wb.worksheets:
            blocks.append(TextBlock(
                block_type="sheet_name",
                path=(sheet.title,),
                original_text=sheet.title or "",
            ))
            # 页眉页脚：oddHeader/oddFooter 等下的 left/center/right
            for attr, part in _xlsx_header_footer_keys():
                try:
                    hf = getattr(sheet, attr, None)
                    if hf is None:
                        continue
                    part_obj = getattr(hf, part, None) if part != "centre" else getattr(hf, "center", None)
                    if part_obj is not None:
                        text = (getattr(part_obj, "text", None) or "").strip()
                        blocks.append(TextBlock(
                            block_type="xlsx_header_footer",
                            path=(sheet.title, attr, part),
                            original_text=text,
                        ))
                except Exception:
                    pass
        for sheet in wb.worksheets:
            # 按固定行列遍历，合并区域只保留左上角一格，避免与 iter_rows 重复格导致译文错位
            max_row = sheet.max_row or 0
            max_col = sheet.max_column or 0
            for row_idx in range(1, max_row + 1):
                for col_idx in range(1, max_col + 1):
                    if _xlsx_is_merged_slave(sheet, row_idx, col_idx):
                        continue
                    value = sheet.cell(row=row_idx, column=col_idx).value
                    text = str(value).strip() if value is not None else ""
                    blocks.append(TextBlock(
                        block_type="table_cell",
                        path=(sheet.title, row_idx, col_idx),
                        original_text=text,
                    ))
    finally:
        wb.close()
    return blocks
