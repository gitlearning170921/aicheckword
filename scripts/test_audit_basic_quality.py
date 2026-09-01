"""基础质量扫描自测：序号跳号、重复文案、字段不一致、引用标准年代号。"""
from src.core.audit_basic_quality import scan_basic_quality_issues


def _kinds(findings):
    text = " | ".join(f.description for f in findings)
    return text


def test_number_gap():
    doc = """1. 目的
本文件规定软件生命周期要求。

2. 范围
适用于本公司独立软件。

4. 职责
质量部负责文件控制。
"""
    fs = scan_basic_quality_issues(doc, "demo.docx")
    assert any("不连续" in f.description or "不连贯" in f.description for f in fs), _kinds(fs)


def test_duplicate_para():
    p = "本软件仅用于医疗机构内由经培训的专业人员进行影像浏览与测量，不得用于家庭自行诊断。"
    doc = f"3.1 概述\n\n{p}\n\n4.1 再次描述\n\n{p}\n"
    fs = scan_basic_quality_issues(doc, "demo.docx")
    assert any("相同或几乎相同" in f.description for f in fs), _kinds(fs)


def test_field_mismatch():
    doc = "产品名称：影像工作站A\n\n正文说明产品名称：影像工作站B 用于浏览。\n"
    fs = scan_basic_quality_issues(doc, "demo.docx")
    assert any("产品名称" in f.description and "不一致" in f.description for f in fs), _kinds(fs)


def test_std_year():
    doc = "引用 GB/T 42061-2022 与后文 GB/T 42061-2016 的条款。\n"
    fs = scan_basic_quality_issues(doc, "demo.docx")
    assert any("年代号" in f.description for f in fs), _kinds(fs)


def test_clean_doc_low_noise():
    doc = """1. 目的
规定要求。

2. 范围
适用于独立软件。

3. 职责
质量部负责。

产品名称：影像工作站A

正文再次写产品名称：影像工作站A。
引用 GB/T 42061-2022。
"""
    fs = scan_basic_quality_issues(doc, "demo.docx")
    assert not any("不连续" in f.description for f in fs), _kinds(fs)
    assert not any("产品名称" in f.description and "不一致" in f.description for f in fs), _kinds(fs)


if __name__ == "__main__":
    test_number_gap()
    test_duplicate_para()
    test_field_mismatch()
    test_std_year()
    test_clean_doc_low_noise()
    print("ok")
