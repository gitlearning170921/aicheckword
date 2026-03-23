"""
文档翻译 CLI：支持单文件、文件夹、压缩包。
用法: python -m src.translation --input <路径> [--output <目录>] [--no-kb] [--collection <名称>]
"""
import argparse
import sys
from pathlib import Path

# 保证项目根在 path 中
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.translation.pipeline import translate_path
from src.translation.models import SUPPORTED_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(
        description="文档翻译：中文↔英文/德文，保持格式与结构。支持 .docx / .txt / .xlsx 及文件夹、.zip。"
    )
    parser.add_argument("--input", "-i", required=True, help="输入：文件、文件夹或 .zip 路径")
    parser.add_argument("--output", "-o", help="输出目录（默认：单文件为同目录，目录/zip 为 xxx_translated）")
    parser.add_argument("--lang", "-l", default="en", choices=["en", "de", "zh"], help="目标语言：en 英文, de 德文, zh 中文")
    parser.add_argument("--collection", "-c", help="知识库名称，用于检索词条/法规/案例（与 aicheckword 一致）")
    parser.add_argument("--no-kb", action="store_true", help="不使用知识库，仅调用 LLM 翻译")
    parser.add_argument("--company-name", help="公司名称（目标语言，翻译时优先采用）")
    parser.add_argument("--company-address", help="地址（目标语言）")
    parser.add_argument("--company-contact", help="联系人（目标语言）")
    parser.add_argument("--company-phone", help="电话")
    args = parser.parse_args()

    company_overrides = None
    if any([args.company_name, args.company_address, args.company_contact, args.company_phone]):
        company_overrides = {k: v for k, v in {
            "company_name": args.company_name,
            "address": args.company_address,
            "contact": args.company_contact,
            "phone": args.company_phone,
        }.items() if v}

    try:
        out_paths = translate_path(
            args.input,
            output_dir=args.output,
            collection_name=args.collection or None,
            use_kb=not args.no_kb,
            target_lang=args.lang,
            company_overrides=company_overrides,
        )
        if not out_paths:
            print("未找到可翻译文件（支持格式: %s）。" % ", ".join(SUPPORTED_EXTENSIONS), file=sys.stderr)
            sys.exit(1)
        for p in out_paths:
            print(p)
    except FileNotFoundError as e:
        print("错误:", e, file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print("错误:", e, file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print("翻译失败:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
