"""AI 审核引擎：基于 RAG 的注册文档审核，支持 Ollama、OpenAI、Cursor Cloud Agents"""

import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict, replace

from .langchain_compat import Document, ChatPromptTemplate

from config import settings
from .review_throttle import wait_before_llm_call
from .knowledge_base import KnowledgeBase
from .document_loader import load_single_file
from .db import get_dimension_options, get_corrections_for_collection, get_review_extra_instructions
from .display_filename import is_probable_temp_upload_basename


def _registration_strictness_context(registration_type: str) -> str:
    """根据注册类别返回审核尺度说明。严格程度：Ⅲ > Ⅱb > Ⅱa > Ⅱ > Ι。"""
    if not registration_type:
        return ""
    if "三类" in registration_type or "Ⅲ" in registration_type:
        return (
            "\n\n【注册类别审核尺度·三类】本产品为**三类**医疗器械，审核尺度最严。"
            "须按三类风险与证据要求逐条核查；风险描述、适应症、禁忌症、临床证据、预期用途等均不得低于三类要求。"
            "若文档中存在证据不足、风险描述偏轻或与三类要求不符的内容，须作为审核点（合规性）明确列出并给出修改建议。"
        )
    if "Ⅱb" in registration_type:
        return (
            "\n\n【注册类别审核尺度·二类Ⅱb】本产品为**二类Ⅱb**，审核尺度严于二类/Ⅱa/一类。"
            "须按Ⅱb风险与证据要求核查；风险级别与证据强度不得低于Ⅱb。"
            "若文档中存在与Ⅱb要求不符或证据不足的内容，须作为审核点（合规性）明确列出并给出修改建议。"
        )
    if "Ⅱa" in registration_type:
        return (
            "\n\n【注册类别审核尺度·二类Ⅱa】本产品为**二类Ⅱa**，审核尺度严于二类/一类。"
            "须按Ⅱa风险与证据要求核查；风险描述、适应症、预期用途等不得超出Ⅱa范围。"
            "若文档中存在与Ⅱa要求不符的内容，须作为审核点（合规性）明确列出并给出修改建议。"
        )
    if "二类" in registration_type or "Ⅱ" in registration_type:
        return (
            "\n\n【注册类别审核尺度·二类】本产品为**二类**医疗器械，审核尺度严于一类。"
            "风险级别不能超过中等风险。文档中的风险描述、适应症、禁忌症、预期用途等不得超出中等风险范围。"
            "若文档中出现高于中等风险的表述（如高风险适应症、超出二类范围的用途）或与二类风险等级不符的内容，须作为审核点（类别为「合规性」）明确列出，并给出修改建议。"
        )
    if "一类" in registration_type or "Ι" in registration_type or "Ⅰ" in registration_type:
        return (
            "\n\n【注册类别审核尺度·一类】本产品为**一类**医疗器械，按一类风险与证据要求审核。"
            "若文档中存在与一类要求不符的内容，须作为审核点列出并给出修改建议。"
        )
    return ""


def _registration_strictness_multiplier(registration_type: Any) -> float:
    """注册类别严格程度系数，用于计算「最有价值条数」：Ⅲ > Ⅱb > Ⅱa > Ⅱ > Ι，越严条数越多。"""
    if not registration_type:
        return 1.0
    rt = (registration_type if isinstance(registration_type, str) else (registration_type[0] if registration_type else "") or "").strip()
    if not rt:
        return 1.0
    if "三类" in rt or "Ⅲ" in rt:
        return 1.5
    if "Ⅱb" in rt:
        return 1.35
    if "Ⅱa" in rt:
        return 1.25
    if "二类" in rt or "Ⅱ" in rt:
        return 1.15
    if "一类" in rt or "Ι" in rt or "Ⅰ" in rt:
        return 1.0
    return 1.0


def _country_strictness_multiplier(registration_countries: Optional[Any]) -> float:
    """注册国家严格程度系数，用于条数与提示词：美国 > 中国 > 欧盟，取所选国家中最高一档。"""
    if not registration_countries:
        return 1.0
    countries = (
        registration_countries
        if isinstance(registration_countries, (list, tuple))
        else [registration_countries]
    )
    mult = 1.0
    for c in countries:
        raw = str(c or "").strip()
        s = raw.lower()
        if not raw and not s:
            continue
        # 美国（最严）— 中文不参与 lower 比较，同时看 raw
        if raw == "美国" or "美国" in raw or s in ("usa", "us", "united states"):
            mult = max(mult, 1.18)
        # 中国
        elif raw == "中国" or "中国" in raw or s in ("china", "cn"):
            mult = max(mult, 1.10)
        # 欧盟
        elif raw == "欧盟" or "欧盟" in raw or s in ("eu", "eea") or raw.upper() == "EU":
            mult = max(mult, 1.04)
    return mult


def _country_strictness_context(registration_countries: Optional[Any]) -> str:
    """根据注册国家追加审核尺度说明：美国 > 中国 > 欧盟。"""
    if not registration_countries:
        return ""
    countries = (
        registration_countries
        if isinstance(registration_countries, (list, tuple))
        else [registration_countries]
    )
    labels = []
    for c in countries:
        raw = str(c or "").strip()
        s = raw.lower()
        if not raw:
            continue
        if raw == "美国" or "美国" in raw or s in ("usa", "us") or "united states" in s:
            if "美国" not in labels:
                labels.append("美国")
        elif raw == "中国" or "中国" in raw or s in ("china", "cn"):
            if "中国" not in labels:
                labels.append("中国")
        elif raw == "欧盟" or "欧盟" in raw or s == "eu" or raw.upper() == "EU":
            if "欧盟" not in labels:
                labels.append("欧盟")
    dedup = labels
    order_note = "本次注册国家审核严格程度按 **美国 > 中国 > 欧盟** 把握：面向美国注册时须按最严口径核查证据与表述；中国次之；欧盟在上述两档之下仍须满足当地法规，不得因「相对略宽」而遗漏明显缺陷。"
    language_note = ""
    if any(x in dedup for x in ("美国", "欧盟")):
        language_note = " 同时，若注册国家包含**美国或欧盟**，须专项核查待审文档语言是否为英文（或提供经受控翻译的英文版）；若仅为中文且无英文受控版本，应作为合规性/完整性问题列出并给出补充英文文档建议。"
    return (
        "\n\n【注册国家审核尺度】"
        f"当前维度涉及：{'、'.join(dedup)}。"
        + order_note
        + " 若文档同时面向多市场，按其中最严一档为底线，并兼顾各国别差异（语言、标签、临床证据要求等）。"
        + language_note
    )


def _project_form_focus_context(project_form: Optional[Any]) -> str:
    """项目形态侧重点（Web / APP / PC），P-2.1 / P-4.1 默认文案。"""
    if not project_form:
        return ""
    forms = (
        project_form
        if isinstance(project_form, (list, tuple))
        else [project_form]
    )
    forms = [str(f).strip() for f in forms if (f or "").strip()]
    if not forms:
        return ""
    parts = []
    for f in forms:
        fl = f.lower()
        if f.upper() == "WEB" or "web" in fl:
            parts.append(
                "**Web**：须重点核对部署/访问方式、浏览器与兼容性、URL 与域名、登录与权限、会话与安全、多租户或云端部署说明是否与文档一致；界面与操作流程描述须可对照实际系统。"
            )
        elif f.upper() == "APP" or "app" in fl or "移动端" in f:
            parts.append(
                "**APP**：须重点核对应用版本号、安装与升级路径、操作系统与权限（存储/相机/定位等）、应用商店或分发渠道相关描述、离线/推送等行为说明是否一致。"
            )
        elif f.upper() == "PC" or "pc" in fl or "桌面" in f:
            parts.append(
                "**PC**：须重点核对安装包与运行环境（OS 版本）、系统资源要求、安装/卸载步骤、与本地组件或服务联动说明是否一致。"
            )
        else:
            parts.append(f"**{f}**：请按该形态补充核对版本、安装/运行环境、权限与部署方式等与文档的一致性。")
    if not parts:
        return ""
    return (
        "\n\n【项目形态审核侧重点】\n"
        "本次项目形态为：" + "、".join(forms) + "。请按下述侧重点加大审核力度（合规性/一致性/完整性均可涉及）：\n"
        + "\n".join(parts)
    )


def _target_review_point_count(
    kb_size: int,
    registration_type: Optional[Any] = None,
    registration_countries: Optional[Any] = None,
) -> int:
    """与审核点知识库数量成正比；注册类别越严、注册国家越严（美国>中国>欧盟），条数越多。"""
    if kb_size <= 0:
        return 15
    base = 15
    step = 15
    inc = 2
    raw = base + (kb_size // step) * inc
    mult = _registration_strictness_multiplier(registration_type)
    mult *= _country_strictness_multiplier(registration_countries)
    target = max(15, min(200, int(raw * mult)))
    return target


def _create_llm():
    from .llm_factory import create_chat_llm
    return create_chat_llm(temperature=0.1)


def _normalize_text_for_dedup(s: str, max_len: int = 240) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    # 去掉常见标点，提升不同分块输出文本的去重命中率
    s = re.sub(r"[，。、“”‘’：:；;,.!?！？()（）\[\]【】<>《》\-_/\\]+", "", s)
    return s[:max_len]


def _infer_action_from_suggestion(suggestion: str) -> str:
    t = (suggestion or "").strip().lower()
    if not t:
        return ""
    no_change_flags = (
        "无需修改", "无需改动", "无须修改", "无需处理", "不需要修改", "保持现状", "仅提示", "供参考"
    )
    if any(k in t for k in no_change_flags):
        return "无需修改"
    postpone_flags = ("后续优化", "后续版本", "延期修改", "可择期", "建议后续")
    if any(k in t for k in postpone_flags):
        return "延期修改"
    return ""


def _estimate_english_ratio(text: str) -> float:
    """
    估计英文占比（忽略数字与大部分标点），用于抑制“英文文档却被说中文”的误报。
    注意：表格/编号/符号会稀释 A-Za-z 比例，所以阈值不能太苛刻。
    """
    blob = text or ""
    letters = re.findall(r"[A-Za-z]", blob)
    cjk = re.findall(r"[\u4e00-\u9fff]", blob)
    if not letters and not cjk:
        return 0.0
    return len(letters) / max(1, (len(letters) + len(cjk)))


def _cjk_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def _likely_english_document(text: str, declared_lang: str = "") -> bool:
    """
    判定是否“有效英文文档”：
    - 英文占比足够，且中文字符很少；或
    - 明确声明为 en
    """
    dl = (declared_lang or "").strip().lower()
    if dl == "en":
        return True
    blob = text or ""
    en_ratio = _estimate_english_ratio(blob)
    cjk_n = _cjk_char_count(blob)
    # 英文文档中可能夹杂少量中文（文件名、备注），但不应被判为中文
    if cjk_n <= 80 and en_ratio >= 0.70:
        return True
    if cjk_n <= 20 and en_ratio >= 0.55:
        return True
    return False


def _extract_ids_by_prefix(text: str, prefix: str) -> set:
    p = (prefix or "").strip().upper()
    if not p:
        return set()
    pat = re.compile(rf"\b{re.escape(p)}[\s_-]?(\d{{1,5}})\b", re.I)
    out = set()
    for m in pat.finditer(text or ""):
        out.add(f"{p}{int(m.group(1))}")
    return out


@dataclass
class AuditPoint:
    category: str
    severity: str
    location: str
    description: str
    regulation_ref: str
    suggestion: str
    modify_docs: List[str] = field(default_factory=list)  # 多文档审核时：需修改的文档名称列表
    action: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    file_name: str
    total_points: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    audit_points: List[AuditPoint] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "total_points": self.total_points,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "info_count": self.info_count,
            "summary": self.summary,
            "audit_points": [p.to_dict() for p in self.audit_points],
        }


REVIEW_SYSTEM_PROMPT = """你是一位资深的注册文档审核专家。你的职责是根据已训练的审核点清单（包括法规条款、检查方法和合规要求），
对提交的注册文档进行严格审核，逐一对照每个审核点，找出所有不符合要求的地方。

你具备以下能力：
1. 熟悉相关法规和标准要求及对应的审核点
2. 能够识别文档中的合规性问题
3. 能够检查文档的完整性和一致性
4. 能够给出专业的、可操作的修改建议

审核时请关注以下维度：
- **合规性**：是否符合相关法规、标准要求
- **完整性**：必要信息是否齐全，是否有遗漏；当上下文提供【历史案例文档章节参考】时，须对比待审文档的章节结构，缺失的章节须作为「文档内容完整性」审核点列出，location 指明缺失的章节名称或应出现的位置
- **一致性**：文档内部数据和表述是否前后一致
- **准确性**：技术参数、数据引用是否准确
- **格式规范**：文档格式是否符合要求
- **翻译正确性**（仅英文文档）：是否符合目标国家/语言语法习惯、是否通顺、是否符合逻辑；须结合词条、法规与英文案例参考

**优先加强核查（文档中较易暴露问题，须提高优先级）**：
- **编审批人员逻辑**：编制、审核、批准职责是否清晰，签批栏与受控文件要求是否一致，审批流程与角色是否前后矛盾或缺失。**注意**：Word/Excel/PDF 转文本时常丢失「签名/日期」嵌入图；若待审文档末尾出现以 **【Word 版式检测** / **【Excel 嵌入图像** / **【PDF 嵌入图像** 开头的版式检测说明且表明含嵌入图片，视为可能已图片签署，勿报签署空白。
- **岗位与人员（全文）**：**文档中出现的所有岗位及人员**（含编审批签批栏及其他章节出现的任何岗位、人员）均须核查：**与岗位名称对应职责**（人员与其岗位名称、该岗位定义职责是否匹配）、**花名册符合性**（是否与组织花名册或受控人员清单一致）；未在册人员、岗位与花名册不符、人员与岗位职责不匹配等须作为合规性/一致性审核点列出。
- **设备编号与设备设施清单（质量体系，优先）**：文档正文、表格、附录中出现的**设备/工装/设施编号**（含仪器编号、资产编号、设备代码等）须与同一文档或相关附件中的**设备设施清单、台账、校准/验证对象清单**等逐项一致：不得出现清单外编号、清单有而正文未使用却声称使用、同一编号对应不同设备名称/型号/位置等矛盾。若上下文或知识库中提供了清单或程序文件要求，须对照核查。
- **设备编号规则与程序文件（质量体系，优先）**：设备编号的**命名规则、前缀/分段含义、编号长度与格式**须与受控**程序文件**（如《设备编号管理规定》《标识与可追溯性控制程序》《设备管理程序》等，以组织实际文件名为准）一致；不得自创与程序不符的编码格式、随意省略规则要求的字段或混用已废止规则。
- **时间线逻辑**：文档中出现的日期、版本号、生效日期、修订履历、各章节时间表述是否前后一致、顺序是否合理，是否存在「未生效已引用」「修订日期倒序」等明显矛盾；同一份文件内多个时间点是否形成合理时间线。若为医疗器械软件相关文档，须结合软件生命周期（需求/设计/开发/测试/发布等阶段）核查文档内时间与阶段表述是否合理。
- **需求—开发—测试—风险可追溯性（仅单文档内，医疗器械软件）**：在**当前这一份**待审文档范围内，核查**追溯标识**（用户需求 ID/REQ、软件需求 ID、风险 ID、**CS 编号**、危害编号、测试用例/报告编号、追溯矩阵行/列引用等）是否**自洽、闭环**：正文、表格与矩阵中对**同一 ID** 的表述是否一致，是否存在「有 ID 无对应条目」「同一 ID 在本文档内指向矛盾」「应追溯却断链」。**不得**编造「未上传其他文档」类理由来输出**跨文件**追溯结论；**多份注册文档之间**同一追溯编号是否对齐、是否与《软件可追溯性管理程序》等制度一致，由系统中**「跨文档可追溯性审核」**专项执行，**不在**单文档审核任务中展开。若上下文或知识库提供**可追溯性受控程序文件**摘录，可审核**本文档**中的追溯格式、矩阵必备要素、签批等是否与程序要求相符。
- **医疗器械软件：器械法规与软件工程双重要求（优先）**：对独立软件/SaMD 相关文档，须**同时**从**医疗器械监管侧**（适用范围与预期用途、风险管理与残余风险、说明书/标签要点、质量管理体系对软件相关活动的受控要求等——以项目资料与参考知识可核对者为准）与**软件生命周期工程侧**（需求、设计、实现、验证与确认、配置与变更、发布、网络安全与数据、现成软件/SOUP 等——以本文档实际涉及章节为准）审视；若明显只覆盖一侧而另一侧存在**关键缺口、矛盾或缺少应有关卡**，须作为合规性/完整性/一致性审核点列出。在 regulation_ref 中引用具体法规或标准条款时，须与上下文或审核点知识库中**可核对**的来源一致，**禁止虚构**条款号或通告号。

**输出要求**：
- 每个问题必须指明**在文档中的具体位置**（如「第3章 适用范围」「第5页 技术指标」或引用有问题的那句话/段落），不得笼统写「文档中」「全文」。
- **页码约束（严禁误报）**：只有当待审内容中**明确出现页码标记**（如“Page 3 of 10”“第3页/共10页”等）或系统提供了 PDF 页信息时，才允许在 location 中写“第X页”。若为 Word/Excel 转文本且正文未含页码信息，必须用「章节/表格标题/表头行/编号」等定位，**禁止**虚构页码。
- 每条**修改建议**必须可操作（如「将第2段中的“XXX”改为“YYY”」「在 4.1 节补充……」「删除第6页重复表述」），不得笼统写「请完善」「请补充」。
- 每个审核点只针对一处具体问题，避免一条里混入多个不相关问题。
- 审核发现须**覆盖全面**且**精准不重复**：对照参考知识中的各维度与条款逐项核查，不同条目的发现不得语义重复或冗余。
- 当上下文同时包含【本次审核维度】或【项目专属要求】时，须按**项目专属审核**处理：严格按项目信息、适用范围与项目资料进行一致性核对，不得仅做泛化合规表述而忽略与项目资料的逐条对照。"""

REVIEW_USER_PROMPT = """请根据以下审核点知识对待审核文档进行审核。

## 审核点参考知识（审核点清单 / 法规 / 纠正经验）

{context}

## 待审核文档内容

文件名：{file_name}

{document_content}

## 审核要求

请逐项审核以上文档，输出所有审核发现。请依据上述审核点参考知识**全面覆盖**各维度（合规性、完整性、一致性、准确性、格式规范等），每条输出对应**不同**的审核发现，避免语义重复或冗余表述。
每个审核点必须严格按以下 JSON 格式输出：

```json
[
  {{
    "category": "审核类别（合规性/完整性/一致性/准确性/格式规范/翻译正确性，英文文档须含翻译正确性）",
    "severity": "严重程度（high/medium/low/info）",
    "location": "问题在文档中的具体位置（必须具体：如章节号、段落、页码，或引用有问题的那句话，不得写「文档中」「全文」）",
    "description": "问题详细描述",
    "regulation_ref": "对应的法规或标准条款引用",
    "suggestion": "具体可操作的修改建议（必须写明改哪里、改成什么，如「将第2段“XXX”改为“YYY”」，不得笼统写「请完善」）"
  }}
]
```

请确保：
1. 仅输出 JSON 数组，不要输出其他内容
2. **location** 必须指明具体位置（章节/页码/引用原文），不能笼统
3. **suggestion** 必须可操作（写明改哪里、改成什么），不能笼统
4. 至少检查合规性、完整性、一致性三个维度
5. 若上下文提供了【本次审核维度】【项目基本信息】或【项目专属要求】，须核对待审文档中的项目名称、产品名称、型号规格等与上述约定及项目资料一致；不一致须作为一致性审核点列出，并指明应对齐至哪一段约定内容
6. 若上下文提供【历史案例文档章节参考】，须进行「文档内容完整性」审核：须**逐条对照**参考章节清单，在待审文档中逐项确认是否存在；每缺失一条应有章节即单列一条完整性审核点，location 写明缺失的章节名称或应出现位置，不得合并笼统描述
7. **若待审文档为英文**：须增加「翻译正确性」审核维度，包括：是否符合目标国家/语言语法习惯、是否通顺、是否符合逻辑；审核时请参考上下文中的词条、法规及英文案例，审核点类别可包含「翻译正确性」
8. 如果文档整体合规，也请输出 info 级别的确认信息
9. **编审批与时间线（优先）**：对文档中出现的编制/审核/批准角色、签批与时间、版本与修订日期等须重点排查；发现职责不清、日期或版本前后矛盾时，优先按一致性或格式规范输出，severity 可适当提高。**签批形式**：若待审文档正文末尾的版式检测说明（**Word / Excel / PDF 任一类**）指出签批相关区域**含有嵌入图片或对象**（手写签名扫描图、电子签、日期图等），应视为已完成签署或已占位，**不得**再输出「签署空白」「未签名」「签批栏为空」等审核点，除非法规或受控文件明确要求必须为**可检索文本姓名**且无任何图像签署形式。
10. **岗位与人员（全文）**：对**文档中出现的所有岗位及人员**（含编审批签批栏及正文各章节中的任何岗位、人员）须核查**与岗位名称对应职责**、**花名册符合性**；人员与岗位职责不匹配、未在册人员、岗位与花名册不符等须作为合规性/一致性审核点列出。
11. **设备编号与设备设施清单（优先）**：提取或识别文档中出现的**设备/工装/设施编号**，与文内**设备设施清单、台账、校准与验证设备列表**等对照：须**逐项一致**，不得遗漏、重复矛盾或「清单无而正文有」「清单有而用途不符」。若多份资料同时出现，须交叉核对。
12. **设备编号规则与程序文件（优先）**：设备编号**格式、前缀、分段规则**须与上下文或知识库中引用的**受控程序文件**对编号的规定一致；若文档未说明依据而编号明显不符合常规规则（如程序要求三段编码却仅两段），须作为一致性或格式规范审核点列出并建议对照程序修订。
13. **历史误报与纠正记录**：若审核点参考知识中含 **【历史误报与纠正记录】**，须严格遵守：对已标记**误报**的同类问题**不得再报**；对**弃用/修订**项勿重复输出已被否定的原表述。
14. **需求—开发—测试—风险可追溯性（仅单文档内）**：核查**当前文档内**用户需求/软件需求/风险/CS/测试编号及追溯矩阵引用是否自洽、闭环；发现 ID 在本文档内矛盾、断链、「有号无文」须作为一致性或完整性审核点列出。**不要**输出须依赖「其他未提供文件」才能判断的跨文档追溯结论；多文档间编号对齐请由用户执行**「跨文档可追溯性审核」**。
15. **可追溯性与程序文件符合性（单文档）**：若参考知识中含**可追溯性程序文件**摘录，可审核**本份待审文档**中的追溯标识规则、矩阵结构、必备栏目、更新与签批等是否与程序规定相符；不符合须作为合规性或一致性审核点。**跨文档**与程序的整体符合性以「跨文档可追溯性审核」为准。
16. **欧盟/美国注册文档语言（优先）**：若【本次审核维度】中的注册国家包含**欧盟或美国**，须核查待审文档是否为英文版本，或是否提供受控且可追溯的英文版本/翻译件；如仅有中文且无英文受控版本、或中英文关键术语不一致，须作为合规性或一致性审核点列出，并在 suggestion 中写明需补充/统一的英文文档与术语。
17. **医疗器械软件双重要求（优先）**：对软件类注册文档，核查是否**同时**体现器械监管语境下的安全与风险证据、以及软件工程语境下的生命周期与验证证据；任一侧明显缺失或与另一侧矛盾时须列审核点。法规/标准引用须可核对，禁止虚构条款号。
18. **验证与确认层次与测试证据（医疗器械软件，优先）**：核查文档中单元/集成/系统/确认等测试层次是否界定清楚、记录是否足以支撑结论（在文档类型范围内）；与需求/风险级别明显不相称时列出审核点。"""

SUMMARY_PROMPT = """请根据以下审核发现，生成一段简洁的审核总结（200字以内）：

文件名：{file_name}
审核发现数量：高风险 {high} 个，中风险 {medium} 个，低风险 {low} 个，提示 {info} 个

审核详情：
{details}

请用中文输出总结。"""

CURSOR_REVIEW_TASK = """你是一位资深的注册文档审核专家。请根据下面的审核点参考知识和待审核文档内容，仅在你的回复中输出一个 JSON 数组，不要修改任何代码或文件。

## 审核点参考知识
{context}

## 待审核文档
文件名：{file_name}

{document_content}

## 输出要求
1. 仅输出一个 JSON 数组，格式如下，不要其他说明或 markdown 标记：
[{{"category":"合规性|完整性|一致性|准确性|格式规范|翻译正确性（英文文档须含此项）","severity":"high|medium|low|info","location":"在文档中的具体位置（必须具体：章节/页码或引用有问题的那句话，不得写「文档中」「全文」）","description":"问题描述","regulation_ref":"法规引用","suggestion":"可操作的修改建议（必须写明改哪里、改成什么，不得笼统写「请完善」）"}}]
2. **location** 必须指明具体位置（如「第3章」「第5页」「第2段中“……”」），不能笼统。
3. **suggestion** 必须可操作（如「将……改为……」「在 4.1 节补充……」），不能笼统。
4. 请依据审核点参考知识**全面覆盖**各维度，每条发现对应**不同**问题，避免语义重复或冗余。
5. 若审核点参考知识中包含【本次审核维度】【项目基本信息】或【项目专属要求】，须核对待审文档与上述约定及项目资料一致；不一致须作为一致性审核点列出。
6. 若审核点参考知识中包含【历史案例文档章节参考】，须**逐条对照**参考章节与待审文档；每缺失一条应有章节即单列一条完整性审核点，location 指明缺失章节或位置。
7. **编审批与时间线（优先）**：编制/审核/批准与时间、版本、修订日期等须重点排查；矛盾处优先输出，severity 可适当提高。若 Word/Excel/PDF 版式检测说明签批处含**嵌入签名/日期图片**，勿报「签署空白」。
8. **岗位与人员（全文）**：文档中出现的**所有岗位及人员**（含编审批及其他章节）须核查与岗位名称对应职责、花名册符合性；不符时作为审核点列出。
9. **设备编号与设施清单及程序一致性**：设备/工装/设施编号须与文内**设备设施清单**一致；编号规则须与**程序文件**约定一致；矛盾时作为一致性审核点列出。
10. **需求—开发—测试—风险可追溯性（仅单文档内）**：在**当前文档**内核查追溯 ID、矩阵与正文是否自洽；**不要**输出须依赖其他文件才能完成的跨文档追溯结论。多文档编号对齐由**「跨文档可追溯性审核」**处理。
11. **欧盟/美国注册文档语言**：若本次注册国家包含欧盟或美国，须核查待审文档是否为英文（或有受控英文版/翻译件）及关键术语中英文一致性；不满足时须作为合规性或一致性审核点列出并给出补充建议。
12. **医疗器械软件双重要求**：同时审视器械监管证据与软件生命周期/验证证据是否齐全、是否自相矛盾；法规引用须可核对。
13. **验证与确认与测试证据**：测试层次与记录是否足以支撑需求与风险级别。
"""

CURSOR_SUMMARY_TASK = """请根据以下审核发现，用中文生成一段简洁的审核总结（200字以内）。不要修改任何文件，仅输出总结文字。

文件名：{file_name}
审核发现：高风险 {high} 个，中风险 {medium} 个，低风险 {low} 个，提示 {info} 个

审核详情：
{details}
"""

# 多文档一致性与模板风格审核：跨文档信息一致性、模板/风格一致性
MULTI_DOC_CONSISTENCY_PROMPT = """你是一位资深的注册文档审核专家。当前已对多份注册文档完成单文档审核，现需进行**跨文档**的补充审核，重点检查：

## 一、信息一致性
- 项目名称、产品名称、型号规格、注册单元名称在各文档中是否完全一致（含全称/简称、空格、标点）。
- **同一类信息在多份文档中的一致性（必须逐项核对）**：凡同一类信息在多于一份文档中出现时，须核对这些信息在各文档间是否一致；不一致须作为审核点列出并写明涉及文档与修改建议。重点包括但不限于：**预期用途**、**适用范围**、**预期用户/适用人群**、**工作原理**、**物理拓扑图/网络拓扑**、**体系结构图/软件架构**、**现成软件（含组件、版本、供应商）**、**风险（含风险分析、风险措施）**、**术语/缩略语**、**运行环境（硬件、软件、网络）**、关键技术指标、性能参数、禁忌症、引用标准、日期与版本号等。若某文档缺失本应在多文档中统一表述的条目，或表述与其它文档矛盾，须作为一致性审核点列出。
- **设备编号与设备设施清单（跨文档）**：各文档中出现的**设备/工装/设施编号**须相互一致，且与**设备设施清单/台账/验证校准设备列表**（任一份本批文档或约定附件）**逐项对齐**；同一编号在不同文档中的设备名称、型号、安装位置、用途描述不得矛盾；不得出现仅在一份文档中出现的「孤立编号」而无法与清单对应的情况（除非文档明确其为临时/外部编号并说明）。
- **设备编号规则与程序文件（跨文档）**：若本批含**程序文件、质量手册附录或编号管理规定**与**记录类/报告类文档**，须核对记录中出现的设备编号**是否符合程序文件规定的编码规则**（前缀、位数、分段含义等）；程序已废止旧规则而记录仍用旧格式，或不同文档对同一规则理解不一致，须作为一致性或格式规范审核点列出。
- 日期、版本号、引用标准等是否一致。
- **岗位与人员（全文）**：各文档中**出现的所有岗位及人员**（含编审批签批栏及正文各章节）与岗位名称对应职责是否一致、是否与组织花名册（或受控人员清单）符合；跨文档间岗位与人员表述是否一致。
- **签批与图片**：若某文档正文含 Word/Excel/PDF 的**嵌入图像版式检测**说明且指出存在签批相关嵌入图，勿再报该文档「签署空白」；跨文档比较时勿因纯文本未含手写姓名字符而认定未签。
- **跨文档追溯编号与追溯矩阵（本任务不包含）**：**REQ/软件需求 ID、风险 ID、CS 编号、单元/系统测试用例编号**在各文档间是否**指向同一对象**、追溯矩阵是否闭环、是否符合组织可追溯性管理制度——**不在本「多文档一致性」任务中审核**，以免与专项重复；此类问题**一律不得**写入下方 JSON。跨文档追溯须由系统**「跨文档可追溯性审核」**专项执行。本任务仍可就**风险描述、控制措施文字表述**等与产品信息相关的**语义/数据一致性**在上文「同一类信息」范围内提出审核点，但**不得**展开「同一编号在多文档间是否对齐」式追溯链核对。

## 二、时间与阶段（医疗器械软件生命周期，优先核查）
- **同一份文件内**：文档中出现的多个日期、版本号、生效日期、修订履历、各章节时间表述是否前后一致、顺序是否合理；是否存在「未生效已引用」「修订日期倒序」「时间逻辑矛盾」等。
- **不同阶段的多份文件之间**：按医疗器械软件生命周期（如需求—设计—开发—测试—发布等阶段）核对待审多份文档之间的时间与阶段是否合理；阶段先后是否与文档类型匹配，跨文档的版本号、发布日期、修订日期是否形成合理时间线，是否存在阶段倒置或时间矛盾。
- 发现时间线或阶段不合理时，须作为一致性/格式规范审核点列出，location 写明涉及文档与具体位置，suggestion 写明应如何修正。

## 三、模板与风格一致性
- 标题层级（如 1 / 1.1 / 1.1.1）、章节编号方式是否统一。
- 术语用词是否统一（如同一概念在不同文档中是否用同一表述）。
- 格式风格（单位、数字与单位间空格、列表格式等）是否统一。

## 本批文档列表与摘要（下方标题即为各文档名称，输出时请直接使用这些名称，勿用「文档1」「文档2」）
{docs_summary}

## 审核要求
1. **location**：必须用**各文档的真实名称**说明涉及哪些文档或位置（如「[说明书] 与 [技术要求] 中产品名称不一致」），不要写「文档1」「文档2」。
2. **suggestion**：修改建议必须写明**需修改哪一份或哪几份文档**及具体改法（如「在 [说明书] 与 [用户手册] 中统一将“XXX”改为“YYY”」）。
3. 每个审核点增加 **modify_docs** 字段：需修改的文档名称数组，如 ["说明书", "技术要求"]。
4. **禁止**输出跨文档**追溯编号/矩阵闭环**类审核点（见上文「跨文档追溯编号与追溯矩阵」说明）；仅输出信息一致性、时间阶段、模板与风格相关发现。

请仅输出一个 JSON 数组，不要其他说明或 markdown 标记：
[{{"category":"一致性|格式规范","severity":"high|medium|low|info","location":"用文档名称说明涉及哪些文档或位置，勿用文档1/文档2","description":"问题描述","regulation_ref":"相关法规或标准（可选）","suggestion":"具体可操作的修改建议，并写明需修改哪份或哪几份文档","modify_docs":["文档名称1","文档名称2"]}}]
若各文档间信息与风格均一致，可输出一条 info 级别确认；若存在不一致或风格不统一，逐条列出。"""

# 跨文档可追溯性专项审核（单独按钮）：避免单文档审核时模型误以为「无其他文档」
# 与「多文档一致性与模板风格」分工：后者不再承担 REQ/风险/CS/测试编号跨文档追溯链核对。
TRACEABILITY_CROSS_DOC_PROMPT = """你是一位资深的医疗器械软件注册文档审核专家。当前任务**仅**做**跨文档可追溯性**审核。

## 硬性约束（必须遵守）
1. 下方已提供 **{doc_count}** 份文档的全文或长摘录，文档名称清单为：**{doc_names}**。（Excel 按工作表分段；每段以「【Excel 工作表：…】」标明表名，须通读所有分段——CS/风险需求等编号可能在非首表。**Word 软件需求类文档**若含「【Word 表格结构化摘录」段，其中按单元格列出的 **URS/REQ/SRS 表列** 与正文具有同等效力，核对追溯编号时必须检索该段，不得以「正文未出现」为由漏判。）
2. **禁止**以「未上传其他文档」「仅有单份文档」「无法跨文档核对」等理由拒绝审核或搪塞；你必须基于已给出的多份文档摘录进行交叉核对。
3. 若某追溯 ID 在摘录中未出现，可输出「在已提供摘录中未检索到该 ID」类 **info**，但不得声称「用户未提供文档」。
3a. **编号必须逐字与摘录一致**：不得改写、归一化或「纠错」。摘录为 **URS001** 时不得写成 URS1/URS-001；摘录为 **CE01** 时不得写成 CS01/CS0/**CS**（CE 与 CS 为不同前缀，禁止混用）。引用任何 ID 时请从摘录中**原样复制**。若摘录中出现 **「【以修订后为准】CSxx（已废止 Cyyy/CSzzz）」**，表示 Word 修订或粘连文本已预处理：**仅 CSxx 为当前生效编号**，审核须以 CSxx 为准，**不得**将 Cyyy 与 CSxx 合并解读为 CS012 等虚构编号。
3b. **风险分析类 Excel**：名称含「风险分析」**或英文「Risk Analysis」/「Risk Management」等**的工作表已尽量排在摘录较前；须在「【Excel 工作表：…】」段内按**表头行**定位列：**中文「风险需求」**或**英文「Risk demand (Measure) ID」/「Risk demand」**等同义列，从中读取 **CS** 等编号原文，不得仅凭表名臆测列位置。
3c. 每份文档前的**「自动扫描得到的追溯相关编号索引」**由程序基于**全文**提取，用于弥补正文摘录可能截断中间表格；**若索引中已列出某 CS/HC**，不得再声称「摘录中只有 CS01–CS08」等漏检，除非正文表格与索引矛盾（以表格单元格原文为准）。
4. **location** 必须用下方「### 文档真实名称」中的名称标注（如「[软件需求规范.docx] 第3.2节 REQ-012」），禁止写「文档1」「文档2」。
5. 每个审核点必须包含 **modify_docs** 数组：需要修改的文档显示名称列表（与上方 ### 标题一致）。
6. **category** 优先使用：一致性、完整性、合规性（必要时可用准确性）；**勿**用「翻译正确性」除非明确是术语双语对照问题。
7. 请结合【可追溯性制度与审核点知识库参考】中的程序/清单条款（若有）判断本批项目文档是否符合组织规定；若无具体程序条文，则按医疗器械软件通用追溯要求（YY/T 0664、注册审查指导原则等）执行，并在 regulation_ref 中写明依据。
8. **若摘录中能识别出《软件需求规范》/SRS/需求规格类文档与《风险管理/风险分析》类文档**：你必须**逐项**核对需求文档中**追溯表、矩阵或附录**里填写的 **CS 编号**（及风险 ID、危害编号等，以文档实际列名为准）是否在风险分析文档中存在**相同编号**，且**控制措施/风险描述的含义一致**（允许表述缩写，但不得张冠李戴或指向不同危害）。SRS 表中出现而风险文档中**无对应 CS**、或**同一 CS 在两处文字明显矛盾**的，须**逐条**列为审核点（勿合并为一句「若干不一致」）。
9. **《软件可追溯性分析报告》/追溯矩阵类文档**（文件名或章节含可追溯性、追溯矩阵、RTM 等）：须将其中的 **需求—设计—实现—测试—风险** 各列（或等价结构）与 **SRS 追溯表**、**风险分析中的 CS/风险项**、**架构/详细设计**、**单元/系统测试用例** 中的**同一编号**交叉核对：**一行关系必须在各文档中都能找到对应条目且语义一致**；矩阵声称的链接若在某文档中无对应 ID 或内容对不上，须单列审核点并写明矩阵位置与缺失/矛盾文档。
9a. 若本批包含「软件可追溯性分析报告/RTM」的**文字描述段落**（非表格），也必须核对其对系统功能、适用范围、风险控制措施与需求来源的叙述，是否与【产品适用范围】与《软件需求规范/SRS》一致；出现范围越界、功能描述与 SRS 矛盾、或追溯报告声称的需求来源与 SRS 不匹配的，须作为一致性审核点列出，并点名相关段落与 SRS 对应章节/表格行。
10. **执行顺序建议（须在推理中落实，并在 description 中体现核对依据）**：先从各文档摘录中提取所有出现的 **CS、REQ/SR、风险编号、测试用例编号** 等形成**心智对照**，再按「SRS 追溯表 → 风险分析 → 可追溯性矩阵 → 设计 → 测试」顺序验证**全链路一致**；禁止只写笼统结论而不点名具体编号与表格位置。

## 审核重点（跨文档）
- 用户需求 ID / 软件需求 ID（REQ、SRS 等）在需求类、设计类、测试类文档间引用是否一致，描述是否指向同一需求内容。
- 风险 ID、危害编号、**CS 编号**（或组织规定的控制措施编号）在风险管理、**SRS 追溯表**、设计、测试之间是否一致，控制措施文字是否匹配（**本条为强制核对项，不得遗漏**）。
- **软件可追溯性分析**文档中的矩阵/文字与 SRS、风险、设计、测试对**同一编号**的指向是否一致，全链路是否闭环。
- 单元测试用例、系统测试用例编号与设计项/需求项/风险项的对应关系是否闭环；是否存在「有号无文」「同一编号不同文档含义不同」「断链」。
- 若某文档声称「见追溯矩阵/可追溯性分析报告」但摘录中出现矩阵片段，须核对矩阵列项与引用 ID 是否一致。

## 本批文档摘录（标题即文档显示名称）
{docs_summary}

## 可追溯性制度与审核点知识库参考
{traceability_kb}

请仅输出一个 JSON 数组，不要其他说明或 markdown 标记：
[{{"category":"一致性|完整性|合规性|准确性","severity":"high|medium|low|info","location":"[真实文档名] 章节/表格/页码/具体ID","description":"问题描述（写明涉及哪些追溯编号）","regulation_ref":"知识库程序条款或法规/指导原则","suggestion":"可操作的修改建议，写明需改哪几份文档、如何对齐编号或补链","modify_docs":["文档显示名称1","文档显示名称2"]}}]
**输出要求补充**：当摘录中同时出现需求类文档（含追溯表）与风险分析类文档时，**至少**应包含对 **CS（及表中出现的风险相关编号）跨文档对齐情况**的审核结论——要么列出具体矛盾/断链点，要么用 **info** 明确说明「在已提供摘录范围内已核对所列 CS，与风险分析一致」并列举已核对样本编号（不少于 3 个，若摘录中不足 3 个则列全部）。《软件可追溯性分析》类文档出现时同理，须体现与 SRS/风险/测试的矩阵一致性结论。"""


def _excerpt_traceability_document(text: str, max_len: int) -> str:
    """长需求类文档的 URS/追溯表多在文末；只取前 max_len 字会漏检。超长时采用「首部 + 尾部」拼接。"""
    t = (text or "").strip()
    if not t or len(t) <= max_len:
        return t
    bridge = (
        "\n\n……【摘录说明】中间部分已省略；以下为文档末尾（常含附录追溯表、URS 编号列、可追溯性矩阵）……\n\n"
    )
    if max_len <= len(bridge) + 80:
        return t[:max_len]
    budget = max_len - len(bridge)
    head_n = budget // 2
    tail_n = budget - head_n
    return t[:head_n] + bridge + t[-tail_n:]


_TRACE_ID_SCAN_PATTERNS = (
    r"\bHC[\s_-]?\d{1,4}\b",
    r"\bCS[\s_-]?\d{1,4}\b",
    r"\bCE[\s_-]?\d{1,5}\b",
    r"\bC[\s_-]?\d{3}\b",  # 如 C201（三位数字；不与 CSxx 冲突）
    r"\bURS[\s_-]?\d{1,8}\b",
    r"\bSR[\s_-]?\d{1,6}\b",
    r"\bREQ[\s_-]?\d{1,6}\b",
)


def _normalize_trace_scan_token(raw: str) -> str:
    s = (raw or "").strip().upper()
    # 统一去掉常见分隔符：CS 11 / CS-11 / CS_11 -> CS11
    s = re.sub(r"[\s_-]+", "", s)
    return s


def _traceability_id_sort_key(token: str):
    m = re.match(r"^([A-Z]+)(\d+)$", token)
    if m:
        return (m.group(1), int(m.group(2)))
    return (token, 0)


def _traceability_ids_registry_priority_order(found: set) -> List[str]:
    """
    风险分析 Excel 中 CS/HC 是追溯核心；按纯字典序会把 C201、CE、REQ 等排在 CS 前，
    在索引长度上限内先占满额度，导致只显示到 CS09 而 CS10+ 被截掉。故 **CS 全量优先，其次 HC**，再其余。
    """
    cs_items: List[Tuple[int, str]] = []
    hc_items: List[Tuple[int, str]] = []
    rest: List[str] = []
    for raw in found:
        t = (raw or "").strip().upper()
        m = re.match(r"^CS(\d+)$", t)
        if m:
            cs_items.append((int(m.group(1)), t))
            continue
        m = re.match(r"^HC(\d+)$", t)
        if m:
            hc_items.append((int(m.group(1)), t))
            continue
        rest.append(t)
    out = [p[1] for p in sorted(cs_items)]
    out.extend(p[1] for p in sorted(hc_items))
    out.extend(sorted(rest, key=_traceability_id_sort_key))
    return out


def _traceability_id_registry_block(text: str, max_chars: int = 1800) -> str:
    """
    从**全文**扫描常见追溯编号，生成短索引置于摘录最前。
    解决：Excel 多 sheet 时「首部+尾部」摘录过短，中间 Risk Analysis 表里的 CS09+ 进不了上下文导致误报漏检。
    """
    blob = text or ""
    if not blob.strip():
        return ""
    found = set()
    for pat in _TRACE_ID_SCAN_PATTERNS:
        try:
            for m in re.finditer(pat, blob, re.I):
                found.add(_normalize_trace_scan_token(m.group(0)))
        except re.error:
            continue
    if not found:
        return ""
    ordered = _traceability_ids_registry_priority_order(found)
    cap = max(200, min(max_chars, 6000))

    def pack_tokens(tokens: List[str], label: str, budget: int) -> Tuple[str, int]:
        if not tokens:
            return "", budget
        head = label + "、".join(tokens)
        if len(head) + 1 <= budget:
            line = head + "\n"
            return line, budget - len(line)
        items: List[str] = []
        cur = len(label)
        for item in tokens:
            sep = 1 if items else 0
            if cur + sep + len(item) > budget - 40:
                items.append(f"…共{len(tokens)}项")
                break
            items.append(item)
            cur += sep + len(item)
        line = label + "、".join(items) + "\n"
        return line, max(0, budget - len(line))

    cs_only = [x for x in ordered if re.match(r"^CS\d+$", x)]
    hc_only = [x for x in ordered if re.match(r"^HC\d+$", x)]
    other = [x for x in ordered if x not in set(cs_only) and x not in set(hc_only)]

    # 先输出 CS、HC 两段（风险分析核心），再输出其他，避免 C201/REQ 等占满额度挤掉 CS10+
    body_parts: List[str] = []
    remaining = cap
    if cs_only:
        line, remaining = pack_tokens(cs_only, "【CS】", remaining)
        if line.strip():
            body_parts.append(line.rstrip())
    if hc_only:
        line, remaining = pack_tokens(hc_only, "【HC】", remaining)
        if line.strip():
            body_parts.append(line.rstrip())
    if other and remaining > 80:
        prefix = "【其他】"
        items: List[str] = []
        cur_len = len(prefix)
        for item in other:
            sep = 1 if items else 0
            if cur_len + sep + len(item) > remaining - 40:
                items.append(f"…其余共{len(other)}项")
                break
            items.append(item)
            cur_len += sep + len(item)
        body_parts.append(prefix + "、".join(items))

    block = "\n".join(body_parts)
    intro = (
        "【本文件自动扫描得到的追溯相关编号索引（防摘录截断遗漏；**核对以表格单元格原文为准**）】\n"
        "说明：勿将**相邻列/相邻单元格**拼成新编号（如左侧 C201、右侧 CS02 **不得**读成 CS012）；"
        "正文若含「【以修订后为准】CSxx」系程序对修订粘连的标注，**仅 CSxx 为当前生效编号**，勿再引用已废止的旧号做审核结论。"
        "勿将 **CE** 与 **CS** 混淆；索引与正文不一致时以正文表格为准。\n"
        "**【CS】/【HC】段须列全**（同一工作簿内全部检出项）；不得以「只看到 CS01–CS09」为由漏报 CS10+。\n"
    )
    return intro + block


def _excerpt_traceability_document_with_registry(text: str, max_len: int) -> str:
    """先附编号索引（基于全文扫描），再对正文做首部+尾部摘录，避免 CS09 等落在被截断的中间段。"""
    full = (text or "").strip()
    if not full:
        return ""
    reg_budget = min(5200, max(1400, max_len * 3 // 5))
    reg = _traceability_id_registry_block(full, max_chars=reg_budget)
    reserve = len(reg) + 4 if reg else 0
    inner_max = max(480, max_len - reserve)
    if len(full) <= inner_max:
        inner = full
    else:
        inner = _excerpt_traceability_document(full, inner_max)
    return (reg + "\n\n" + inner) if reg else inner


class DocumentReviewer:
    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        collection_name: str = "regulations",
        feedback_knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.collection_name = collection_name
        self.kb = knowledge_base or KnowledgeBase(collection_name)
        # 误报/纠正独立向量库（与第二步审核点清单库区分）；可为 None（测试或旧调用）
        self.feedback_kb = feedback_knowledge_base
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _create_llm()
        return self._llm

    def reset_client(self):
        self._llm = None

    def _retrieve_context(
        self,
        document_text: str,
        top_k: int = 15,
        registration_type: Optional[Any] = None,
        registration_countries: Optional[Any] = None,
        for_provider: Optional[str] = None,
    ) -> str:
        """筛选审核点知识库：先检索全库（或足够多候选），按注册类别过滤、去重、按类别多样化，再取与知识库规模及注册类别/国家严格程度成正比的最有价值前 N 条。"""
        pv = (for_provider or getattr(settings, "provider", "") or "").strip().lower()
        try:
            kb_size = self.kb.get_collection_stats().get("document_count", 0) or 0
        except Exception:
            kb_size = 0
        target_n = _target_review_point_count(
            kb_size, registration_type, registration_countries=registration_countries
        )
        # DeepSeek 多文件审核时，过大 fetch_k 会显著增加 Chroma CPU/内存与提示长度，易拖垮单机
        if pv == "deepseek":
            cap = int(getattr(settings, "review_deepseek_chroma_fetch_cap", 96) or 96)
            cap = max(24, min(cap, 200))
            fetch_k = min(cap, max(kb_size, 40) if kb_size > 0 else 40)
            tcap = int(getattr(settings, "review_deepseek_target_cap", 64) or 64)
            tcap = max(12, min(tcap, 120))
            target_n = min(target_n, tcap)
        else:
            fetch_k = min(500, max(kb_size, 80)) if kb_size > 0 else 80
        results = self.kb.search(document_text[:2000], top_k=fetch_k)
        if registration_type:
            reg_list = [registration_type] if isinstance(registration_type, str) else list(registration_type or [])
            reg_list = [str(r).strip() for r in reg_list if (r or "").strip()]
            if reg_list:
                filtered = []
                for doc in results:
                    app = (doc.metadata.get("applicable_registration_types") or "all").strip()
                    if app in ("all", ""):
                        filtered.append(doc)
                    else:
                        allowed = [x.strip() for x in app.split(",") if x.strip()]
                        if not allowed or any((r or "").strip() in allowed for r in reg_list):
                            filtered.append(doc)
                results = filtered
        # 按 point_id 去重，避免同一审核点重复出现
        seen_ids = set()
        deduped = []
        for doc in results:
            pid = (doc.metadata.get("point_id") or "").strip() or (doc.metadata.get("source_file") or "")
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            deduped.append(doc)
        # 按审核类别多样化，保证覆盖全面；再取前 target_n 条（最有价值条数与知识库规模、注册类别严格程度相关）
        by_category: dict = {}
        for doc in deduped:
            cat = (doc.metadata.get("category") or "未分类").strip() or "未分类"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(doc)
        ordered_cats = list(by_category.keys())
        diversified = []
        idx_per_cat = {c: 0 for c in ordered_cats}
        n_want = min(target_n, len(deduped))
        while len(diversified) < n_want:
            added = 0
            for cat in ordered_cats:
                if idx_per_cat[cat] < len(by_category[cat]):
                    diversified.append(by_category[cat][idx_per_cat[cat]])
                    idx_per_cat[cat] += 1
                    added += 1
                    if len(diversified) >= n_want:
                        break
            if added == 0:
                break
        if not diversified:
            diversified = deduped[:n_want]
        results = diversified[:n_want]
        if not results:
            main_block = "（暂无相关参考知识，请根据通用法规标准进行审核）"
        else:
            context_parts = []
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source_file", "未知来源")
                context_parts.append(f"【审核点清单·参考{i}】来源：{source}\n{doc.page_content}")
            main_block = "\n\n".join(context_parts)

        # 用户误报/纠正专用库（独立存储，检索结果单独成段，避免与清单条款混淆）
        fb_block = ""
        if self.feedback_kb is not None:
            try:
                if pv == "deepseek":
                    fb_n = min(8, max(4, target_n // 3 + 2))
                else:
                    fb_n = min(18, max(6, target_n // 2 + 4))
                fb_results = self.feedback_kb.search(document_text[:2000], top_k=fb_n)
            except Exception:
                fb_results = []
            if fb_results:
                fb_parts = [
                    "【用户误报与纠正专用库（非第二步审核点清单原文）】",
                    "以下条目来自人工标记的误报或纠正并已单独入库，**与上文「审核点清单·参考」区分**；"
                    "误报类条目不得再输出等价审核点；纠正类请优先遵循纠正后的结论。",
                ]
                for i, doc in enumerate(fb_results, 1):
                    meta = doc.metadata or {}
                    fk = (meta.get("feedback_kind") or meta.get("correction_category") or "unknown").strip()
                    src = meta.get("source_file", "未知")
                    fb_parts.append(f"【反馈{i}】类型：{fk} | 存储名：{src}\n{doc.page_content}")
                fb_block = "\n\n".join(fb_parts)

        if fb_block:
            return main_block + "\n\n" + fb_block
        return main_block

    def _retrieve_context_by_country_keywords(
        self,
        registration_countries,
        top_k_per_keyword: int = 5,
        for_provider: Optional[str] = None,
        max_docs: int = 15,
    ) -> str:
        """按页面选中的注册国家，用「国家→额外关键词」从知识库扩展检索该国家相关法规，扩大审核面。关键词匹配不区分大小写。"""
        if not registration_countries or not self.kb:
            return ""
        countries = (
            registration_countries
            if isinstance(registration_countries, (list, tuple))
            else [registration_countries]
        )
        dims = get_dimension_options()
        country_extra_keywords = dims.get("country_extra_keywords") or {}
        if not isinstance(country_extra_keywords, dict):
            return ""
        key_lower_to_keywords = {
            str(k).strip().lower(): (v if isinstance(v, list) else [v])
            for k, v in country_extra_keywords.items()
            if k
        }
        extra_terms = []
        for c in countries:
            c = (c or "").strip()
            if not c:
                continue
            kws = key_lower_to_keywords.get(c.lower())
            if kws:
                extra_terms.extend(kw for kw in kws if (kw or "").strip())
        if not extra_terms:
            return ""
        pv = (for_provider or getattr(settings, "provider", "") or "").strip().lower()
        if pv == "deepseek":
            top_k_per_keyword = min(top_k_per_keyword, 3)
            max_docs = min(max_docs, 8)
        seen = set()
        all_docs = []
        for term in extra_terms:
            try:
                results = self.kb.search(term, top_k=top_k_per_keyword)
                for doc in results:
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            except Exception:
                continue
        if not all_docs:
            return ""
        parts = [
            "【按注册国家扩展检索的法规/审核点参考】",
            "以下为根据所选注册国家配置的额外关键词（如 CE→MDR）从知识库检索到的相关内容，用于扩大审核面。",
        ]
        for i, doc in enumerate(all_docs[:max_docs], 1):
            source = doc.metadata.get("source_file", "未知来源")
            parts.append(f"【扩展参考{i}】来源：{source}\n{doc.page_content}")
        return "\n\n" + "\n\n".join(parts)

    def _retrieve_glossary_for_translation(self, document_text: str, top_k: int = 8) -> str:
        """检索词条（glossary）内容，供英文文档翻译正确性审核参考。"""
        if not self.kb:
            return ""
        try:
            results = self.kb.search_by_category(document_text[:2000], "glossary", top_k=top_k)
        except Exception:
            return ""
        if not results:
            return ""
        parts = [
            "【翻译正确性审核参考·词条】",
            "以下为知识库中的词条内容，供英文文档的术语、语法与表述审核参考。法规及英文案例见上方审核点参考知识。",
        ]
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "未知来源")
            parts.append(f"【词条参考{i}】来源：{source}\n{doc.page_content}")
        return "\n\n" + "\n\n".join(parts)

    def _country_audit_scope_hint(self, registration_countries) -> str:
        """根据页面选中的注册国家，生成「扩大审核面」提示：要求结合该国常见法规审核，即知识库未收录也需补充性审核。"""
        if not registration_countries:
            return ""
        countries = (
            registration_countries
            if isinstance(registration_countries, (list, tuple))
            else [registration_countries]
        )
        countries = [str(c).strip() for c in countries if (c or "").strip()]
        if not countries:
            return ""
        dims = get_dimension_options()
        country_extra_keywords = dims.get("country_extra_keywords") or {}
        if not isinstance(country_extra_keywords, dict):
            country_extra_keywords = {}
        key_lower_to_keywords = {
            str(k).strip().lower(): (v if isinstance(v, list) else [v])
            for k, v in country_extra_keywords.items()
            if k
        }
        # 所选国家对应的法规/标准关键词（供模型参照，即使未在知识库中命中）
        keywords_per_country = []
        for c in countries:
            kws = key_lower_to_keywords.get(c.lower())
            if kws:
                kws = [kw for kw in kws if (kw or "").strip()]
                if kws:
                    keywords_per_country.append(f"{c}：{'、'.join(kws)}")
        lines = [
            "【按注册国家扩大审核面】",
            f"本次注册国家以页面选择为准：{'、'.join(countries)}。",
            "除上述知识库检索到的内容外，请结合该国家/地区通常适用的法规与标准进行补充性审核；"
            "即使用户知识库中未收录某法规全文，也应根据该国家一般性要求对文档的合规性、完整性等提出审核意见，扩大审核面。",
        ]
        if keywords_per_country:
            lines.append("以下为各国家/地区常涉及的法规或标准关键词（即未在知识库中检索到，也请结合常识参照）：" + "；".join(keywords_per_country) + "。")
        return "\n\n" + "\n".join(lines)

    def _retrieve_audit_corrections_context(
        self, current_file_name: str = "", limit: int = 160
    ) -> str:
        """从数据库读取本集合下的误报/弃用/修订记录，注入审核参考知识，避免重复输出已人工否定的问题。"""
        coll = (self.collection_name or "").strip() or "regulations"
        try:
            rows = get_corrections_for_collection(coll, limit=limit)
        except Exception:
            return ""
        if not rows:
            return ""

        cur = (current_file_name or "").strip().lower()

        def _sort_key(r):
            fn = (str(r.get("file_name") or "")).strip().lower()
            if cur and fn == cur:
                return (0, -int(r.get("id") or 0))
            if cur and cur in fn:
                return (1, -int(r.get("id") or 0))
            return (2, -int(r.get("id") or 0))

        rows = sorted(rows, key=_sort_key)

        def _clip(s: str, n: int = 240) -> str:
            s = (s or "").replace("\n", " ").replace("\r", " ").strip()
            return s if len(s) <= n else s[: n - 1] + "…"

        lines_fp: List[str] = []
        lines_other: List[str] = []
        max_chars = 11000
        used = 0

        for row in rows:
            orig = row.get("original") if isinstance(row.get("original"), dict) else {}
            corr = row.get("corrected") if isinstance(row.get("corrected"), dict) else {}
            fn = row.get("file_name") or "（未知文件）"
            is_fp = corr.get("correction_kind") == "false_positive" or bool(
                (corr.get("false_positive_reason") or "").strip()
            )
            if is_fp:
                reason = (corr.get("false_positive_reason") or "").strip() or "（未填写原因）"
                seg = (
                    f"- **[误报]** 文件「{fn}」| 原类别：{orig.get('category', '')} | "
                    f"原位置：{_clip(str(orig.get('location', '')), 180)} | "
                    f"原描述摘要：{_clip(str(orig.get('description', '')), 300)} | "
                    f"**人工结论**：{_clip(reason, 320)} → **不得再输出与此条语义等价的审核点**。"
                )
                lines_fp.append(seg)
            elif corr.get("deprecated"):
                seg = (
                    f"- **[弃用]** 文件「{fn}」| 原描述：{_clip(str(orig.get('description', '')), 220)} | "
                    f"弃用说明：{_clip(str(corr.get('deprecation_note', '')), 160)} → 勿再以同一依据重复报同类有效问题。"
                )
                lines_other.append(seg)
            else:
                seg = (
                    f"- **[修订]** 文件「{fn}」| 原摘要：{_clip(str(orig.get('description', '')), 200)} | "
                    f"已人工修订；除非待审内容仍不符合修订后要求，勿重复报「原错误表述」类问题。"
                )
                lines_other.append(seg)
            used += len(seg)
            if used >= max_chars:
                break

        if not lines_fp and not lines_other:
            return ""

        parts: List[str] = [
            "【历史误报与纠正记录（本知识库，须严格遵守）】",
            "以下条目来自审核报告中的「纠正 / 误报 / 弃用」并已写入数据库，**与上方审核点参考知识一并参与本次判断**。",
            "**误报**：若待审文档与下列「原审核点」在类别、法规点、位置与描述含义上高度相似，**禁止**再输出等价审核点。",
            "**弃用 / 修订**：已采纳人工结论；勿重复输出已被否定或已按纠正更新的同类问题。",
        ]
        if lines_fp:
            parts.append("### 误报记录")
            parts.extend(lines_fp)
        if lines_other:
            parts.append("### 弃用与修订记录")
            parts.extend(lines_other)

        block = "\n".join(parts)
        if len(block) > max_chars + 800:
            block = block[:max_chars] + "\n\n（记录过长已截断；继续标记误报可累积本库。）"
        return "\n\n" + block

    def _retrieve_traceability_kb_context(
        self,
        doc_list: List[Tuple[str, str]],
        review_context: Optional[dict] = None,
    ) -> str:
        """从审核点知识库检索可追溯性制度、清单、程序相关片段，供跨文档追溯专项审核使用。"""
        if not self.kb:
            return "（未配置审核点知识库）"
        pv = (
            (review_context or {}).get("current_provider")
            or getattr(settings, "provider", "")
            or ""
        ).strip().lower()
        seed_parts = []
        for name, text in doc_list[:10]:
            seed_parts.append(f"{name}\n{(text or '')[:800]}")
        seed = "\n\n".join(seed_parts)[:5000]
        queries = [
            seed[:2200],
            "软件可追溯性 管理程序 标识与可追溯性 追溯矩阵 需求 测试",
            "REQ SRS 软件需求 风险 ID CS 危害 单元测试 系统测试 追溯",
            "软件需求规范 追溯表 CS 风险控制措施 矩阵",
            "风险分析 CS 编号 控制措施 危害 软件风险管理",
            "软件可追溯性分析 矩阵 需求 设计 实现 测试 风险 RTM",
        ]
        seen = set()
        chunks: List[str] = []
        max_chunks = 32 if pv != "deepseek" else 14
        per_q = 10 if pv != "deepseek" else 5
        for q in queries:
            try:
                res = self.kb.search(q, top_k=per_q)
            except Exception:
                continue
            for doc in res or []:
                key = (doc.metadata.get("source_file"), (doc.page_content or "")[:100])
                if key in seen:
                    continue
                seen.add(key)
                src = doc.metadata.get("source_file", "未知来源")
                chunks.append(f"【来源：{src}】\n{doc.page_content}")
                if len(chunks) >= max_chunks:
                    break
            if len(chunks) >= max_chunks:
                break
        if not chunks:
            return (
                "（知识库未检索到与「可追溯性/标识与可追溯性/追溯矩阵」强相关的条目；"
                "请仍按提示词中的医疗器械软件追溯通用规则审核，并建议将贵司《软件可追溯性管理程序》等制度入库以便对照条款。）"
            )
        body = "\n\n---\n\n".join(chunks)
        cap = 13000 if pv != "deepseek" else 7000
        if len(body) > cap:
            body = body[:cap] + "\n\n（知识库参考过长已截断）"
        return body

    def _parse_audit_points(self, llm_response: str) -> List[AuditPoint]:
        text = llm_response.strip()

        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        if json_start == -1 or json_end == 0:
            return [AuditPoint(
                category="解析错误",
                severity="info",
                location="N/A",
                description=f"LLM 响应无法解析为结构化数据：{text[:200]}",
                regulation_ref="N/A",
                suggestion="请重新审核",
            )]

        try:
            data = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return [AuditPoint(
                category="解析错误",
                severity="info",
                location="N/A",
                description=f"JSON 解析失败：{text[json_start:json_start+200]}",
                regulation_ref="N/A",
                suggestion="请重新审核",
            )]

        points = []
        for item in data:
            modify_docs = item.get("modify_docs")
            if not isinstance(modify_docs, list):
                modify_docs = []
            points.append(AuditPoint(
                category=item.get("category", "未分类"),
                severity=item.get("severity", "info"),
                location=item.get("location", "未知"),
                description=item.get("description", ""),
                regulation_ref=item.get("regulation_ref", ""),
                suggestion=item.get("suggestion", ""),
                modify_docs=modify_docs,
            ))
        return points

    def _normalize_actions(self, points: List[AuditPoint]) -> List[AuditPoint]:
        for p in points:
            inferred = _infer_action_from_suggestion(p.suggestion)
            if inferred:
                p.action = inferred
                if inferred == "无需修改" and (p.severity or "").lower() == "high":
                    # 建议无需修改时不应是高风险，避免界面出现“建议不改但风险高”的冲突
                    p.severity = "info"
        return points

    def _deduplicate_audit_points(self, points: List[AuditPoint]) -> List[AuditPoint]:
        if not points:
            return points
        deduped: List[AuditPoint] = []
        seen_strong = set()
        seen_soft = set()
        for p in points:
            desc = _normalize_text_for_dedup(p.description, 220)
            sug = _normalize_text_for_dedup(p.suggestion, 140)
            loc = _normalize_text_for_dedup(p.location, 80)
            cat = _normalize_text_for_dedup(p.category, 40)
            strong_key = (cat, loc, desc)
            if strong_key in seen_strong:
                continue
            soft_key = (cat, desc[:80], sug[:70])
            if soft_key in seen_soft:
                continue
            seen_strong.add(strong_key)
            seen_soft.add(soft_key)
            deduped.append(p)
        return deduped

    def _normalize_language_consistency(
        self,
        points: List[AuditPoint],
        text: str,
        review_context: Optional[dict] = None,
    ) -> List[AuditPoint]:
        if not points:
            return points
        doc_lang = ((review_context or {}).get("document_language") or "").strip().lower()
        is_effective_english = _likely_english_document(text, declared_lang=doc_lang)
        if not is_effective_english:
            return points
        # 英文文档下，剔除“文档是中文/非英文”的结论，避免同一报告内语言结论自相矛盾
        zh_claim_tokens = ("文档为中文", "文档是中文", "中文文档", "存在中文", "非英文文档", "语言为中文")
        kept: List[AuditPoint] = []
        for p in points:
            text_blob = " ".join([p.category or "", p.description or "", p.suggestion or ""]).lower()
            if any(t in text_blob for t in zh_claim_tokens):
                continue
            kept.append(p)
        return kept

    def _rule_based_obvious_checks(self, text: str, file_name: str) -> List[AuditPoint]:
        """
        程序级显性检查：补足模型容易漏掉但人眼一眼能看出的错误。
        当前仅做低风险、通用、误报率低的检查：文件编号是否在正文出现。
        """
        pts: List[AuditPoint] = []
        fn = (file_name or "").strip()
        blob = (text or "")
        # 从文件名中提取疑似“受控文件编号”
        m = re.search(r"([A-Z]{2,6}[A-Z0-9]{0,4}-[A-Z]{2,6}-\d{2,4})", fn, re.I)
        cand = (m.group(1) if m else "").upper()
        if cand:
            # 容错：允许空格/下划线/连字符差异
            norm = re.sub(r"[\s_]+", "-", cand)
            variants = {cand, norm, cand.replace("-", ""), norm.replace("-", "")}
            found = any(v in re.sub(r"[\s_]+", "", blob.upper()) for v in {cand.replace("-", ""), norm.replace("-", "")})
            if not found:
                pts.append(
                    AuditPoint(
                        category="一致性",
                        severity="medium",
                        location=f"{fn}（文件编号）",
                        description=f"文件名中包含受控文件编号「{cand}」，但在已提取的文档正文/表格中未检索到该编号；可能导致受控编号与正文不一致或页眉页脚编号丢失。",
                        regulation_ref="受控文件识别/编号一致性（通用质量体系要求）",
                        suggestion=f"请在文档首页/页眉页脚/封面处确认并统一显示文件编号为「{cand}」；若确有该编号但因版式/页眉页脚提取未覆盖，请提供带页眉页脚的可复制文本版或导出 PDF 再审。",
                        modify_docs=[fn] if fn else [],
                    )
                )
        return pts

    @staticmethod
    def _backfill_empty_modify_docs(
        points: List[AuditPoint],
        *,
        primary_display_name: str,
        multi_doc_mode: bool = False,
    ) -> List[AuditPoint]:
        """单文档审核：modify_docs 为空时自动补为当前文档显示名，便于报告与界面「需修改文档」一致。多文档专项不补。"""
        fn = (primary_display_name or "").strip()
        if multi_doc_mode or not fn or not points:
            return points
        out: List[AuditPoint] = []
        for p in points:
            md = [str(x).strip() for x in (p.modify_docs or []) if x is not None and str(x).strip()]
            if md:
                out.append(p)
            else:
                out.append(replace(p, modify_docs=[fn]))
        return out

    def _rewrite_temp_modify_docs(
        self,
        points: List[AuditPoint],
        *,
        storage_basename: str,
        display_basename: str,
        multi_doc_mode: bool,
    ) -> List[AuditPoint]:
        """将 modify_docs 中的临时上传名（tmp*.docx）替换为用户展示名，避免界面出现系统临时文件名。"""
        if multi_doc_mode or not display_basename:
            return points
        sb = (storage_basename or "").strip()
        if not sb:
            return points
        sb_key = Path(sb.replace("\\", "/")).name.casefold()
        dd = display_basename.strip()
        if not dd or sb_key == Path(dd.replace("\\", "/")).name.casefold():
            return points
        out: List[AuditPoint] = []
        for p in points:
            md_in = [str(x).strip() for x in (p.modify_docs or []) if x is not None and str(x).strip()]
            new_md: List[str] = []
            seen = set()
            for ms in md_in:
                bn = Path(ms.replace("\\", "/")).name
                if bn.casefold() == sb_key or is_probable_temp_upload_basename(bn):
                    val = dd
                else:
                    val = ms
                if val not in seen:
                    seen.add(val)
                    new_md.append(val)
            out.append(replace(p, modify_docs=new_md))
        return out

    def _post_process_audit_points(
        self,
        points: List[AuditPoint],
        text: str,
        review_context: Optional[dict] = None,
        *,
        primary_display_name: str = "",
        logical_display_name: str = "",
        storage_basename: str = "",
        multi_doc_mode: bool = False,
    ) -> List[AuditPoint]:
        points = self._normalize_actions(points)
        points = self._deduplicate_audit_points(points)
        points = self._normalize_language_consistency(points, text=text, review_context=review_context)
        # 若正文/索引中已出现某 CS 编号，避免模型误报“缺失该编号”
        scanned_cs = _extract_ids_by_prefix(text, "CS")
        if scanned_cs:
            filtered: List[AuditPoint] = []
            for p in points:
                blob = " ".join([(p.location or ""), (p.description or ""), (p.suggestion or "")]).upper()
                wrong_missing = False
                for cs in scanned_cs:
                    if cs in blob and ("未检索" in blob or "找不到" in blob or "缺失" in blob or "不存在" in blob):
                        wrong_missing = True
                        break
                if wrong_missing:
                    filtered.append(
                        AuditPoint(
                            category="准确性",
                            severity="info",
                            location=p.location or "跨文档可追溯性审核",
                            description="模型存在编号漏读风险：程序全文扫描已检出该 CS 编号，请以原始表格单元格复核后再判定缺失。",
                            regulation_ref=p.regulation_ref or "",
                            suggestion="请按对应工作表「风险需求 / Risk demand (Measure) ID」列逐行复核该编号，确认后再输出缺失结论。",
                            modify_docs=p.modify_docs or [],
                        )
                    )
                else:
                    filtered.append(p)
            points = filtered
        points = self._deduplicate_audit_points(points)
        backfill_name = (logical_display_name or primary_display_name or "").strip()
        points = self._rewrite_temp_modify_docs(
            points,
            storage_basename=storage_basename,
            display_basename=backfill_name,
            multi_doc_mode=multi_doc_mode,
        )
        points = self._backfill_empty_modify_docs(
            points,
            primary_display_name=backfill_name,
            multi_doc_mode=multi_doc_mode,
        )
        return points

    def review_text(
        self,
        text: str,
        file_name: str = "未命名文档",
        review_context: Optional[dict] = None,
        project_context_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        *,
        storage_basename: str = "",
        logical_display_name: str = "",
    ) -> AuditReport:
        pv = (
            (review_context or {}).get("current_provider")
            or getattr(settings, "provider", "")
            or ""
        ).strip().lower()
        # 展示用文件名：单文件为上传名；分块审核时 file_name 为「xxx (第i/n段)」，纠正/知识库检索用 eff_display
        eff_display = (logical_display_name or file_name).strip() or file_name
        # 按项目审核时用项目类别过滤审核点；通用审核不过滤，匹配所有审核点
        reg_type_for_filter = None
        reg_countries_for_retrieval = None
        if review_context:
            if review_context.get("_filter_by_registration_type"):
                reg_type_for_filter = review_context.get("registration_type")
            reg_countries_for_retrieval = review_context.get("registration_country")
        context = self._retrieve_context(
            text,
            registration_type=reg_type_for_filter,
            registration_countries=reg_countries_for_retrieval,
            for_provider=pv,
        )
        _corr_ctx = self._retrieve_audit_corrections_context(eff_display)
        if _corr_ctx:
            context += _corr_ctx
        # 按页面选中的注册国家，用「国家→额外关键词」扩展检索知识库中该国家相关法规，扩大审核面
        if review_context:
            reg_countries = review_context.get("registration_country")
            if reg_countries:
                extra = self._retrieve_context_by_country_keywords(
                    reg_countries, for_provider=pv
                )
                if extra:
                    context += extra
                # 要求结合该国常见法规补充审核，即知识库中未收录的法规也纳入审核面
                scope_hint = self._country_audit_scope_hint(reg_countries)
                if scope_hint:
                    context += "\n\n" + scope_hint
                # P-2.2 注册国家严格程度（美国>中国>欧盟）提示词加强
                _country_ctx = _country_strictness_context(reg_countries)
                if _country_ctx:
                    context += _country_ctx
            parts = []
            if review_context.get("project_name") or review_context.get("project_name_en"):
                v = review_context.get("project_name") or ""
                v_en = review_context.get("project_name_en") or ""
                parts.append(f"项目名称：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("product_name") or review_context.get("product_name_en"):
                v = review_context.get("product_name") or ""
                v_en = review_context.get("product_name_en") or ""
                parts.append(f"产品名称：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("model") or review_context.get("model_en"):
                v = review_context.get("model") or ""
                v_en = review_context.get("model_en") or ""
                parts.append(f"型号（Model）：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("registration_country") or review_context.get("registration_country_en"):
                v = review_context.get("registration_country") or ""
                v_en = review_context.get("registration_country_en") or ""
                parts.append(f"注册国家：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("registration_type"):
                v = review_context["registration_type"]
                parts.append(f"注册类别：{v if isinstance(v, str) else '、'.join(v)}")
            if review_context.get("registration_component"):
                v = review_context["registration_component"]
                parts.append(f"注册组成：{v if isinstance(v, str) else '、'.join(v)}")
            if review_context.get("project_form"):
                v = review_context["project_form"]
                parts.append(f"项目形态：{v if isinstance(v, str) else '、'.join(v)}")
            if parts:
                context += "\n\n【本次审核维度】\n" + "；".join(parts) + "。请根据上述维度识别适用的法规、程序与项目案例要求。"
            # P-4.1 项目形态侧重点（Web/APP/PC 默认）
            _form_ctx = _project_form_focus_context(review_context.get("project_form"))
            if _form_ctx:
                context += _form_ctx
            # 待审文档语言：仅当有项目案例上下文时生效；法规/程序审核本身为所有语言通用
            doc_lang = review_context.get("document_language") or ""
            has_case = bool(review_context.get("case_context_text"))
            if doc_lang and has_case:
                if doc_lang == "zh":
                    context += "\n\n【待审文档语言】本次待审文档为**中文版**，请按中文注册文档规范与一致性要求进行审核；术语、格式、表述须符合中文注册文档习惯。"
                elif doc_lang == "en":
                    context += "\n\n【待审文档语言】本次待审文档为**英文版**，请按英文注册文档规范与一致性要求进行审核；术语、格式、表述须符合英文注册文档习惯。须增加「翻译正确性」审核维度（国家/语言语法习惯、是否通顺、是否符合逻辑），并参考下方词条及上方法规、英文案例。"
                    context += self._retrieve_glossary_for_translation(text)
                elif doc_lang == "both":
                    context += "\n\n【待审文档语言】本次待审文档可能为**中文版或英文版**（中英文混合或分批），请按中英文注册文档规范与一致性要求进行审核；术语、格式、表述须兼顾中英文，关键信息中英文一致时须同时核对。"
            elif doc_lang == "en":
                # 无项目案例时仍对英文文档做翻译正确性审核，参考词条与法规
                context += "\n\n【待审文档语言】本次待审文档为**英文版**，请按英文注册文档规范审核，并增加「翻译正确性」维度（国家/语言语法习惯、通顺性、逻辑性），参考下方词条及上方法规与英文案例。"
                context += self._retrieve_glossary_for_translation(text)
            if review_context.get("project_name") or review_context.get("project_name_en") or review_context.get("product_name") or review_context.get("product_name_en") or review_context.get("model") or review_context.get("model_en"):
                context += " 待审文档中出现的项目名称、产品名称、型号（含中英文，**字段名称**如「型号」「Model」「产品名称」等识别时不区分大小写；**取值须区分大小写、精确匹配（含空格），不能有不一致**，否则须作为审核点（一致性）列出。"
        if review_context and review_context.get("basic_info_text"):
            context += "\n\n【项目基本信息（已入库，须与待审文档一致）】\n" + (review_context.get("basic_info_text") or "")
            context += "\n\n待审文档中的项目名称、产品名称、型号规格、注册单元名称等须与上述基本信息一致；型号/Model 字段名称中英文不区分大小写，**取值须区分大小写、精确匹配（含空格），不能有不一致**；若不一致须作为审核点（一致性）列出。"
        if review_context and review_context.get("scope_of_application"):
            context += "\n\n【产品适用范围】\n" + (review_context.get("scope_of_application") or "")
            context += "\n\n**范围约束**：所有文档描述的内容（功能、适应症、适用人群、使用场景等）不能超过上述适用范围。若文档中出现超出适用范围的功能描述、适应症、适用人群或使用场景，须作为审核点（类别可为「合规性」或「一致性」）明确列出，并给出修改建议。"
        # 注册类别审核尺度：严格程度 Ⅲ > Ⅱb > Ⅱa > Ⅱ > Ι
        _reg_type = review_context.get("registration_type") if review_context else None
        if _reg_type:
            _rt = _reg_type if isinstance(_reg_type, str) else ("、".join(_reg_type) if _reg_type else "")
            if _rt.strip():
                context += _registration_strictness_context(_rt.strip())
        if review_context and review_context.get("system_functionality_text"):
            context += "\n\n【系统功能描述（来自安装包或 URL 识别，须与待审文档一致）】\n" + (review_context.get("system_functionality_text") or "")
            context += "\n\n请核对待审文档中的功能描述、界面说明、操作流程、模块列表等是否与上述系统功能一致；若不一致须作为审核点（一致性）明确列出，并给出修改建议。"
        if review_context and review_context.get("reference_docs_excerpt"):
            context += (
                "\n\n【参考文件（本次上传，须作为对照依据）】\n"
                + (review_context.get("reference_docs_excerpt") or "")
                + "\n\n请将以上参考文件中的要求/约束/规则作为本次审核的对照依据：逐项核对待审文档是否满足；"
                "不满足处须输出审核点并给出可操作修改建议。若参考文件与审核点知识库/项目维度冲突，"
                "须在审核点中明确指出冲突位置，并建议以受控程序/项目维度为准处理。"
            )
        if project_context_text:
            # P-4.2 项目专属与基本信息、待审文档三者对齐
            _p4_2 = ""
            if review_context and review_context.get("basic_info_text"):
                _p4_2 = "以下内容须与【项目基本信息】及待审文档**三者对齐**；若仅存在项目专属与待审文档，则二者须完全一致。\n\n"
            context += (
                "\n\n【项目专属要求（技术要求、说明书等，以下为项目资料中须与待审文档保持一致的内容）】\n"
                + _p4_2
                + project_context_text
            )
            context += "\n\n**一致性要求**：待审文档中的**项目名称、产品名称、型号规格、注册单元名称**等须与上述项目专属资料中出现的对应信息保持一致；型号/Model 字段名称中英文不区分大小写，**取值须区分大小写、精确匹配（含空格），不能有不一致**；若不一致须作为审核点（类别为「一致性」）明确列出，并给出修改建议。"
        if review_context and review_context.get("case_context_text"):
            context += review_context["case_context_text"]
        if review_context and review_context.get("extra_instructions"):
            context += "\n\n【自定义审核要求（请严格遵守）】\n" + (review_context.get("extra_instructions") or "")

        sys_prompt = system_prompt if (system_prompt and system_prompt.strip()) else REVIEW_SYSTEM_PROMPT
        usr_prompt = user_prompt if (user_prompt and user_prompt.strip()) else REVIEW_USER_PROMPT

        if settings.is_cursor:
            from .cursor_agent import complete_task
            template = CURSOR_REVIEW_TASK if not (user_prompt and user_prompt.strip()) else usr_prompt
            prompt_text = template.format(
                context=context,
                file_name=file_name,
                document_content=text,
            )
            response_content = complete_task(prompt_text)
            audit_points = self._parse_audit_points(response_content)
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prompt),
                ("human", usr_prompt),
            ])
            chain = prompt | self.llm
            wait_before_llm_call(pv)
            response = chain.invoke({
                "context": context,
                "file_name": file_name,
                "document_content": text,
            })
            audit_points = self._parse_audit_points(response.content)

        audit_points = self._post_process_audit_points(
            audit_points,
            text=text,
            review_context=review_context,
            primary_display_name=file_name,
            logical_display_name=logical_display_name,
            storage_basename=storage_basename,
            multi_doc_mode=False,
        )
        # 程序级显性检查（补漏）
        try:
            audit_points = (self._rule_based_obvious_checks(text, eff_display) or []) + (audit_points or [])
        except Exception:
            pass
        audit_points = self._deduplicate_audit_points(audit_points)

        report = AuditReport(file_name=eff_display)
        report.audit_points = audit_points
        report.total_points = len(audit_points)
        report.high_count = sum(1 for p in audit_points if p.severity == "high")
        report.medium_count = sum(1 for p in audit_points if p.severity == "medium")
        report.low_count = sum(1 for p in audit_points if p.severity == "low")
        report.info_count = sum(1 for p in audit_points if p.severity == "info")

        report.summary = self._generate_summary(report, review_context)
        return report

    def review_file(
        self,
        file_path,
        review_context: Optional[dict] = None,
        project_context_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        display_file_name: Optional[str] = None,
    ) -> AuditReport:
        path = Path(file_path)
        storage_bn = path.name
        fn = (display_file_name or "").strip() or storage_bn
        force_ocr_refresh = bool((review_context or {}).get("_force_ocr_refresh"))
        docs = load_single_file(
            path,
            force_ocr_refresh=force_ocr_refresh,
            ocr_cache_file_name=fn or storage_bn,
        )
        full_text = "\n\n".join(doc.page_content for doc in docs)

        if len(full_text) > 30000:
            return self._review_long_document(
                full_text,
                fn,
                review_context,
                project_context_text,
                system_prompt,
                user_prompt,
                storage_basename=storage_bn,
            )

        return self.review_text(
            full_text,
            fn,
            review_context,
            project_context_text,
            system_prompt,
            user_prompt,
            storage_basename=storage_bn,
            logical_display_name=fn,
        )

    def _review_long_document(
        self,
        text: str,
        file_name: str,
        review_context: Optional[dict] = None,
        project_context_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        *,
        storage_basename: str = "",
    ) -> AuditReport:
        chunk_size = 25000
        overlap = 2000
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            # 到达文末必须退出；否则 end-overlap 可能停在同一位置导致死循环
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

        # 长文档分块审核：与侧栏/会话 current_provider 对齐（避免仅用 settings 与 UI 不一致）
        ctx_prov = (review_context or {}).get("current_provider") if review_context else None
        provider = (ctx_prov or getattr(settings, "provider", "") or "").strip().lower()
        max_workers = 3
        if provider in ("deepseek",):
            max_workers = 1  # DeepSeek 串行分块，避免网关/限流抖动（批量多文件本身已是逐份串行）
        elif provider in ("openai", "lingyi"):
            max_workers = 2
        if getattr(settings, "is_cursor", False):
            max_workers = 2  # Cursor Agent 并发过高易超时，保守为 2
        all_points: List[AuditPoint] = []

        def _review_chunk(args):
            i, chunk = args
            chunk_name = f"{file_name} (第{i+1}/{len(chunks)}段)"
            # DeepSeek 轻微节流，降低连续请求过快导致的网关抖动
            if provider == "deepseek" and i > 0:
                time.sleep(0.55)
            report = self.review_text(
                chunk,
                chunk_name,
                review_context,
                project_context_text,
                system_prompt,
                user_prompt,
                storage_basename=storage_basename,
                logical_display_name=file_name,
            )
            return i, report.audit_points

        if len(chunks) <= 1:
            _, pts = _review_chunk((0, chunks[0]))
            all_points.extend(pts)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_review_chunk, (i, ch)): i
                    for i, ch in enumerate(chunks)
                }
                # 按块序号排序合并，便于报告顺序与原文大致一致
                indexed: List[Tuple[int, List[AuditPoint]]] = []
                for fut in as_completed(futures):
                    try:
                        indexed.append(fut.result())
                    except Exception as e:
                        indexed.append(
                            (
                                futures[fut],
                                [
                                    AuditPoint(
                                        category="系统错误",
                                        severity="high",
                                        location=f"{file_name} 第{futures[fut]+1}段",
                                        description=f"分块审核失败：{e}",
                                        regulation_ref="N/A",
                                        suggestion="请重试或缩短单段长度",
                                    )
                                ],
                            )
                        )
                indexed.sort(key=lambda x: x[0])
                for _, pts in indexed:
                    all_points.extend(pts)

        all_points = self._post_process_audit_points(
            all_points,
            text=text,
            review_context=review_context,
            primary_display_name=file_name,
            logical_display_name=file_name,
            storage_basename=storage_basename,
            multi_doc_mode=False,
        )

        final_report = AuditReport(file_name=file_name)
        final_report.audit_points = all_points
        final_report.total_points = len(all_points)
        final_report.high_count = sum(1 for p in all_points if p.severity == "high")
        final_report.medium_count = sum(1 for p in all_points if p.severity == "medium")
        final_report.low_count = sum(1 for p in all_points if p.severity == "low")
        final_report.info_count = sum(1 for p in all_points if p.severity == "info")
        final_report.summary = self._generate_summary(final_report, review_context)
        return final_report

    def review_multiple_files(
        self,
        file_paths: List[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> List[AuditReport]:
        reports = []
        for fp in file_paths:
            try:
                report = self.review_file(fp, system_prompt=system_prompt, user_prompt=user_prompt)
                reports.append(report)
            except Exception as e:
                error_report = AuditReport(file_name=str(fp))
                error_report.audit_points = [AuditPoint(
                    category="系统错误",
                    severity="high",
                    location=str(fp),
                    description=f"文件审核失败：{str(e)}",
                    regulation_ref="N/A",
                    suggestion="请检查文件格式是否正确",
                )]
                error_report.total_points = 1
                error_report.high_count = 1
                error_report.summary = f"文件审核失败：{str(e)}"
                reports.append(error_report)
        return reports

    def review_multi_document_consistency(
        self,
        doc_list: List[Tuple[str, str]],
        review_context: Optional[dict] = None,
        max_chars_per_doc: int = 4000,
    ) -> AuditReport:
        """
        多文档一致性与模板风格审核。doc_list = [(display_name, text), ...]。
        返回一份虚拟报告「多文档一致性与模板风格审核」，审核点均为跨文档问题。
        """
        if not doc_list or len(doc_list) < 2:
            report = AuditReport(file_name="多文档一致性与模板风格审核")
            report.summary = "不足两份文档，跳过多文档一致性审核。"
            return report

        parts = []
        for name, text in doc_list:
            t = (text or "").strip()[:max_chars_per_doc]
            parts.append(f"### {name}\n{t}")
        docs_summary = "\n\n".join(parts)

        extra = ""
        if review_context:
            if review_context.get("project_name") or review_context.get("product_name"):
                extra = "\n\n【约定】项目名称、产品名称等须在各文档中与上述一致；若某文档与另一文档或与约定不一致，须作为一致性审核点列出。"
            if review_context.get("basic_info_text"):
                extra += "\n\n【项目基本信息】\n" + (review_context.get("basic_info_text") or "")
            # 待审文档语言：仅当有项目案例上下文时生效；法规/程序审核为所有语言通用
            doc_lang = review_context.get("document_language") or ""
            has_case = bool(review_context.get("case_context_text"))
            if doc_lang and has_case:
                if doc_lang == "zh":
                    extra += "\n\n【待审文档语言】本批文档为**中文版**，请按中文注册文档规范检查各文档间一致性及术语、格式统一。"
                elif doc_lang == "en":
                    extra += "\n\n【待审文档语言】本批文档为**英文版**，请按英文注册文档规范检查各文档间一致性及术语、格式统一。"
                elif doc_lang == "both":
                    extra += "\n\n【待审文档语言】本批文档可能含**中文版与英文版**，请按中英文注册文档规范检查各文档间一致性，并兼顾中英文术语、格式与表述统一。"

        prompt_text = MULTI_DOC_CONSISTENCY_PROMPT.format(docs_summary=docs_summary) + extra
        _mcorr = self._retrieve_audit_corrections_context("")
        if _mcorr:
            prompt_text += _mcorr

        try:
            if settings.is_cursor:
                from .cursor_agent import complete_task
                # 多文档一致性审核内容多，总超时与单次读超时需更长，避免 ReadTimeout
                response_content = complete_task(
                    prompt_text,
                    poll_interval=2.0,
                    timeout=600,
                )
                response_content = (response_content or "").strip()
            else:
                # 仅用直接 HTTP 调用，绝不走 LangChain invoke，否则 content 会被当模板解析导致 {"category"} 报错
                from .llm_factory import invoke_chat_direct
                _provider_override = (review_context or {}).get("current_provider") if review_context else None
                try:
                    response_content = invoke_chat_direct(
                        prompt_text or "",
                        provider=_provider_override,
                    ).strip()
                except RuntimeError as re:
                    if "invoke_chat_direct 暂不支持" in str(re):
                        raise RuntimeError(
                            "多文档一致性与模板风格审核当前仅支持 Cursor、OpenAI、DeepSeek、零一万物、Ollama。"
                            "请先在侧栏将 AI 服务切换为上述之一后再试。"
                        ) from re
                    raise
            audit_points = self._parse_audit_points(response_content or "[]")
        except Exception as e:
            raise RuntimeError(f"多文档一致性审核接口调用失败：{e}") from e

        audit_points = self._post_process_audit_points(
            audit_points,
            text=docs_summary,
            review_context=review_context,
            primary_display_name="",
            multi_doc_mode=True,
        )

        report = AuditReport(file_name="多文档一致性与模板风格审核")
        report.audit_points = audit_points
        report.total_points = len(audit_points)
        report.high_count = sum(1 for p in audit_points if p.severity == "high")
        report.medium_count = sum(1 for p in audit_points if p.severity == "medium")
        report.low_count = sum(1 for p in audit_points if p.severity == "low")
        report.info_count = sum(1 for p in audit_points if p.severity == "info")
        report.summary = (
            self._generate_summary(report, review_context)
            if audit_points
            else "各文档间信息与风格一致性已检查；未发现不一致项。"
        )
        return report

    def review_traceability_cross_document(
        self,
        doc_list: List[Tuple[str, str]],
        review_context: Optional[dict] = None,
        max_chars_per_doc: int = 11000,
        max_total_chars: int = 120000,
    ) -> AuditReport:
        """
        跨文档可追溯性专项审核。doc_list = [(display_name, text), ...]。
        结合知识库检索的可追溯性制度/清单，核对需求—设计—测试—风险等追溯编号与内容一致性。
        """
        if not doc_list or len(doc_list) < 2:
            report = AuditReport(file_name="跨文档可追溯性审核")
            report.summary = "不足两份文档，跳过跨文档可追溯性审核。"
            return report

        pv = (
            (review_context or {}).get("current_provider")
            or getattr(settings, "provider", "")
            or ""
        ).strip().lower()
        if pv == "deepseek":
            max_chars_per_doc = min(max_chars_per_doc, 6800)
            max_total_chars = min(max_total_chars, 88000)

        names = [n for n, _ in doc_list]
        n_docs = len(doc_list)
        # 追溯表/矩阵常在中后部，单份摘录过短易漏 CS/URS；提高每份下限，并用首部+尾部摘录
        per = min(max_chars_per_doc, max(3600, max_total_chars // max(n_docs, 1)))
        parts = []
        for name, text in doc_list:
            t = _excerpt_traceability_document_with_registry((text or "").strip(), per)
            parts.append(f"### {name}\n{t}")
        docs_summary = "\n\n".join(parts)
        if len(docs_summary) > max_total_chars:
            docs_summary = (
                docs_summary[:max_total_chars]
                + "\n\n（总摘录长度已达上限并截断；请基于已提供内容分析，勿要求用户补传已在清单中的文档。）"
            )

        traceability_kb = self._retrieve_traceability_kb_context(doc_list, review_context)
        prompt_text = TRACEABILITY_CROSS_DOC_PROMPT.format(
            doc_count=len(doc_list),
            doc_names="、".join(names),
            docs_summary=docs_summary,
            traceability_kb=traceability_kb,
        )

        extra = ""
        if review_context:
            if review_context.get("project_name") or review_context.get("product_name"):
                extra += "\n\n【约定】项目名称、产品名称等须与本批各文档及下列基本信息一致；追溯编号指向的内容须与这些约定对齐。"
            if review_context.get("basic_info_text"):
                extra += "\n\n【项目基本信息】\n" + (review_context.get("basic_info_text") or "")
            doc_lang = review_context.get("document_language") or ""
            has_case = bool(review_context.get("case_context_text"))
            if doc_lang and has_case:
                if doc_lang == "zh":
                    extra += "\n\n【待审文档语言】本批为**中文**为主，追溯编号与表格栏目以中文文档习惯解析。"
                elif doc_lang == "en":
                    extra += "\n\n【待审文档语言】本批为**英文**为主，追溯编号与表格栏目以英文文档习惯解析。"
                elif doc_lang == "both":
                    extra += "\n\n【待审文档语言】本批可能**中英混排**，同一 ID 在中英文表述中须指向一致对象。"

        _gex = ((review_context or {}).get("extra_instructions") or "").strip()
        if not _gex:
            try:
                _gex = (get_review_extra_instructions() or "").strip()
            except Exception:
                _gex = ""
        if _gex:
            extra += "\n\n【自定义审核要求】\n" + _gex

        prompt_text = prompt_text + extra
        _mcorr = self._retrieve_audit_corrections_context("")
        if _mcorr:
            prompt_text += _mcorr

        try:
            if settings.is_cursor:
                from .cursor_agent import complete_task
                response_content = complete_task(
                    prompt_text,
                    poll_interval=2.0,
                    timeout=600,
                )
                response_content = (response_content or "").strip()
            else:
                from .llm_factory import invoke_chat_direct
                _provider_override = (review_context or {}).get("current_provider") if review_context else None
                try:
                    response_content = invoke_chat_direct(
                        prompt_text or "",
                        provider=_provider_override,
                    ).strip()
                except RuntimeError as re:
                    if "invoke_chat_direct 暂不支持" in str(re):
                        raise RuntimeError(
                            "跨文档可追溯性审核当前仅支持 Cursor、OpenAI、DeepSeek、零一万物、Ollama。"
                            "请先在侧栏将 AI 服务切换为上述之一后再试。"
                        ) from re
                    raise
            audit_points = self._parse_audit_points(response_content or "[]")
        except Exception as e:
            raise RuntimeError(f"跨文档可追溯性审核接口调用失败：{e}") from e

        audit_points = self._post_process_audit_points(
            audit_points,
            text=docs_summary,
            review_context=review_context,
            primary_display_name="",
            multi_doc_mode=True,
        )

        report = AuditReport(file_name="跨文档可追溯性审核")
        report.audit_points = audit_points
        report.total_points = len(audit_points)
        report.high_count = sum(1 for p in audit_points if p.severity == "high")
        report.medium_count = sum(1 for p in audit_points if p.severity == "medium")
        report.low_count = sum(1 for p in audit_points if p.severity == "low")
        report.info_count = sum(1 for p in audit_points if p.severity == "info")
        report.summary = (
            self._generate_summary(report, review_context)
            if audit_points
            else "跨文档可追溯性已检查；在已提供摘录范围内未发现明显断链或编号矛盾。"
        )
        return report

    @staticmethod
    def _heuristic_summary(report: AuditReport) -> str:
        if not report.audit_points:
            return "未发现审核问题。"
        return (
            f"共 {report.total_points} 条审核点（高 {report.high_count} / 中 {report.medium_count} / "
            f"低 {report.low_count} / 提示 {report.info_count}）。已启用轻量摘要（跳过二次模型调用以降低负载）；详见下方列表。"
        )

    def _generate_summary(
        self, report: AuditReport, review_context: Optional[dict] = None
    ) -> str:
        if not report.audit_points:
            return "未发现审核问题。"

        if review_context is not None and "_skip_llm_summary" in review_context:
            skip = bool(review_context.get("_skip_llm_summary"))
        else:
            skip = False
            if getattr(settings, "review_batch_skip_llm_summary", True):
                pv0 = (
                    (review_context or {}).get("current_provider")
                    or getattr(settings, "provider", "")
                    or ""
                ).strip().lower()
                if (
                    pv0 == "deepseek"
                    and review_context
                    and review_context.get("_batch_review")
                ):
                    skip = True
        if skip:
            return self._heuristic_summary(report)

        details = "\n".join(
            f"- [{p.severity}] {p.category}: {p.description}"
            for p in report.audit_points[:10]
        )
        inv = {
            "file_name": report.file_name,
            "high": report.high_count,
            "medium": report.medium_count,
            "low": report.low_count,
            "info": report.info_count,
            "details": details,
        }
        from .db import get_prompt_by_key
        summary_tpl = (get_prompt_by_key("review_summary_prompt") or "").strip() or None
        if not summary_tpl:
            summary_tpl = CURSOR_SUMMARY_TASK if settings.is_cursor else SUMMARY_PROMPT

        if settings.is_cursor:
            from .cursor_agent import complete_task
            prompt_text = summary_tpl.format(**inv)
            return complete_task(prompt_text).strip()

        pv = (
            (review_context or {}).get("current_provider")
            or getattr(settings, "provider", "")
            or ""
        ).strip().lower()
        wait_before_llm_call(pv)
        prompt = ChatPromptTemplate.from_messages([("human", summary_tpl)])
        chain = prompt | self.llm
        response = chain.invoke(inv)
        return response.content
