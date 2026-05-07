# 演示文稿（PPT）

## 考试训练中心功能介绍

由脚本生成，便于随代码迭代更新内容。

### 依赖

仅生成 PPT 时需要（不写入主 `requirements.txt`）：

```bash
pip install -r requirements-dev.txt
```

### 生成命令

在仓库根目录执行：

```bash
python scripts/generate_exam_center_pptx.py
```

默认输出：`docs/presentations/exam_center_overview.pptx`

指定输出路径：

```bash
python scripts/generate_exam_center_pptx.py --out D:/tmp/exam_center_overview.pptx
```

### 修改内容

编辑 [`scripts/generate_exam_center_pptx.py`](../../scripts/generate_exam_center_pptx.py) 中的 `slides` 列表（标题与要点），保存后重新运行上述命令即可。
