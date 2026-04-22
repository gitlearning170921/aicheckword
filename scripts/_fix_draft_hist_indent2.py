"""Fix broken indentation in _render_draft_history after partial dedent."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "src" / "app.py"
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
start = None
end = None
for i, ln in enumerate(lines):
    if ln.strip() == "idx = int(_pick_g)" and i > 0 and lines[i - 1].strip().startswith("rec = _grp"):
        start = i + 1
        break
if start is None:
    raise SystemExit("start not found")
for j in range(start, len(lines)):
    if lines[j].startswith("    def _render_post_audit_mode"):
        end = j
        break
if end is None:
    raise SystemExit("end not found")

def fix_line(ln: str) -> str:
    if not ln.strip():
        return ln
    stripped = ln.lstrip(" ")
    sp = len(ln) - len(stripped)
    if sp >= 12:
        return " " * (sp - 4) + stripped
    return ln

for k in range(start, end):
    lines[k] = fix_line(lines[k])

p.write_text("".join(lines), encoding="utf-8")
print("fixed", start + 1, "to", end)
