from pathlib import Path

p = Path(__file__).resolve().parents[1] / "src" / "app.py"
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
lines = [ln for ln in lines if "for _once in (0,):" not in ln]
start = None
end = None
for i, ln in enumerate(lines):
    if "with st.expander(title, expanded=(idx < 1)):" in ln:
        start = i
        break
if start is None:
    raise SystemExit("expander not found")
for j in range(start + 1, len(lines)):
    if lines[j].startswith("    def _render_post_audit_mode"):
        end = j
        break
if end is None:
    raise SystemExit("end not found")
head = lines[:start]
new_open = [
    '        st.markdown("---")\n',
    '        st.markdown(f"##### {title}")\n',
]
mid = []
for k in range(start + 1, end):
    ln = lines[k]
    if ln.startswith("                "):
        mid.append(ln[4:])
    elif ln.startswith("            ") and ln.strip():
        mid.append(ln[4:])
    else:
        mid.append(ln)
out = head + new_open + mid + lines[end:]
p.write_text("".join(out), encoding="utf-8")
print("ok lines", start + 1, "to", end)
