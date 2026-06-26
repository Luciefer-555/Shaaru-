import pathlib
p = pathlib.Path("run_e2e_raqs_test.py")
data = p.read_bytes()
enc = "utf-16" if data.count(b"\x00") > 100 else "utf-8"
text = data.decode(enc)
text = text.replace(
    'sys.path.insert(0, os.path.join(ROOT, "pipeline"))',
    '# pipeline path removed to avoid circular import',
)
p.write_text(text, encoding="utf-8", newline="\n")
print("nulls", p.read_bytes().count(b"\x00"))
