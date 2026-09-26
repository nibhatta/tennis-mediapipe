#!/usr/bin/env python3
"""Build a single self-contained demo HTML: inline viewer.js + sample JSON into index.html."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "index.html")) as f:
    html = f.read()
with open(os.path.join(HERE, "viewer.js")) as f:
    js = f.read()
with open(os.path.join(HERE, "sample", "sample_landmarks.json")) as f:
    sample = json.load(f)
with open(os.path.join(HERE, "sample", "sample_landmarks.csv")) as f:
    sample_csv = f.read()

# Swap the fetch-based boot for embedded data.
old_boot = """// ---------------------------------------------------------------- boot
fetch('./sample/sample_landmarks.json')
  .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then(j => loadData(j, 'sample_landmarks.json'))
  .catch(err => showErr(
    'Could not load the bundled demo data.\\n' +
    'Serve this folder over HTTP (e.g. `python3 -m http.server`) or use "Load JSON".\\n\\n' + err.message));

requestAnimationFrame(tick);"""
new_boot = """// ---------------------------------------------------------------- boot
try {
  loadData(window.__POSE_DATA__, 'sample_landmarks.json (embedded demo)');
} catch (err) { showErr('Could not load embedded demo data.\\n\\n' + err.message); }

requestAnimationFrame(tick);"""
assert old_boot in js, "boot block not found in viewer.js"
js = js.replace(old_boot, new_boot)

# Inline the module script.
tag = '<script type="module" src="./viewer.js"></script>'
assert tag in html, "script tag not found in index.html"
inline = (
    "<script>window.__POSE_DATA__ = "
    + json.dumps(sample, separators=(",", ":"))
    + ";</script>\n<script>window.__SAMPLE_CSV__ = "
    + json.dumps(sample_csv)
    + ";</script>\n<script type=\"module\">\n"
    + js
    + "\n</script>"
)
html = html.replace(tag, inline)

out_dir = os.path.join(HERE, "dist")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "pose-lab-demo.html")
with open(out, "w") as f:
    f.write(html)
print(f"Wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
