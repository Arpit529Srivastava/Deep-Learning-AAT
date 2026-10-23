"""Render groups of executed notebook cells as Jupyter-style HTML pages, then screenshot them with headless Chrome."""
import html, os, subprocess
import nbformat
from PIL import Image, ImageOps
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

NB = "aat1/tensor_operations_executed.ipynb"
OUT = "report/figures"
TMP = "aat1/shots_html"
os.makedirs(TMP, exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

nb = nbformat.read(NB, as_version=4)
cells = {c["id"]: c for c in nb.cells}
fmt = HtmlFormatter(style="default", nowrap=True)
CSS = """
body{margin:0;background:#fff;font-family:-apple-system,Helvetica,Arial,sans-serif}
.wrap{padding:14px 18px 10px 6px}
.cell{display:flex;margin-bottom:10px}
.prompt{width:58px;flex:none;text-align:right;padding-right:8px;font:10.5px Menlo,monospace;color:#303f9f;padding-top:9px}
.prompt.out{color:#d84315}
.src{flex:1;background:#f7f7f7;border:1px solid #cfcfcf;border-radius:3px;padding:8px 10px;font:11.5px/1.4 Menlo,monospace;white-space:pre-wrap;word-break:break-all;min-width:0}
.output{flex:1;padding:6px 10px;font:11.5px/1.35 Menlo,monospace;white-space:pre-wrap;color:#000;min-width:0}
""" + fmt.get_style_defs(".src")


def render(name, ids, height):
    parts = []
    n = 1
    for cid in ids:
        c = cells[cid]
        count = c.get("execution_count") or n
        n += 1
        parts.append(f'<div class="cell"><div class="prompt">In [{count}]:</div>'
                     f'<div class="src">{highlight(c["source"], PythonLexer(), fmt)}</div></div>')
        text = "".join(o.get("text", "") for o in c.get("outputs", []) if o.get("name") == "stdout")
        if text:
            parts.append(f'<div class="cell"><div class="prompt out"></div><div class="output">{html.escape(text.rstrip())}</div></div>')
    page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body><div class='wrap'>{''.join(parts)}</div></body></html>"
    path = os.path.abspath(f"{TMP}/{name}.html")
    open(path, "w").write(page)
    png = os.path.abspath(f"{OUT}/{name}.png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=560,{height}", f"--screenshot={png}", f"file://{path}"],
                   check=True, capture_output=True)
    im = Image.open(png).convert("RGB")
    bbox = ImageOps.invert(im).getbbox()          # trim the empty white area
    im.crop((0, 0, im.width, min(im.height, bbox[3] + 24))).save(png)
    print(png, im.size)


render("aat1_shot_1a", ["0fb5ed75", "3d2a8aab"], 2400)
render("aat1_shot_1b", ["58af7aee", "91338569"], 2400)
render("aat1_shot_2a", ["3451ff6a", "c1f123bc", "622c7735"], 2400)
render("aat1_shot_2b", ["4773f660", "737ebd94"], 2400)
render("aat1_shot_3a", ["f8f4245b", "35eb49b1", "ae5bf74f"], 2400)
render("aat1_shot_3b", ["e20b9ecc", "b075ad95"], 2400)
