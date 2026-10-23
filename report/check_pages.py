"""Convert the report to PDF with LibreOffice, find the page of every section heading,
write report/toc_pages.json and render page previews for checking."""
import json, os, subprocess
import fitz

DOCX = "report/AAT_Report_Tensor_Ops_and_Drowsiness_Detection.docx"
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", "report", DOCX], check=True, capture_output=True)
doc = fitz.open(DOCX.replace(".docx", ".pdf"))
print("pages:", len(doc))

TOC = {"aat1": ["1. Introduction", "2. Objectives", "3. Case Description", "4. Analysis and Discussion", "5. Results", "6. Conclusion"],
       "aat2": ["1. Introduction", "2. Objectives", "3. Methodology", "4. Results (With Screenshots)", "5. Software Used", "6. Conclusion"]}
BODY_START = 2                                   # cover and contents pages come first
aat2_start = next(i for i in range(BODY_START, len(doc)) if "AAT - 2:" in doc[i].get_text())


def lines(i):
    return [l.strip() for l in doc[i].get_text().split("\n") if l.strip()]


def find(heading, lo, hi):
    return next(i for i in range(lo, hi) if any(l.startswith(heading) for l in lines(i)))


def at_top(i, heading):
    # header line, then the heading (or a title line before it on the first page of a project)
    body = [l for l in lines(i) if not l.startswith("AAT - ")]
    return any(l.startswith(heading) for l in body[:3])


toc = {}
for key, lo, hi in [("aat1", BODY_START, aat2_start), ("aat2", aat2_start, len(doc))]:
    idx = [find(h, lo, hi) for h in TOC[key]]
    for k in range(6):
        start = idx[k]
        if k == 5:
            end = hi - 1
        else:
            end = idx[k + 1] - 1 if at_top(idx[k + 1], TOC[key][k + 1]) else idx[k + 1]
        end = max(start, end)
        a, b = start - BODY_START + 1, end - BODY_START + 1
        toc[f"{key}:{k + 1}"] = str(a) if a == b else f"{a} - {b}"
print(toc)
json.dump(toc, open("report/toc_pages.json", "w"), indent=1)

os.makedirs("report/preview", exist_ok=True)
for i, page in enumerate(doc):
    page.get_pixmap(dpi=70).save(f"report/preview/p{i + 1:02d}.png")
