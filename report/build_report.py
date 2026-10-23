"""Build the AAT report (.docx). Page numbers in the table of contents come from report/toc_pages.json,
which is filled in by check_pages.py after a first conversion to PDF."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from docx_helpers import *
import content_aat1
ONLY1 = os.environ.get("ONLY_AAT1") == "1"

A = "report/assets/"
AAT1_TITLE = "Perform Basic Tensor Operations using TensorFlow"
AAT2_TITLE = "Driver Drowsiness Detection using CNN"
TOC1 = ["Introduction", "Objectives", "Case Description / Background Information", "Analysis and Discussion", "Results", "Conclusion"]
TOC2 = ["Introduction", "Objectives", "Methodology", "Results (With Screenshots)", "Software Used", "Conclusion"]
pages = json.load(open("report/toc_pages.json")) if os.path.exists("report/toc_pages.json") else {}

doc = new_document()
cover = doc.sections[0]
page_border(cover)

# ---------- cover ----------
t = doc.add_table(rows=1, cols=3)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
for col, w in zip(t.columns, (0.95, 4.7, 0.95)):
    col.width = Inches(w)
    for c in col.cells:
        c.width = Inches(w)
t.cell(0, 0).paragraphs[0].add_run().add_picture(A + "logo_0.png", width=Inches(0.85))
t.cell(0, 2).paragraphs[0].add_run().add_picture(A + "logo_1.png", width=Inches(0.75))
t.cell(0, 2).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
mid = t.cell(0, 1)
mid.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
p = mid.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
set_font(p.add_run("DAYANANDA SAGAR COLLEGE OF ENGINEERING"), size=15)
for txt, sz in [("(An Autonomous Institute affiliated to Visvesvaraya Technological University (VTU), Belagavi, "
                 "Approved by AICTE and UGC, Accredited by NAAC with ‘A’ grade)", 7.5),
                ("Shavige Malleshwara Hills, Kumaraswamy Layout, Bengaluru-560 111, India", 9)]:
    q = mid.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.space_after = Pt(0)
    set_font(q.add_run(txt), size=sz)

def c(text, size, before=0, after=0):
    return para(doc, text, size=size, align="center", space_before=before, space_after=after)

c("DEPARTMENT OF INFORMATION SCIENCE & ENGINEERING", 14, before=26)
c("(Accredited by NBA Tier 1: 2025-2028)", 10, after=22)
c("Introduction to Deep Learning (22IS741)", 12)
c("AAT report", 12)
c("on", 10, after=18)
c(f"AAT1: {AAT1_TITLE}", 15)
c(f"AAT2: {AAT2_TITLE}", 15, after=20)
c("Bachelor of Engineering in", 14)
c("Information Science and Engineering", 14, after=22)
c("Submitted by", 11.5, after=6)
students = [("ARPIT SRIVASTAVA", "1DS23IS023"), ("KHUSHI AGRAWAL", "1DS23IS074"),
            ("HIMANSHU", "1DS23IS061"), ("HARSH KUMAR GUPTA", "1DS23IS054")]
for name, usn in students:
    q = para(doc, "", align="left", space_after=0)
    q.paragraph_format.left_indent = Inches(1.6)
    q.paragraph_format.tab_stops.add_tab_stop(Inches(4.1))
    set_font(q.add_run(f"{name}\t({usn})"), size=10.5)
c("Under the Guidance of", 12, before=26, after=6)
c("Dr. Rajeshwari J - Associate Professor", 10.5)
c("Prof Ayush Mishra - Assistant Professor", 10.5)
c("Dept of ISE", 10.5, after=26)
c("VISVESVARAYA TECHNOLOGICAL UNIVERSITY", 15)
c("JNANASANGAMA, BELAGAVI-590018, KARNATAKA, INDIA", 15)
c("2026-27", 15)
page_break(doc)

# ---------- table of contents ----------
c("TABLE OF CONTENTS", 20, before=24, after=14)
table(doc, [["Sl. no.", "Topics", "Page no."]], widths=[0.8, 4.6, 0.9])
for sec_title, items, key in [(AAT1_TITLE, TOC1, "aat1"), (AAT2_TITLE, TOC2, "aat2")]:
    c(sec_title, 13, before=8, after=6)
    rows = [[str(i), it, pages.get(f"{key}:{i}", "-")] for i, it in enumerate(items, 1)]
    table(doc, rows, widths=[0.8, 4.6, 0.9], header=False)

# ---------- AAT 1 ----------
s1 = new_section(doc)
setup_body_section(s1, f"AAT - 1: {AAT1_TITLE}", restart=True)
content_aat1.build(doc, 1)

# ---------- AAT 2 ----------
body_sections = [s1]
if not ONLY1:
    import content_aat2
    s2 = new_section(doc)
    setup_body_section(s2, f"AAT - 2: {AAT2_TITLE}")
    body_sections.append(s2)
    content_aat2.build(doc, 1)
for k, s in enumerate(doc.sections):             # border only on the cover and contents; numbering runs on
    if k > 0:
        b = s._sectPr.find(qn("w:pgBorders"))
        if b is not None:
            s._sectPr.remove(b)
    if k > 1:
        for pg in s._sectPr.findall(qn("w:pgNumType")):
            s._sectPr.remove(pg)

# ---------- checks: no em or en dashes anywhere ----------
text = "\n".join(p.text for p in doc.paragraphs)
for tbl in doc.tables:
    for row in tbl.rows:
        for cell in row.cells:
            text += "\n" + cell.text
bad = [ch for ch in ("—", "–") if ch in text]
assert not bad, f"dash characters found: {bad}"

out = "report/AAT_Report_Tensor_Ops_and_Drowsiness_Detection.docx"
doc.save(out)
print("saved", out)
