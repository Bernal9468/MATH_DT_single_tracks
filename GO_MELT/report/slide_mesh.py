from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
p = Presentation(); p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
s = p.slides.add_slide(p.slide_layouts[6])
def txt(x, y, w, h, t, size, bold=False, col="1F2A44"):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.margin_left = tf.margin_right = 0; r = tf.paragraphs[0].add_run(); r.text = t
    r.font.size = Pt(size); r.font.bold = bold; r.font.name = "Calibri"; r.font.color.rgb = RGBColor.from_string(col)
txt(0.6, 0.4, 12.1, 0.9, "Multi-level mesh: elements per level", 36, True)
rows = [["Level", "Domain", "Element size", "Elements", "Nodes"],
 ["L1, coarse plate", "Whole plate, 25.4 × 25.4 × 3.17 mm", "0.1 mm near the track, up to 1.0 mm; 20 µm top layer", "105,000", "113,322"],
 ["L2, meso box", "3.4 × 1.2 × 0.8 mm, follows the laser", "40 µm laterally, 20 µm vertically", "102,000", "109,306"],
 ["L3, fine box", "2.1 × 0.4 × 0.4 mm, follows the laser", "20 µm", "42,000", "46,746"],
 ["Multi-level total", "", "", "249,000", ""],
 ["Single-level baseline", "Track region", "20 µm everywhere", "306,240", "323,610"]]
cw = [2.6, 3.6, 3.7, 1.2, 1.0]
t = s.shapes.add_table(6, 5, Inches(0.6), Inches(1.7), Inches(12.1), Inches(4.2)).table
for j, w in enumerate(cw): t.columns[j].width = Inches(w)
for i, r in enumerate(rows):
    t.rows[i].height = Inches(0.7)
    for j, v in enumerate(r):
        c = t.cell(i, j); c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.fill.solid()
        c.fill.fore_color.rgb = RGBColor.from_string("2B7BBA" if i == 0 else ("E8F0F8" if i == 4 else "FFFFFF"))
        pg = c.text_frame.paragraphs[0]; pg.alignment = PP_ALIGN.CENTER if (i == 0 or j >= 3) else PP_ALIGN.LEFT
        run = pg.add_run(); run.text = v; run.font.size = Pt(18); run.font.name = "Calibri"; run.font.bold = (i in (0, 4, 5) and j in (0, 3)) or i == 0
        run.font.color.rgb = RGBColor.from_string("FFFFFF" if i == 0 else ("B35C00" if i == 5 and j in (0, 3) else "222222"))
txt(0.6, 6.3, 12.1, 0.4, "Case 0 (2.1 mm single track), final configuration", 14, False, "555555")
p.save("slide_mesh.pptx")
