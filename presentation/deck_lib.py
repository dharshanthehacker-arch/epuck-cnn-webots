"""Small helper layer over python-pptx for a consistent, designed deck."""
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).parent / "assets"

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x0F, 0x2A, 0x3A)      # dominant dark
TEAL = RGBColor(0x1B, 0x7F, 0x79)      # primary accent
TEAL_LT = RGBColor(0xE3, 0xF1, 0xEF)   # tinted panels
AMBER = RGBColor(0xF2, 0xA9, 0x3B)     # sharp accent
RED = RGBColor(0xD6, 0x45, 0x3D)
INK = RGBColor(0x1E, 0x2B, 0x33)
MUTED = RGBColor(0x5B, 0x6B, 0x73)
LINE = RGBColor(0xD5, 0xDD, 0xE0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PANEL = RGBColor(0xF3, 0xF6, 0xF7)
HEAD = "Calibri"
BODY = "Calibri"

W, H = Inches(13.333), Inches(7.5)


def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs, dark=False, bg=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = NAVY if dark else WHITE
    pic = s.shapes.add_picture(str(ASSETS / (bg or ("bg_dark.png" if dark else "bg_light.png"))), 0, 0, W, H)
    pic.name = "Background"
    return s


def soft_shadow(shape, alpha=16, blur=12, dist=3):
    """Soft drop shadow (outer shadow, straight down)."""
    spPr = shape._element.spPr
    for old in spPr.findall(qn("a:effectLst")):
        spPr.remove(old)
    eff = spPr.makeelement(qn("a:effectLst"), {})
    sh = eff.makeelement(qn("a:outerShdw"), {"blurRad": str(int(Pt(blur))), "dist": str(int(Pt(dist))),
                                             "dir": "5400000", "algn": "t", "rotWithShape": "0"})
    clr = sh.makeelement(qn("a:srgbClr"), {"val": "0F2A3A"})
    clr.append(clr.makeelement(qn("a:alpha"), {"val": str(alpha * 1000)}))
    sh.append(clr)
    eff.append(sh)
    ln = spPr.find(qn("a:ln"))
    if ln is not None:
        ln.addnext(eff)
    else:
        spPr.append(eff)


def cover_image(slide, path, x, y, w, h, name=None):
    """Fill the box completely (crop the overflow), like CSS object-fit: cover."""
    iw, ih = Image.open(path).size
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    box_r, img_r = w / h, iw / ih
    if img_r > box_r:      # too wide: crop left/right
        c = (1 - box_r / img_r) / 2
        pic.crop_left = pic.crop_right = c
    else:                  # too tall: crop top/bottom
        c = (1 - img_r / box_r) / 2
        pic.crop_top = pic.crop_bottom = c
    if name:
        pic.name = name
    return pic


def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font=BODY, italic=False, margin=0.0, line_spacing=None, name=None):
    """content: str | list of paragraphs; a paragraph is str or list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = content if isinstance(content, list) else [content]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        if line_spacing:
            para.line_spacing = line_spacing
        runs = p if isinstance(p, list) else [(p, {})]
        for t, o in runs:
            r = para.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, space=8, bullet_color=TEAL):
    """items: list of str or (bold_lead, rest) tuples. Real bullets via XML."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.28)))
        pPr.set("indent", str(-Inches(0.28)))
        bc = pPr.makeelement(qn("a:buClr"), {})
        clr = bc.makeelement(qn("a:srgbClr"), {"val": str(bullet_color)})
        bc.append(clr)
        pPr.append(bc)
        bf = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        pPr.append(bf)
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
        lead, rest = (it if isinstance(it, tuple) else (None, it))
        if lead:
            r = p.add_run()
            r.text = lead
            r.font.bold, r.font.size, r.font.color.rgb, r.font.name = True, Pt(size), color, BODY
        r = p.add_run()
        r.text = rest
        r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, BODY
    return tb


def title(slide, t, tag=None, dark=False):
    if tag:
        text(slide, 0.6, 0.32, 9, 0.3, tag.upper(), size=12, bold=True,
             color=AMBER if dark else TEAL, name="Section tag")
    text(slide, 0.6, 0.58, 12.1, 0.8, t, size=30, bold=True, font=HEAD,
         color=WHITE if dark else NAVY, name="Title")


def footer(slide, n, dark=False):
    text(slide, 12.0, 7.0, 0.8, 0.3, str(n), size=10, color=MUTED if not dark else LINE,
         align=PP_ALIGN.RIGHT, name="Slide number")


def box(slide, x, y, w, h, fill=None, line=None, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None, shadow=True):
    fill = WHITE if fill is None else fill
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        sh.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    if shadow and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        soft_shadow(sh)
    else:
        sh.shadow.inherit = False
    sh.text_frame.margin_left = sh.text_frame.margin_right = Inches(0.1)
    return sh


def shape_text(sh, content, size=14, color=INK, bold=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        runs = p if isinstance(p, list) else [(p, {})]
        for t, o in runs:
            r = para.add_run()
            r.text = t
            r.font.name = BODY
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = o.get("color", color)
    return sh


def circle_num(slide, x, y, d, label, fill=TEAL, color=WHITE, size=14):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    shape_text(c, label, size=size, color=color, bold=True)
    c.text_frame.margin_left = c.text_frame.margin_right = 0
    return c


def image(slide, path, x, y, w, h, rounded=True, border=True, align="center", name=None):
    """Place image fitted (contain) inside the box; rounded-corner frame motif."""
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == "center" else (x if align == "left" else x + w - pw)
    py = y + (h - ph) / 2
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    if name:
        pic.name = name
    if rounded:
        geom = pic._element.spPr.find(qn("a:prstGeom"))
        geom.set("prst", "roundRect")
        av = geom.find(qn("a:avLst"))
        if av is None:
            av = geom.makeelement(qn("a:avLst"), {})
            geom.append(av)
        gd = av.makeelement(qn("a:gd"), {"name": "adj", "fmla": "val 4000"})
        av.append(gd)
    if border:
        pic.line.color.rgb = WHITE
        pic.line.width = Pt(2.5)
        soft_shadow(pic, alpha=22, blur=16, dist=4)
    return pic, (px, py, pw, ph)


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def bar_chart(slide, x, y, w, h, categories, series, colors, title_text=None, number_format='0.0',
              horizontal=False, legend=False, val_max=None, val_min=None, label_size=11, gap=60):
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series:
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(label_size)
    ch.font.name = BODY
    ch.font.color.rgb = MUTED
    ch.has_title = bool(title_text)
    if title_text:
        ch.chart_title.text_frame.text = title_text
        tp = ch.chart_title.text_frame.paragraphs[0]
        tp.runs[0].font.size = Pt(13)
        tp.runs[0].font.bold = True
        tp.runs[0].font.color.rgb = INK
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(11)
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = number_format
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(label_size)
    dl.font.color.rgb = INK
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.number_format = number_format
    va.tick_labels.number_format_is_linked = False
    if val_max is not None:
        va.maximum_scale = val_max
    if val_min is not None:
        va.minimum_scale = val_min
    ca = ch.category_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(label_size)
    ca.has_major_gridlines = False
    return ch


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def divider(prs, num, title_t, sub, img_path):
    """Section divider: big number, title, subtitle, framed hero image."""
    s = blank(prs, dark=True, bg="bg_divider.png")
    text(s, 0.75, 1.55, 4.0, 1.6, num, size=110, bold=True, color=AMBER, font=HEAD)
    text(s, 0.8, 3.35, 5.6, 1.6, title_t, size=40, bold=True, color=WHITE, font=HEAD, line_spacing=0.95)
    text(s, 0.8, 4.95, 5.4, 1.2, sub, size=17, color=LINE)
    image(s, img_path, 6.9, 1.2, 5.85, 5.1)
    return s
