"""Normalize the Mermaid main-flow SVG so PowerPoint renders every label.

Mermaid emits node labels as <foreignObject> (HTML), which PowerPoint's SVG
importer drops. This converts them to plain <text>/<tspan> elements, makes
edge-label backgrounds opaque, and removes the max-width style attribute.
Idempotent: re-running on an already-normalized file is a no-op.
"""
from pathlib import Path
import re
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / 'assets/ppt_mainflow/mermaid/main_flow.svg'
RAW = ROOT / 'assets/ppt_mainflow/mermaid/main_flow.raw.svg'
NS = 'http://www.w3.org/2000/svg'
FONT = 'Microsoft YaHei,WenQuanYi Zen Hei,sans-serif'


def tag(n):
    return '{' + NS + '}' + n


def local(el):
    return E.QName(el).localname


def paragraph_lines(p):
    """Split an HTML <p> into visual lines, honouring <br/> (XHTML ns)."""
    lines, cur = [], p.text or ''
    for child in p:
        if local(child) == 'br':
            lines.append(cur)
            cur = ''
        else:
            cur += child.text or ''
        if child.tail:
            cur += child.tail
    lines.append(cur)
    return [ln.strip() for ln in lines]


def label_color(fo):
    for el in fo.iter():
        m = re.search(r'color:\s*(#[A-Fa-f0-9]{6})', el.get('style', '') or '')
        if m:
            return m.group(1)
    return '#244C68'


def main():
    source = RAW if RAW.exists() else SVG
    root = E.parse(str(source)).getroot()

    for fo in root.findall('.//' + tag('foreignObject')):
        width, height = float(fo.get('width')), float(fo.get('height'))
        color = label_color(fo)
        paragraphs = [p for p in fo.iter() if local(p) == 'p'] or [fo]
        lines = []
        for p in paragraphs:
            lines.extend(paragraph_lines(p))
        lines = [ln for ln in lines if ln] or [''.join(fo.itertext()).strip()]
        font_size = 24.0
        line_height = font_size * 1.5
        y0 = height / 2 - (len(lines) - 1) * line_height / 2
        text = E.Element(tag('text'), attrib={
            'text-anchor': 'middle',
            'style': f'font-family:{FONT};font-size:{font_size}px;fill:{color} !important;',
        })
        for i, ln in enumerate(lines):
            tspan = E.SubElement(text, tag('tspan'), attrib={
                'x': str(width / 2),
                'y': str(y0 + i * line_height),
                'dominant-baseline': 'central',
            })
            tspan.text = ln
        fo.getparent().replace(fo, text)

    # Solid white edge-label backgrounds (Mermaid defaults to 50% opacity).
    for group in root.findall('.//' + tag('g')):
        if group.get('class') == 'edgeLabel':
            for bg in group.findall('.//' + tag('rect')):
                bg.set('style', 'fill:#FFFFFF !important;opacity:1 !important;')

    root.attrib.pop('style', None)
    vb = root.get('viewBox').split()
    root.set('height', vb[3])
    SVG.write_bytes(E.tostring(root, encoding='utf-8', xml_declaration=True))
    assert not root.findall('.//' + tag('foreignObject')), 'foreignObject left in SVG'
    print('Normalized', SVG, 'text nodes =', len(root.findall('.//' + tag('text'))))


if __name__ == '__main__':
    main()
