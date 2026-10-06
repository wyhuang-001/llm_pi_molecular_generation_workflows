"""Normalize Mermaid SVG labels and assemble a coordinate-faithful figure plate."""
from pathlib import Path
import base64
import re
from lxml import etree as E
ROOT=Path(__file__).resolve().parents[1]
AS=ROOT/'assets/ppt_4wkq'
M=AS/'mermaid'
OUT=ROOT/'deliverables/4WKQ_figure_sources'
OUT.mkdir(parents=True,exist_ok=True)
NS='http://www.w3.org/2000/svg'

def tag(n):return '{'+NS+'}'+n
for name in ['overview','key_loop','next_steps','workflow_publication']:
    path=M/(name+'.svg')
    root=E.parse(str(path)).getroot()
    foreign=root.findall('.//'+tag('foreignObject'))
    if foreign:
        (M/(name+'.raw.svg')).write_bytes(path.read_bytes())
    for fo in foreign:
        width,height=float(fo.get('width')),float(fo.get('height'))
        value=''.join(fo.itertext()).strip()
        spans=list(fo.iter())
        color='#244C68'
        for el in spans:
            style=el.get('style','')
            match=re.search(r'color:\s*(#[A-Fa-f0-9]{6})',style)
            if match:color=match.group(1)
        text=E.Element(tag('text'),x=str(width/2),y=str(height/2),
                       attrib={'text-anchor':'middle','dominant-baseline':'central',
                               'style':f'font-family:Microsoft YaHei,WenQuanYi Zen Hei,sans-serif;font-size:24px;fill:{color} !important;'})
        text.text=value
        fo.getparent().replace(fo,text)
    # Mermaid defaults edge-label backgrounds to 50% opacity. Make them solid
    # white so feedback lines never run visibly through label glyphs.
    for group in root.findall('.//'+tag('g')):
        if group.get('class') == 'edgeLabel':
            for bg in group.findall('.//'+tag('rect')):
                bg.set('style','fill:#FFFFFF !important;opacity:1 !important;')
    root.attrib.pop('style',None)
    root.set('height', root.get('viewBox').split()[3])
    path.write_bytes(E.tostring(root,encoding='utf-8',xml_declaration=True))
    assert not root.findall('.//'+tag('foreignObject'))

# English multi-panel figure; molecular panels are unaltered PyMOL renderings.
svg=E.Element(tag('svg'),nsmap={None:NS},width='1600',height='1050',viewBox='0 0 1600 1050')
E.SubElement(svg,tag('rect'),x='0',y='0',width='1600',height='1050',fill='white')

def text(x,y,value,size=23,color='#244C68',bold=False):
    t=E.SubElement(svg,tag('text'),x=str(x),y=str(y),fill=color,
                   attrib={'font-family':'Arial, sans-serif','font-size':str(size),'font-weight':'700' if bold else '400'})
    t.text=value

def image(name,x,y,w,h):
    data=base64.b64encode((AS/name).read_bytes()).decode()
    E.SubElement(svg,tag('image'),x=str(x),y=str(y),width=str(w),height=str(h),
                 href='data:image/png;base64,'+data,preserveAspectRatio='xMidYMid meet')

text(55,62,'Structure-guided, scaffold-preserving molecular optimization',32,bold=True)
text(57,99,'Experimental structure: EGFR–gefitinib, PDB 4WKQ, 1.85 Å',20,'#647D8E')
text(57,151,'a',26,bold=True);text(87,151,'Structural context',23,bold=True)
text(560,151,'b',26,bold=True);text(592,151,'Binding-site environment',23,bold=True)
text(1122,151,'c',26,bold=True);text(1154,151,'Editable region',23,bold=True)
image('4wkq_overview.png',65,178,425,420)
image('4wkq_pocket_light.png',525,175,560,423)
image('4wkq_ligand.png',1130,180,360,410)
text(76,637,'EGFR kinase domain',20,'#647D8E')
text(560,637,'Crystallographic binding pose',20,'#647D8E')
E.SubElement(svg,tag('circle'),cx='1144',cy='620',r='6',fill='#149CAA')
text(1162,627,'Retained scaffold',19,'#647D8E')
E.SubElement(svg,tag('circle'),cx='1144',cy='652',r='6',fill='#E6A441')
text(1162,659,'Editable side chain',19,'#647D8E')
text(56,733,'d',26,bold=True);text(89,733,'Closed-loop optimization',23,bold=True)
mer=E.parse(str(M/'workflow_publication.svg')).getroot()
mer.set('x','55');mer.set('y','750');mer.set('width','1490');mer.set('height','235')
svg.append(mer)
text(58,1022,'Molecular panels rendered from deposited coordinates; workflow is conceptual. No generative structural redraw.',17,'#647D8E')
(OUT/'Figure_4WKQ_workflow.svg').write_bytes(E.tostring(svg,encoding='utf-8',xml_declaration=True))
print(OUT/'Figure_4WKQ_workflow.svg')
