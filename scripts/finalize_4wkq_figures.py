"""A lighting/transparency-only refinement of the coordinate-based PyMOL scene."""
from pathlib import Path
import json
import pymol
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/ppt_4wkq'
pymol.finish_launching(['pymol','-cq'])
from pymol import cmd
cmd.load(str(OUT/'4wkq_figures.pse'))
cmd.set('max_threads',8)
cmd.set('antialias',2)
cmd.set('ray_opaque_background',0)
cmd.scene('pocket','recall',animate=0)
cmd.set('transparency',.65,'pocket_surface')
cmd.hide('sticks','egfr')
cmd.zoom('site or gefitinib',.35)
cmd.scene('pocket_light','store')
cmd.png(str(OUT/'4wkq_pocket_light.png'),width=2800,height=2000,ray=1,quiet=1,dpi=300)
im=Image.open(OUT/'4wkq_pocket_light.png').convert('RGBA')
b=im.getchannel('A').getbbox()
if b:
 l,t,r,bt=b
 im=im.crop((max(0,l-30),max(0,t-30),min(im.width,r+30),min(im.height,bt+30)))
im.save(OUT/'4wkq_pocket_light.png',dpi=(300,300))
cmd.save(str(OUT/'4wkq_figures_final.pse'))
p=OUT/'structure_render_manifest.json'
d=json.loads(p.read_text());d['pocket_light_surface_transparency']=.65;d['pocket_light_view']=list(cmd.get_view());d['publication_assets'].append('4wkq_pocket_light.png');p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
cmd.quit()
