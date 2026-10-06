from pathlib import Path
import pymol
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/ppt_4wkq'
pymol.finish_launching(['pymol','-cq'])
from pymol import cmd
cmd.load(str(OUT/'4wkq_figures.pse'))
cmd.set('max_threads',8)
cmd.set('antialias',1)
cmd.set('ray_opaque_background',1)
angles=[(0,0),(0,180),(50,0),(-50,0),(0,60),(0,-60)]
paths=[]
for i,(x,y) in enumerate(angles):
 cmd.scene('pocket','recall',animate=0)
 cmd.turn('x',x);cmd.turn('y',y)
 cmd.set('transparency',.60,'pocket_surface')
 cmd.set('transparency_mode',2)
 p=OUT/f'view_test_{i}.png';cmd.png(str(p),width=760,height=570,ray=1,quiet=1)
 paths.append(p)
sheet=Image.new('RGB',(1520,3*610),'white')
for i,p in enumerate(paths):
 sheet.paste(Image.open(p).convert('RGB'),((i%2)*760,(i//2)*610))
 ImageDraw.Draw(sheet).text(((i%2)*760+20,(i//2)*610+575),str((i,angles[i])),fill='black')
sheet.save(OUT/'camera_tests.jpg')
cmd.quit()
