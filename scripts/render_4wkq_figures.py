"""Publication-oriented deterministic PyMOL figures from deposited 4WKQ coordinates.
Run: QT_QPA_PLATFORM=offscreen /tmp/4wkq-pymol/bin/python scripts/render_4wkq_figures.py
No geometry optimization, generative redraw, or ligand alignment is performed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pymol
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'ppt_4wkq'
OUT.mkdir(parents=True, exist_ok=True)
PDB = ROOT / '4WKQ' / 'raw' / '4WKQ.original.pdb'
pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd

cmd.load(str(PDB), 'deposited')
cmd.remove('not chain A or (not alt ""+A)')
cmd.create('egfr', 'deposited and (polymer.protein or resn CSX)')
cmd.create('gefitinib', 'deposited and resn IRE and resi 1101')
cmd.delete('deposited')
cmd.remove('hydro')
assert cmd.count_atoms('gefitinib') == 31
# Side-chain atom names correspond to reference SDF indices 0..9, verified by coordinates.
raw_atoms = []
for row in PDB.read_text().splitlines():
    if row.startswith('HETATM') and row[17:20] == 'IRE' and row[21] == 'A' and row[22:26].strip() == '1101':
        raw_atoms.append({'name': row[12:16].strip(), 'xyz': [float(row[30:38]), float(row[38:46]), float(row[46:54])]})
sdf = (ROOT/'runs/4wkq-llm-20260927-193317/reference-ligand.sdf').read_text().splitlines()
n_atoms = int(sdf[3][:3])
ref_xyz = np.array([[float(r[0:10]),float(r[10:20]),float(r[20:30])] for r in sdf[4:4+n_atoms]])
names = []
for xyz in ref_xyz[:31]:
    distances = np.linalg.norm(np.array([a['xyz'] for a in raw_atoms])-xyz, axis=1)
    ix = int(distances.argmin())
    assert distances[ix] < .001
    names.append(raw_atoms[ix]['name'])
cmd.select('editable_chain', 'gefitinib and name ' + '+'.join(names[:10]))
cmd.select('fixed_core', 'gefitinib and not editable_chain')
assert cmd.count_atoms('fixed_core') == 21

cmd.set_color('protein_blue', [0.57,0.76,0.89])
cmd.set_color('protein_light', [0.72,0.85,0.93])
cmd.set_color('core_teal', [0.05,0.63,0.70])
cmd.set_color('edit_orange', [0.96,0.61,0.20])
cmd.set_color('nitrogen_blue', [.21,.40,.79])
cmd.set_color('oxygen_red', [.88,.29,.26])
cmd.set_color('halogen_green', [.32,.72,.52])
cmd.set_color('residue_gray', [.51,.60,.66])
cmd.bg_color('white')
cmd.set('orthoscopic', 1)
cmd.set('antialias', 2)
cmd.set('max_threads', 8)
cmd.set('ray_trace_mode', 0)
cmd.set('ray_shadows', 0)
cmd.set('ray_trace_fog', 0)
cmd.set('depth_cue', 0)
cmd.set('ambient', .58)
cmd.set('direct', .42)
cmd.set('specular', .24)
cmd.set('shininess', 35)
cmd.set('reflect', .15)
cmd.set('light_count', 3)
cmd.set('ray_opaque_background', 0)
cmd.set('opaque_background', 0)
cmd.set('surface_quality', 1)
cmd.set('cartoon_sampling', 14)
cmd.set('cartoon_smooth_loops', 1)
cmd.set('cartoon_fancy_helices', 1)
cmd.set('cartoon_flat_sheets', 1)
cmd.set('cartoon_oval_length', 1.12)
cmd.set('cartoon_oval_width', .22)
cmd.set('stick_quality', 24)
cmd.set('sphere_quality', 3)
cmd.set('transparency_mode', 2)
cmd.color('protein_blue', 'egfr')
cmd.color('protein_light', 'egfr and resi 800-1022')
cmd.color('core_teal', 'fixed_core')
cmd.color('edit_orange', 'editable_chain')
cmd.color('nitrogen_blue', 'gefitinib and elem N')
cmd.color('oxygen_red', 'gefitinib and elem O')
cmd.color('halogen_green', 'gefitinib and elem F+Cl')

# Put the major ligand axis roughly horizontal; keep one camera for comparable panels.
cmd.orient('fixed_core')
cmd.turn('z', 15)
cmd.turn('y', 12)
cmd.turn('x', -12)
rotation_view = cmd.get_view()

def save(name, width=2400, height=1800):
    path = OUT / (name+'.png')
    if path.exists():
        print('Reuse', path, flush=True)
        return path
    print('Rendering', path, flush=True)
    cmd.png(str(path), width=width, height=height, dpi=300, ray=1, quiet=1)
    im = Image.open(path).convert('RGBA')
    bbox = im.getchannel('A').getbbox()
    if bbox:
        l,t,r,b=bbox
        p=45
        box=(max(0,l-p),max(0,t-p),min(im.width,r+p),min(im.height,b+p))
        im=im.crop(box)
    im.save(path)
    return path

# Whole biological context, not a fabricated molecular surface.
cmd.hide('everything')
cmd.show('cartoon', 'egfr')
cmd.show('sticks', 'gefitinib')
cmd.set('stick_radius', .25, 'gefitinib')
cmd.show('spheres', 'gefitinib')
cmd.set('sphere_scale', .25, 'gefitinib')
cmd.zoom('egfr or gefitinib', 2.4)
cmd.scene('overview', 'store')
overview_view = cmd.get_view()
save('4wkq_overview', 2400, 2100)

# Actual crystallographic ligand and the local binding-site environment.
cmd.hide('cartoon')
cmd.select('site', 'byres (egfr within 6.0 of gefitinib)')
cmd.create('pocket_surface', 'site')
cmd.color('protein_light', 'pocket_surface')
cmd.show('surface', 'pocket_surface')
cmd.set('transparency', .43, 'pocket_surface')
cmd.show('sticks', 'egfr and resi 793 and name N+CA+C+O')
cmd.color('residue_gray', 'egfr and resi 793')
cmd.set('stick_radius', .13, 'egfr')
cmd.set('stick_radius', .22, 'gefitinib')
cmd.set('sphere_scale', .23, 'gefitinib')
cmd.set_view(rotation_view)
cmd.zoom('gefitinib', 5.1)
cmd.scene('pocket', 'store')
pocket_view = cmd.get_view()
save('4wkq_pocket', 2700, 1900)

# Alternate surface-free local view for unambiguous ligand topology.
cmd.hide('surface')
cmd.show('cartoon', 'egfr')
cmd.set('cartoon_transparency', .40, 'egfr')
cmd.zoom('gefitinib', 4.2)
cmd.scene('binding_pose', 'store')
save('4wkq_binding_pose', 2400, 1700)

# Isolated deposited ligand: exact 3D pose, no 2D replacement or chemical invention.
cmd.hide('cartoon')
cmd.hide('sticks', 'egfr')
cmd.zoom('gefitinib', 1.3)
cmd.scene('ligand', 'store')
save('4wkq_ligand', 2400, 1500)
cmd.save(str(OUT/'4wkq_figures.pse'))

manifest = {
    'source': str(PDB.relative_to(ROOT)),
    'source_url': 'https://files.rcsb.org/download/4WKQ.pdb',
    'source_sha256': hashlib.sha256(PDB.read_bytes()).hexdigest(),
    'structure': '4WKQ, human EGFR kinase domain with gefitinib, X-ray 1.85 Angstrom',
    'chain': 'A',
    'ligand': 'IRE A 1101',
    'ligand_heavy_atoms': 31,
    'fixed_core_atoms': 21,
    'editable_sidechain_atom_names': names[:10],
    'ligand_reference_sdf_coordinate_match_max_angstrom': 0.001,
    'coordinates_modified': False,
    'generative_redrawing': False,
    'altloc_policy': 'blank/A',
    'CSX_A_797_retained': bool(cmd.count_atoms('egfr and resn CSX and resi 797')),
    'overview_view': list(overview_view),
    'pocket_view': list(pocket_view),
    'render_engine': 'PyMOL ' + cmd.get_version()[0],
    'publication_assets': ['4wkq_overview.png','4wkq_pocket.png','4wkq_binding_pose.png','4wkq_ligand.png'],
}
(OUT/'structure_render_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in manifest.items() if 'view' not in k}, ensure_ascii=False, indent=2))
cmd.quit()
