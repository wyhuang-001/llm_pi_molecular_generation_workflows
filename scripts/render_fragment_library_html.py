#!/usr/bin/env python
"""Render every fragment library to 2D structure grids and one HTML gallery.

Each fragment is already defined by an attachment SMILES with one mapped dummy
atom ``[*:1]``, so RDKit draws the real chemical structure including the
attachment point.  No image model is used: a generative image model cannot draw a
chemically correct structure, so the depictions come from the molecule graph.

Output is a directory of page images plus ``index.html`` that embeds all of them.

    python scripts/render_fragment_library_html.py --out-dir deliverables/fragment-libraries-2d
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LIBRARIES = [
    "molecular_agent/data/fragments.json",
    "molecular_agent/data/chembl_fragments.json",
    "molecular_agent/data/chembl_fragments_working.json",
    "molecular_agent/data/fragments_unified.json",
    "molecular_agent/data/fragment_library_v2.json",
    "4WKQ/design/fragments.json",
    "4WKQ/design/fragments_v2.json",
]

# Keep one page inside a size a browser can actually open.  At the default cell
# size this is roughly 3000 x 13000 px (about 160 MB of pixels) per page.
MAX_PAGE_PIXELS = 40_000_000


@dataclass
class Fragment:
    index: int
    fragment_id: str
    smiles: str
    change_types: list[str] = field(default_factory=list)


@dataclass
class Library:
    path: Path
    fragments: list[Fragment] = field(default_factory=list)
    unparsable: int = 0


def _change_types(record: dict) -> list[str]:
    """Read the editable-method metadata without rewriting the frozen libraries."""
    from molecular_agent.edit_taxonomy import CHANGE_TYPES, change_types_from_legacy_operations

    explicit = record.get("allowed_change_types")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit if item in CHANGE_TYPES]
    operations = record.get("allowed_operations")
    if not isinstance(operations, list):
        operation = record.get("operation")
        operations = [operation] if isinstance(operation, str) else []
    return change_types_from_legacy_operations([str(item) for item in operations])


def load_library(path: Path) -> Library:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        records = document
    elif isinstance(document, dict):
        records = document.get("fragments") or document.get("records") or []
    else:
        raise ValueError(f"{path}: unsupported library document type")

    library = Library(path=path)
    for record in records:
        if not isinstance(record, dict):
            continue
        smiles = record.get("smiles") or record.get("fragment_smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            continue
        fragment_id = record.get("fragment_id") or record.get("name") or f"record-{len(library.fragments) + 1}"
        library.fragments.append(
            Fragment(
                index=len(library.fragments),
                fragment_id=str(fragment_id),
                smiles=smiles.strip(),
                change_types=_change_types(record),
            )
        )
    return library


def _molecule(fragment: Fragment) -> Chem.Mol | None:
    molecule = Chem.MolFromSmiles(fragment.smiles)
    if molecule is None:
        return None
    # Keep the attachment point visible but drop the atom map number, which would
    # otherwise print as a distracting "1" on every grid cell.
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
    return molecule


def _legend(fragment: Fragment) -> str:
    return fragment.fragment_id


def _page_size(total: int, columns: int, cell: int) -> int:
    """Fragments per page so one page stays under the pixel budget."""
    rows_budget = max(1, MAX_PAGE_PIXELS // max(1, columns * cell * cell))
    per_page = max(columns, columns * rows_budget)
    return min(per_page, total) if total else per_page


def render_library(
    library: Library,
    out_dir: Path,
    *,
    columns: int,
    cell: int,
    legends: bool,
    single_image: bool,
    id_pattern: str | None = None,
    label: str | None = None,
) -> tuple[list[Path], int, int]:
    """Write page images for one library; return (paths, drawn, skipped)."""
    # Two libraries can share a file name (fragments.json); qualify with the parent.
    slug = label or "-".join(library.path.parts[-2:]).replace(".json", "")
    if id_pattern:
        matcher = re.compile(id_pattern)
        library = Library(
            path=library.path,
            fragments=[item for item in library.fragments if matcher.search(item.fragment_id)],
            unparsable=library.unparsable,
        )
    molecules: list[Chem.Mol] = []
    kept: list[Fragment] = []
    skipped = 0
    for fragment in library.fragments:
        molecule = _molecule(fragment)
        if molecule is None:
            skipped += 1
            continue
        molecules.append(molecule)
        kept.append(fragment)

    if not molecules:
        return [], 0, skipped

    per_page = len(molecules) if single_image else _page_size(len(molecules), columns, cell)
    paths: list[Path] = []
    pages = math.ceil(len(molecules) / per_page)
    for page in range(pages):
        chunk = molecules[page * per_page:(page + 1) * per_page]
        chunk_fragments = kept[page * per_page:(page + 1) * per_page]
        options: dict = {
            "molsPerRow": columns,
            "subImgSize": (cell, cell),
            "useSVG": False,
            "returnPNG": False,
        }
        if legends:
            options["legends"] = [_legend(item) for item in chunk_fragments]
        image = Draw.MolsToGridImage(chunk, **options)
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        suffix = f"-p{page + 1:03d}" if pages > 1 else ""
        path = out_dir / f"{slug}{suffix}.png"
        image.save(path, optimize=True)
        paths.append(path)
        print(f"  {path.relative_to(out_dir.parent)}  {image.size[0]}x{image.size[1]}  "
              f"{len(chunk)} fragments")
    return paths, len(molecules), skipped


def write_html(
    sections: list[tuple[Library, list[Path], int, int]],
    out_dir: Path,
    *,
    filename: str = "index.html",
    title: str = "Fragment libraries - 2D structures",
) -> Path:
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{background:#fff;color:#111;font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}",
        "h2{font-size:15px;font-weight:600;margin:32px 0 8px;border-bottom:1px solid #ddd;padding-bottom:6px}",
        "img{display:block;width:100%;max-width:2400px;height:auto;margin:0 0 16px;border:1px solid #eee}",
        "</style></head><body>",
    ]
    for library, paths, drawn, skipped in sections:
        if not paths:
            continue
        try:
            label = library.path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            label = library.path.name
        note = f"{label} - {drawn} structures"
        if skipped:
            note += f" ({skipped} unparsable)"
        parts.append(f"<h2>{html.escape(note)}</h2>")
        for path in paths:
            parts.append(f'<img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}">')
    parts.append("</body></html>")
    index = out_dir / filename
    index.write_text("\n".join(parts), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", action="append", default=None,
                        help="library JSON to render; repeatable (default: all known libraries)")
    parser.add_argument("--out-dir", default="deliverables/fragment-libraries-2d")
    parser.add_argument("--columns", type=int, default=28)
    parser.add_argument("--cell", type=int, default=110)
    parser.add_argument("--legends", action="store_true",
                        help="draw the fragment id under each structure (default: structures only)")
    parser.add_argument("--single-image", action="store_true",
                        help="force one image per library even when it becomes very tall")
    parser.add_argument("--small-threshold", type=int, default=1000,
                        help="libraries with at most this many fragments also get their own HTML page")
    parser.add_argument("--small-html", default="index-small.html",
                        help="file name of the small-library page")
    parser.add_argument("--id-pattern", default=None,
                        help="regex; keep only fragments whose id matches")
    parser.add_argument("--label", default=None,
                        help="override the output image name prefix")
    args = parser.parse_args(argv)

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    libraries = args.library or DEFAULT_LIBRARIES
    sections: list[tuple[Library, list[Path], int, int]] = []
    for raw in libraries:
        path = (PROJECT_ROOT / raw).resolve()
        if not path.is_file():
            print(f"skip missing library: {raw}", file=sys.stderr)
            continue
        library = load_library(path)
        print(f"{raw}: {len(library.fragments)} fragments")
        paths, drawn, skipped = render_library(
            library,
            out_dir,
            columns=args.columns,
            cell=args.cell,
            legends=args.legends,
            single_image=args.single_image,
            id_pattern=args.id_pattern,
            label=args.label,
        )
        if skipped:
            print(f"  skipped {skipped} unparsable SMILES", file=sys.stderr)
        sections.append((library, paths, drawn, skipped))

    index = write_html(sections, out_dir)
    print(f"\n{index}")

    # The curated / task-specific libraries are small enough to review on their
    # own, without scrolling past the tens of thousands of ChEMBL fragments.
    small = [section for section in sections if len(section[0].fragments) <= args.small_threshold]
    if small and len(small) < len(sections):
        small_index = write_html(
            small,
            out_dir,
            filename=args.small_html,
            title="Curated and task-specific fragment libraries - 2D structures",
        )
        print(f"{small_index}  ({', '.join(section[0].path.name for section in small)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
