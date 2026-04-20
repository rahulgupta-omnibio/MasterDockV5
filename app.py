# -*- coding: utf-8 -*-
"""
MasterDock — Advanced Molecular Docking & Analysis Platform
Professional white-theme UI | AutoDock Vina | NGL 3D viewer | Discovery Studio-style 2D interactions
"""

import io, os, re, math, tempfile, subprocess, base64, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MasterDock",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS: Clean White Professional Theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --primary:   #2563eb;
  --primary-l: #eff6ff;
  --success:   #16a34a;
  --success-l: #f0fdf4;
  --warning:   #d97706;
  --warning-l: #fffbeb;
  --danger:    #dc2626;
  --danger-l:  #fef2f2;
  --purple:    #7c3aed;
  --gray-50:   #f9fafb;
  --gray-100:  #f3f4f6;
  --gray-200:  #e5e7eb;
  --gray-300:  #d1d5db;
  --gray-400:  #9ca3af;
  --gray-600:  #4b5563;
  --gray-700:  #374151;
  --gray-800:  #1f2937;
  --gray-900:  #111827;
  --white:     #ffffff;
  --shadow:    0 1px 3px rgba(0,0,0,.10), 0 1px 2px rgba(0,0,0,.06);
  --shadow-md: 0 4px 6px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.06);
  --radius:    8px;
  --font:      'Inter', sans-serif;
  --mono:      'JetBrains Mono', monospace;
}

/* Base */
.stApp { background: var(--gray-50) !important; font-family: var(--font); color: var(--gray-800); }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--white) !important;
  border-right: 1px solid var(--gray-200);
}
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* Tabs */
[data-testid="stTabs"] button {
  font-family: var(--font) !important;
  font-weight: 500;
  font-size: .88rem;
}

/* Cards */
.card {
  background: var(--white);
  border: 1px solid var(--gray-200);
  border-radius: var(--radius);
  padding: 16px 20px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}

/* Section headers */
.sec-hdr {
  display: flex; align-items: center; gap: 8px;
  font-size: .95rem; font-weight: 600;
  color: var(--gray-800);
  border-bottom: 2px solid var(--primary);
  padding-bottom: 6px;
  margin: 16px 0 12px;
}

/* Metric row */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 12px; }
.metric-card {
  background: var(--white);
  border: 1px solid var(--gray-200);
  border-radius: var(--radius);
  padding: 10px 16px;
  flex: 1; min-width: 90px;
  box-shadow: var(--shadow);
}
.metric-card .val {
  font-size: 1.5rem; font-weight: 700;
  color: var(--primary);
  font-family: var(--mono);
  line-height: 1.2;
}
.metric-card .lbl {
  font-size: .68rem; color: var(--gray-400);
  text-transform: uppercase; letter-spacing: .06em;
  margin-top: 2px;
}

/* Affinity badge */
.aff-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-weight: 700;
  font-family: var(--mono);
  font-size: .9rem;
}
.aff-strong { background: var(--success-l); color: var(--success); }
.aff-mod    { background: var(--primary-l); color: var(--primary); }
.aff-weak   { background: var(--warning-l); color: var(--warning); }

/* Info/warn boxes */
.info-box {
  background: var(--primary-l);
  border-left: 3px solid var(--primary);
  padding: 10px 14px; border-radius: 0 6px 6px 0;
  font-size: .85rem; color: var(--gray-700); margin: 8px 0;
}
.ok-box {
  background: var(--success-l);
  border-left: 3px solid var(--success);
  padding: 10px 14px; border-radius: 0 6px 6px 0;
  font-size: .85rem; color: var(--gray-700); margin: 8px 0;
}
.warn-box {
  background: var(--warning-l);
  border-left: 3px solid var(--warning);
  padding: 10px 14px; border-radius: 0 6px 6px 0;
  font-size: .85rem; color: var(--gray-700); margin: 8px 0;
}

/* Pose table */
.pose-tbl { width:100%; border-collapse:collapse; font-size:.84rem; font-family:var(--mono); }
.pose-tbl th { background:var(--gray-50); color:var(--gray-600); padding:8px 12px;
  text-align:left; border-bottom:2px solid var(--gray-200); font-weight:600; font-family:var(--font);}
.pose-tbl td { padding:7px 12px; border-bottom:1px solid var(--gray-100); }
.pose-tbl tr.best td { color:var(--success); font-weight:600; background:var(--success-l);}
.pose-tbl tr:hover td { background:var(--gray-50); }
.pose-tbl tr.sel td { background:var(--primary-l); }

/* Lipinski pills */
.rule-ok   { color:var(--success); font-weight:600; }
.rule-fail { color:var(--danger);  font-weight:600; }

/* Hide Streamlit branding */
#MainMenu, footer, [data-testid="stDecoration"] { display:none !important; }

/* Viewer container */
.viewer-wrap {
  border: 1px solid var(--gray-200);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
}

/* Plotly charts in white mode */
.js-plotly-plot .plotly { background: white !important; }

</style>
""", unsafe_allow_html=True)


# ─── NGL.js 3D Viewer (same engine as SeamDock) ─────────────────────────────
def ngl_viewer(pdb_str: str, ligand_pdb: str = "", height: int = 450,
               show_surface: bool = False, bg: str = "white") -> str:
    """
    NGL.js viewer — same engine SeamDock uses.
    Embeds structures directly via srcdoc/data URI for Streamlit compatibility.
    """
    lig_js = ""
    if ligand_pdb:
        lig_b64 = base64.b64encode(ligand_pdb.encode()).decode()
        lig_js = f"""
        var ligData = atob("{lig_b64}");
        var ligBlob = new Blob([ligData], {{type:'text/plain'}});
        var ligUrl = URL.createObjectURL(ligBlob);
        stage.loadFile(ligUrl, {{ext:'pdb', name:'ligand'}}).then(function(comp){{
            comp.addRepresentation('ball+stick', {{
                colorScheme:'element',
                radiusScale: 0.5,
                bondScale: 0.4
            }});
        }});
        """

    rec_b64 = base64.b64encode(pdb_str.encode()).decode()
    surf_js = ""
    if show_surface:
        surf_js = """
        comp.addRepresentation('surface', {
            opacity: 0.25, colorScheme: 'electrostatic',
            surfaceType: 'ms', probeRadius: 1.4
        });
        """

    html = f"""<!DOCTYPE html><html><head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:{bg}; width:100%; height:{height}px; overflow:hidden; }}
  #viewport {{ width:100%; height:{height}px; }}
</style>
</head><body>
<div id="viewport"></div>
<script src="https://unpkg.com/ngl@2.0.0-dev.38/dist/ngl.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var stage = new NGL.Stage('viewport', {{
    backgroundColor: '{bg}',
    quality: 'high',
    impostor: true
  }});

  // Resize handler
  window.addEventListener('resize', function(){{
    stage.handleResize();
  }});

  var recData = atob("{rec_b64}");
  var recBlob = new Blob([recData], {{type:'text/plain'}});
  var recUrl  = URL.createObjectURL(recBlob);

  stage.loadFile(recUrl, {{ext:'pdb', name:'receptor'}}).then(function(comp){{
    comp.addRepresentation('cartoon', {{
      colorScheme: 'chainindex',
      smoothSheet: true,
      aspectRatio: 4.0,
      cylinderOnly: false
    }});
    comp.addRepresentation('licorice', {{
      sele: 'hetero and not water',
      colorScheme: 'element',
      radiusScale: 0.5
    }});
    {surf_js}
    {lig_js}
    stage.autoView();
  }});
}});
</script></body></html>"""
    return html


def ngl_overlay(receptor_pdb: str, poses_list: list, height: int = 480) -> str:
    """NGL viewer showing receptor + all docked poses overlaid with different colors."""
    COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12',
              '#9b59b6','#1abc9c','#e67e22','#34495e',
              '#e91e63','#00bcd4']

    rec_b64 = base64.b64encode(receptor_pdb.encode()).decode()

    poses_js = ""
    for i, pose_pdb in enumerate(poses_list[:10]):
        color = COLORS[i % len(COLORS)]
        p_b64 = base64.b64encode(pose_pdb.encode()).decode()
        poses_js += f"""
        (function(idx, col){{
            var d = atob("{p_b64}");
            var b = new Blob([d], {{type:'text/plain'}});
            stage.loadFile(URL.createObjectURL(b), {{ext:'pdb'}}).then(function(c){{
                c.addRepresentation('ball+stick', {{
                    colorValue: col,
                    radiusScale: 0.5,
                    bondScale: 0.4,
                    opacity: 0.9
                }});
                if(idx === 0) stage.autoView();
            }});
        }})({i}, '{color}');
        """

    return f"""<!DOCTYPE html><html><head>
<style>*{{margin:0;padding:0}}body{{background:white;width:100%;height:{height}px;overflow:hidden}}
#v{{width:100%;height:{height}px}}</style></head><body>
<div id="v"></div>
<script src="https://unpkg.com/ngl@2.0.0-dev.38/dist/ngl.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function(){{
  var stage = new NGL.Stage('v', {{backgroundColor:'white', quality:'high'}});
  window.addEventListener('resize', function(){{ stage.handleResize(); }});
  var rd = atob("{rec_b64}");
  stage.loadFile(URL.createObjectURL(new Blob([rd],{{type:'text/plain'}})),{{ext:'pdb'}}).then(function(c){{
    c.addRepresentation('cartoon',{{colorScheme:'chainindex', opacity:0.4}});
    {poses_js}
  }});
}});
</script></body></html>"""


# ─── Discovery Studio-style 2D Interaction Diagram ───────────────────────────
def draw_2d_interaction_svg(ligand_mol, contacts_df, width=560, height=480) -> str:
    """
    Generate a Discovery Studio-style 2D protein-ligand interaction diagram.
    - Ligand in centre with 2D structure
    - Residue labels radiating outward as colored rounded boxes
    - Colored dashed lines for interaction types
    """
    if ligand_mol is None or contacts_df is None or contacts_df.empty:
        return ""

    # Interaction colour scheme (same as Discovery Studio)
    INT_COLORS = {
        "H-Bond":       "#2563eb",   # blue
        "Hydrophobic":  "#16a34a",   # green
        "π-Stacking":   "#7c3aed",   # purple
        "Electrostatic":"#dc2626",   # red
        "Covalent":     "#d97706",   # amber
        "Water":        "#0891b2",   # cyan
    }

    # Draw ligand 2D at centre
    cx, cy = width // 2, height // 2
    lig_w, lig_h = 240, 200

    try:
        mol_c = Chem.RWMol(ligand_mol)
        AllChem.Compute2DCoords(mol_c)
        drawer = rdMolDraw2D.MolDraw2DSVG(lig_w, lig_h)
        drawer.drawOptions().clearBackground = False
        drawer.drawOptions().addAtomIndices    = False
        drawer.drawOptions().addBondIndices    = False
        drawer.DrawMolecule(mol_c)
        drawer.FinishDrawing()
        lig_svg_inner = drawer.GetDrawingText()
        # Extract just the SVG content (no header)
        lig_svg_inner = re.sub(r'<\?xml[^>]*\?>', '', lig_svg_inner)
        lig_svg_inner = re.sub(r'<!DOCTYPE[^>]*>', '', lig_svg_inner)
        # Remove outer svg tag, keep contents
        inner_match = re.search(r'<svg[^>]*>(.*?)</svg>', lig_svg_inner, re.DOTALL)
        lig_inner_content = inner_match.group(1) if inner_match else ""
    except Exception:
        lig_inner_content = ""

    # Layout residues in a circle around the ligand
    contacts = contacts_df.head(12).to_dict('records')
    n = len(contacts)
    radius = min(width, height) * 0.38
    svg_parts = []

    # Background
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="white" rx="8"/>')

    # Draw connection lines FIRST (behind everything)
    for i, row in enumerate(contacts):
        angle = (2 * math.pi * i / n) - math.pi / 2
        res_x = cx + radius * math.cos(angle)
        res_y = cy + radius * math.sin(angle)

        itype  = row.get("Type", "Hydrophobic")
        color  = INT_COLORS.get(itype, "#6b7280")
        dist   = row.get("Min dist (Å)", 4.0)

        # Line from ligand edge to residue
        # Ligand edge point (approximate)
        lx = cx + (lig_w / 2 - 10) * math.cos(angle)
        ly = cy + (lig_h / 2 - 10) * math.sin(angle)
        # Residue box edge
        rx_end = res_x - 40 * math.cos(angle)
        ry_end = res_y - 16 * math.sin(angle)

        dash = "6,4" if itype in ("H-Bond", "Electrostatic") else "3,3"
        stroke_w = 2.2 if itype == "H-Bond" else 1.5
        svg_parts.append(
            f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{rx_end:.1f}" y2="{ry_end:.1f}" '
            f'stroke="{color}" stroke-width="{stroke_w}" stroke-dasharray="{dash}" opacity="0.85"/>'
        )
        # Distance label on the line
        mid_x = (lx + rx_end) / 2
        mid_y = (ly + ry_end) / 2
        svg_parts.append(
            f'<rect x="{mid_x-18:.1f}" y="{mid_y-9:.1f}" width="36" height="14" '
            f'fill="white" stroke="{color}" stroke-width="0.8" rx="3" opacity="0.9"/>'
        )
        svg_parts.append(
            f'<text x="{mid_x:.1f}" y="{mid_y+4:.1f}" text-anchor="middle" '
            f'font-size="8.5" font-family="JetBrains Mono,monospace" '
            f'fill="{color}" font-weight="500">{dist:.2f}Å</text>'
        )

    # Ligand box (white rounded rect)
    svg_parts.append(
        f'<rect x="{cx - lig_w//2}" y="{cy - lig_h//2}" width="{lig_w}" height="{lig_h}" '
        f'fill="white" stroke="#d1d5db" stroke-width="1.5" rx="8" '
        f'filter="url(#shadow)"/>'
    )

    # Embed ligand SVG
    if lig_inner_content:
        svg_parts.append(
            f'<g transform="translate({cx - lig_w//2},{cy - lig_h//2})">'
            f'{lig_inner_content}</g>'
        )

    # Residue boxes
    for i, row in enumerate(contacts):
        angle = (2 * math.pi * i / n) - math.pi / 2
        res_x = cx + radius * math.cos(angle)
        res_y = cy + radius * math.sin(angle)

        res_label = row.get("Residue", "UNK")
        itype     = row.get("Type", "Hydrophobic")
        color     = INT_COLORS.get(itype, "#6b7280")

        box_w = max(70, len(res_label) * 7 + 16)
        box_h = 26

        svg_parts.append(
            f'<rect x="{res_x - box_w//2:.1f}" y="{res_y - box_h//2:.1f}" '
            f'width="{box_w}" height="{box_h}" rx="13" '
            f'fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="1.5"/>'
        )
        svg_parts.append(
            f'<text x="{res_x:.1f}" y="{res_y + 5:.1f}" text-anchor="middle" '
            f'font-size="11" font-family="Inter,sans-serif" '
            f'font-weight="600" fill="{color}">{res_label}</text>'
        )

    # Legend
    legend_x, legend_y = 12, 12
    svg_parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="130" height="{len(INT_COLORS)*18+14}" '
        f'fill="white" stroke="#e5e7eb" stroke-width="1" rx="6" opacity="0.95"/>'
    )
    for j, (ltype, lcolor) in enumerate(INT_COLORS.items()):
        ly = legend_y + 10 + j * 18
        svg_parts.append(f'<line x1="{legend_x+8}" y1="{ly+5}" x2="{legend_x+26}" y2="{ly+5}" '
                         f'stroke="{lcolor}" stroke-width="2.5" stroke-dasharray="4,2"/>')
        svg_parts.append(f'<text x="{legend_x+32}" y="{ly+9}" font-size="9.5" '
                         f'font-family="Inter,sans-serif" fill="{lcolor}" font-weight="500">{ltype}</text>')

    # Compose full SVG with drop-shadow filter
    full_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.12"/>
  </filter>
</defs>
{''.join(svg_parts)}
</svg>"""
    return full_svg


# ─── Chemistry helpers ───────────────────────────────────────────────────────
def pdb_stats(txt: str) -> dict:
    p = PDBParser(QUIET=True)
    try:
        s = p.get_structure("m", io.StringIO(txt))
    except Exception as e:
        raise PDBConstructionException(str(e)) from e
    d = dict(Chains=0, Residues=0, Atoms=0, Models=0)
    for model in s:
        d["Models"] += 1
        for chain in model:
            d["Chains"] += 1
            for res in chain:
                d["Residues"] += 1
                for _ in res:
                    d["Atoms"] += 1
    return d


def pdb_centroid(txt: str):
    xs, ys, zs = [], [], []
    for line in txt.splitlines():
        if line.startswith(("ATOM","HETATM")):
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except ValueError:
                pass
    if not xs:
        return 0.0, 0.0, 0.0
    return round(sum(xs)/len(xs),2), round(sum(ys)/len(ys),2), round(sum(zs)/len(zs),2)


def mol_2d_svg(mol, w=300, h=220) -> str:
    if mol is None:
        return ""
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        d.drawOptions().clearBackground = False
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return ""


def mol_props(mol) -> dict:
    return {
        "MW":       round(Descriptors.MolWt(mol), 2),
        "LogP":     round(Descriptors.MolLogP(mol), 2),
        "HBD":      rdMolDescriptors.CalcNumHBD(mol),
        "HBA":      rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":     round(Descriptors.TPSA(mol), 1),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings":    rdMolDescriptors.CalcNumRings(mol),
        "HeavyAt":  mol.GetNumHeavyAtoms(),
    }


def lipinski(props) -> list:
    return [
        ("MW ≤ 500 Da",  props["MW"] <= 500,  f"{props['MW']} Da"),
        ("LogP ≤ 5",     props["LogP"] <= 5,   str(props["LogP"])),
        ("HBD ≤ 5",      props["HBD"] <= 5,    str(props["HBD"])),
        ("HBA ≤ 10",     props["HBA"] <= 10,   str(props["HBA"])),
        ("TPSA ≤ 140",   props["TPSA"] <= 140, f"{props['TPSA']} Å²"),
    ]


# ─── AD4 Atom types & PDBQT ─────────────────────────────────────────────────
AD4 = {
    "C":"C","N":"NA","O":"OA","S":"SA","H":"HD","P":"P","F":"F",
    "CL":"Cl","BR":"Br","I":"I","FE":"Fe","ZN":"Zn","CA":"Ca",
    "MG":"Mg","MN":"Mn","CU":"Cu","K":"K",
}

def pdb_to_pdbqt(pdb_text: str) -> str:
    """Pure-Python PDB→PDBQT. Confirmed working with Vina (column-based format)."""
    out = []
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM","HETATM"):
            element = ""
            if len(line) > 76:
                element = line[76:78].strip().upper()
            if not element:
                aname   = line[12:16].strip().lstrip("0123456789")
                element = "".join(c for c in aname if c.isalpha())[:2].upper()
            ad4 = AD4.get(element, AD4.get(element[:1], "C"))
            pdb_part   = line[:66].ljust(66)
            charge_str = f"{0.0:>10.3f}"
            type_str   = f" {ad4:<2}"
            out.append(pdb_part + charge_str + type_str)
        elif rec == "REMARK" or line.startswith(("TER","END","MODEL","ENDMDL")):
            out.append(line)
    return "\n".join(out) + "\n"


def ligand_to_pdbqt(mol) -> tuple:
    """Ligand RDKit mol → PDBQT via Meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        m = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(m)
        prep   = MoleculePreparation()
        setups = prep.prepare(m)
        pdbqt, _, _ = PDBQTWriterLegacy.write_string(setups[0])
        return pdbqt, ""
    except Exception as e:
        return None, str(e)


def ligand_pdbqt_to_pdb(pdbqt: str) -> str:
    """Convert ligand PDBQT to renderable PDB for NGL."""
    lines = []
    for line in pdbqt.splitlines():
        if line.startswith(("ATOM","HETATM","MODEL","ENDMDL","END","CONECT")):
            lines.append(line[:66])
    return "\n".join(lines) + "\n"


# ─── Vina docking ────────────────────────────────────────────────────────────
def run_vina(rec_pdbqt: str, lig_pdbqt: str,
             center: tuple, box_size: tuple,
             exhaustiveness: int = 8, n_poses: int = 9,
             scoring: str = "vina") -> tuple:
    try:
        from vina import Vina
    except ImportError:
        return False, None, "vina not installed"

    rf = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False, prefix="md_rec_")
    lf = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False, prefix="md_lig_")
    of = rf.name.replace("md_rec_", "md_out_")
    try:
        rf.write(rec_pdbqt); rf.close()
        lf.write(lig_pdbqt); lf.close()
        v = Vina(sf_name=scoring, cpu=0, verbosity=0)
        v.set_receptor(rf.name)
        v.set_ligand_from_file(lf.name)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        v.write_poses(of, n_poses=n_poses, overwrite=True)
        with open(of) as f:
            return True, f.read(), ""
    except Exception as e:
        return False, None, str(e)
    finally:
        for p in (rf.name, lf.name, of):
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass


def parse_scores(pdbqt: str) -> pd.DataFrame:
    rows = []
    for line in pdbqt.splitlines():
        m = re.match(r"REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if m:
            rows.append({
                "Pose":          len(rows)+1,
                "Affinity":      float(m.group(1)),
                "RMSD lb":       float(m.group(2)),
                "RMSD ub":       float(m.group(3)),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def split_poses(pdbqt: str) -> list:
    poses, cur = [], []
    for line in pdbqt.splitlines():
        cur.append(line)
        if line.startswith("ENDMDL"):
            poses.append("\n".join(cur))
            cur = []
    if cur:
        poses.append("\n".join(cur))
    return [p for p in poses if p.strip()]


# ─── Interaction analysis ────────────────────────────────────────────────────
HBOND_ATOMS   = {"N","O","F","S"}
PI_RESIDUES   = {"PHE","TYR","TRP","HIS"}
CHARGED_PLUS  = {"ARG","LYS","HIS"}
CHARGED_MINUS = {"ASP","GLU"}

def classify(dist: float, atom: str, resname: str) -> str:
    sym = "".join(c for c in atom if c.isalpha())[:2].upper()[:1]
    is_hb = sym in HBOND_ATOMS
    is_pi = resname.upper() in PI_RESIDUES
    is_water = resname.upper() in ("HOH","WAT","H2O")
    if is_water:        return "Water"
    if dist <= 3.5 and is_hb: return "H-Bond"
    if dist <= 4.0 and is_pi: return "π-Stacking"
    if resname.upper() in CHARGED_PLUS | CHARGED_MINUS: return "Electrostatic"
    return "Hydrophobic"


def find_contacts(receptor_pdb: str, ligand_pdbqt: str,
                  cutoff: float = 4.5) -> pd.DataFrame:
    lig_coords = []
    for line in ligand_pdbqt.splitlines():
        if line.startswith(("ATOM","HETATM")):
            try:
                lig_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    if not lig_coords:
        return pd.DataFrame()
    lig_arr = np.array(lig_coords)

    contacts = {}
    for line in receptor_pdb.splitlines():
        if not line.startswith(("ATOM","HETATM")):
            continue
        try:
            rc = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        dists = np.linalg.norm(lig_arr - rc, axis=1)
        d = float(dists.min())
        if d <= cutoff:
            res_name = line[17:20].strip()
            chain    = line[21].strip()
            res_num  = line[22:26].strip()
            atom_nm  = line[12:16].strip()
            key = f"{chain}:{res_name}{res_num}"
            if key not in contacts or contacts[key]["Min dist (Å)"] > round(d,2):
                contacts[key] = {
                    "Residue":      key,
                    "Res name":     res_name,
                    "Chain":        chain,
                    "Res num":      res_num,
                    "Nearest atom": atom_nm,
                    "Min dist (Å)": round(d,2),
                    "Type":         classify(d, atom_nm, res_name),
                }
    if not contacts:
        return pd.DataFrame()
    return pd.DataFrame(contacts.values()).sort_values("Min dist (Å)").reset_index(drop=True)


# ─── Plotly charts (white theme) ─────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", color="#374151"),
    margin=dict(l=50, r=20, t=30, b=50),
)
XAXIS = dict(gridcolor="#f3f4f6", linecolor="#d1d5db", zerolinecolor="#e5e7eb", tickfont=dict(size=11))
YAXIS = dict(gridcolor="#f3f4f6", linecolor="#d1d5db", zerolinecolor="#e5e7eb", tickfont=dict(size=11))


def affinity_chart(df: pd.DataFrame) -> go.Figure:
    colors = ["#16a34a" if i == 0 else "#2563eb" for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df["Pose"], y=df["Affinity"],
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"{v:.2f}" for v in df["Affinity"]],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    fig.update_layout(**LAYOUT, height=280,
        xaxis=dict(title="Pose", **XAXIS),
        yaxis=dict(title="Affinity (kcal/mol)", **YAXIS),
        showlegend=False,
    )
    return fig


def rmsd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Pose"], y=df["RMSD lb"],
        name="RMSD lb", mode="lines+markers",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=5, color="#2563eb")))
    fig.add_trace(go.Scatter(x=df["Pose"], y=df["RMSD ub"],
        name="RMSD ub", mode="lines+markers",
        line=dict(color="#d97706", width=2),
        marker=dict(size=5, color="#d97706")))
    fig.update_layout(**LAYOUT, height=280,
        xaxis=dict(title="Pose", **XAXIS),
        yaxis=dict(title="RMSD (Å)", **YAXIS),
        legend=dict(x=0.75, y=0.98, font=dict(size=11)),
    )
    return fig


def contacts_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return None
    INT_COLORS = {
        "H-Bond":       "#2563eb",
        "Hydrophobic":  "#16a34a",
        "π-Stacking":   "#7c3aed",
        "Electrostatic":"#dc2626",
        "Water":        "#0891b2",
        "Covalent":     "#d97706",
    }
    top = df.head(14).copy()
    clrs = [INT_COLORS.get(t,"#6b7280") for t in top["Type"]]
    fig = go.Figure(go.Bar(
        x=top["Min dist (Å)"], y=top["Residue"],
        orientation="h",
        marker=dict(color=clrs, line=dict(color="white", width=1)),
        text=[f"{d:.2f} Å" for d in top["Min dist (Å)"]],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    fig.update_layout(**LAYOUT, height=max(280, len(top)*28),
        xaxis=dict(title="Distance (Å)", **XAXIS),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, family="JetBrains Mono"), **YAXIS),
        showlegend=False,
    )
    return fig


def distance_hist(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return None
    fig = go.Figure(go.Histogram(
        x=df["Min dist (Å)"], nbinsx=12,
        marker=dict(color="#2563eb", opacity=0.8,
                    line=dict(color="white", width=1)),
    ))
    fig.add_vline(x=3.5, line_dash="dash", line_color="#16a34a",
                  annotation_text="H-bond", annotation_font_color="#16a34a",
                  annotation_font_size=10)
    fig.add_vline(x=4.5, line_dash="dash", line_color="#d97706",
                  annotation_text="VdW", annotation_font_color="#d97706",
                  annotation_font_size=10)
    fig.update_layout(**LAYOUT, height=220,
        xaxis=dict(title="Distance (Å)", **XAXIS),
        yaxis=dict(title="Count", **YAXIS),
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def interaction_donut(profile: dict) -> go.Figure:
    INT_COLORS = {
        "H-Bond":       "#2563eb",
        "Hydrophobic":  "#16a34a",
        "π-Stacking":   "#7c3aed",
        "Electrostatic":"#dc2626",
        "Water":        "#0891b2",
        "Covalent":     "#d97706",
    }
    labels = list(profile.keys())
    values = list(profile.values())
    clrs   = [INT_COLORS.get(l,"#6b7280") for l in labels]
    total  = sum(values)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=clrs, line=dict(color="white", width=2)),
        textfont=dict(size=11, family="Inter, sans-serif"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#374151"),
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(font=dict(size=11), orientation="v", x=1.02, y=0.5),
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:10px'>contacts</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, family="Inter, sans-serif", color="#374151")
        )],
    )
    return fig


def radar_chart(props: dict) -> go.Figure:
    cats = ["MW/500","LogP/5","HBD/5","HBA/10","TPSA/140","RotBonds/10"]
    vals = [
        min(props["MW"]/500, 1.5),
        min((props["LogP"]+2)/7, 1.5),
        min(props["HBD"]/5, 1.5),
        min(props["HBA"]/10, 1.5),
        min(props["TPSA"]/140, 1.5),
        min(props["RotBonds"]/10, 1.5),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1]*len(cats), theta=cats, fill="toself",
        name="Ro5 limit", fillcolor="rgba(22,163,74,0.08)",
        line=dict(color="#16a34a", width=1.5, dash="dash")))
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        name="Ligand", fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="#2563eb", width=2)))
    fig.update_layout(
        paper_bgcolor="white", font=dict(family="Inter", color="#374151"),
        height=300, margin=dict(l=50, r=50, t=20, b=20),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0,1.5],
                           gridcolor="#e5e7eb",
                           tickfont=dict(color="#9ca3af", size=9)),
            angularaxis=dict(gridcolor="#e5e7eb",
                             tickfont=dict(color="#374151", size=11)),
        ),
        legend=dict(font=dict(size=11), x=0.8, y=1.15),
        showlegend=True,
    )
    return fig


# ─── Session state ────────────────────────────────────────────────────────────
DEFAULTS = dict(
    rec_content=None, rec_pdbqt=None,
    ligands=[],          # list of {name, mol, smiles, sdf}
    active_lig=0,
    poses_pdbqt=None, scores_df=None, selected_pose=0,
    run_done=False,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 4px 8px;text-align:center">
      <div style="font-size:2rem;margin-bottom:4px">🔬</div>
      <div style="font-size:1.15rem;font-weight:700;color:#111827;font-family:Inter,sans-serif">
        MasterDock</div>
      <div style="font-size:.72rem;color:#6b7280;margin-top:2px">
        Advanced Docking Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Receptor
    st.markdown("**📥 Receptor (PDB)**")
    rec_file = st.file_uploader("Receptor", type=["pdb"],
                                label_visibility="collapsed", key="rec_up")

    st.divider()

    # Ligands — multiple
    st.markdown("**💊 Ligand(s)**")
    lig_mode = st.radio("Input", ["SMILES", "SDF/MOL files"],
                        horizontal=True, label_visibility="collapsed")
    if lig_mode == "SMILES":
        smiles_txt = st.text_area("SMILES (one per line)",
                                  placeholder="CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(O)=O",
                                  height=80, label_visibility="collapsed")
        lig_files = None
    else:
        lig_files = st.file_uploader("SDF/MOL files (multi-select)",
                                     type=["sdf","mol"],
                                     accept_multiple_files=True,
                                     label_visibility="collapsed", key="lig_up")
        smiles_txt = None

    st.divider()

    # Grid box
    st.markdown("**📦 Grid Box**")
    auto_center = st.checkbox("Auto-center on protein", value=True)
    if not auto_center:
        col1, col2, col3 = st.columns(3)
        cx = col1.number_input("X", value=0.0, format="%.1f")
        cy = col2.number_input("Y", value=0.0, format="%.1f")
        cz = col3.number_input("Z", value=0.0, format="%.1f")
    else:
        cx = cy = cz = 0.0

    col1, col2, col3 = st.columns(3)
    sx = col1.number_input("SX", value=25.0, format="%.0f")
    sy = col2.number_input("SY", value=25.0, format="%.0f")
    sz = col3.number_input("SZ", value=25.0, format="%.0f")

    st.divider()

    # Parameters
    st.markdown("**⚙️ Parameters**")
    scoring  = st.selectbox("Scoring", ["vina","vinardo","ad4"])
    exhaust  = st.slider("Exhaustiveness", 1, 32, 8)
    n_poses  = st.slider("Poses", 1, 20, 9)
    cutoff   = st.slider("Contact cutoff (Å)", 2.5, 6.0, 4.5, 0.5)
    show_surf = st.checkbox("Show receptor surface", value=False)

    st.divider()

    # Ligand selector if multiple
    if len(st.session_state.ligands) > 1:
        st.markdown("**🔀 Active Ligand**")
        lig_names = [l["name"] for l in st.session_state.ligands]
        st.session_state.active_lig = st.selectbox(
            "Select ligand for docking",
            range(len(lig_names)), format_func=lambda i: lig_names[i],
            label_visibility="collapsed")

    run_btn = st.button("🚀  Run Docking", use_container_width=True,
                        type="primary",
                        disabled=(rec_file is None and st.session_state.rec_content is None))

    if st.session_state.run_done:
        if st.button("🔄  Reset Results", use_container_width=True):
            for k in ("poses_pdbqt","scores_df","run_done"):
                st.session_state[k] = DEFAULTS[k]
            st.rerun()


# ─── Load structures ──────────────────────────────────────────────────────────
# Receptor
if rec_file is not None:
    st.session_state.rec_content = rec_file.read().decode("utf-8")

rec = st.session_state.rec_content

# Ligands — parse all
new_ligands = []
if lig_mode == "SMILES" and smiles_txt and smiles_txt.strip():
    for line in smiles_txt.strip().splitlines():
        smi = line.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol:
            new_ligands.append({
                "name":   Chem.MolToSmiles(mol)[:20]+"…" if len(smi)>20 else smi,
                "mol":    mol,
                "smiles": Chem.MolToSmiles(mol),
                "sdf":    None,
            })
elif lig_mode == "SDF/MOL files" and lig_files:
    for f in lig_files:
        raw = f.read().decode("utf-8")
        sup = Chem.SDMolSupplier()
        sup.SetData(raw, removeHs=False)
        for mol in sup:
            if mol:
                new_ligands.append({
                    "name":   f.name,
                    "mol":    mol,
                    "smiles": Chem.MolToSmiles(mol),
                    "sdf":    raw,
                })
if new_ligands:
    st.session_state.ligands = new_ligands
    st.session_state.active_lig = 0

ligands    = st.session_state.ligands
active_idx = st.session_state.active_lig
lig_data   = ligands[active_idx] if ligands else None
lig_mol    = lig_data["mol"] if lig_data else None


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:Inter,sans-serif;font-size:1.75rem;font-weight:700;
           color:#111827;margin:0 0 2px">
  MasterDock
  <span style="font-size:1rem;font-weight:400;color:#6b7280;margin-left:8px">
  · Advanced Docking Platform</span>
</h1>
<p style="color:#6b7280;font-size:.85rem;margin:0 0 8px">
  AutoDock Vina · 3D visualization · Binding analysis · Drug-likeness · Pose comparison
</p>
""", unsafe_allow_html=True)
st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Structure Viewer",
    "⚗️ Docking & Results",
    "📊 Analysis",
    "💊 Drug-likeness",
])


# ═══════════════ TAB 1 — Structure Viewer ════════════════════════════════════
with tab1:
    col_r, col_l = st.columns(2)

    # Receptor
    with col_r:
        st.markdown('<div class="sec-hdr">🔵 Receptor</div>', unsafe_allow_html=True)
        if rec:
            try:
                stats = pdb_stats(rec)
                st.markdown(f"""<div class="metric-row">
                  <div class="metric-card"><div class="val">{stats['Chains']}</div>
                    <div class="lbl">Chains</div></div>
                  <div class="metric-card"><div class="val">{stats['Residues']}</div>
                    <div class="lbl">Residues</div></div>
                  <div class="metric-card"><div class="val">{stats['Atoms']}</div>
                    <div class="lbl">Atoms</div></div>
                </div>""", unsafe_allow_html=True)

                if auto_center:
                    acx, acy, acz = pdb_centroid(rec)
                    st.markdown(
                        f'<div class="info-box">📍 Auto box center: '
                        f'<code>({acx}, {acy}, {acz})</code></div>',
                        unsafe_allow_html=True)

                html = ngl_viewer(rec, show_surface=show_surf, height=420)
                components.html(html, height=440, scrolling=False)
            except PDBConstructionException as e:
                st.error(f"PDB parse error: {e}")
        else:
            st.markdown('<div class="info-box">Upload a receptor PDB file in the sidebar.</div>',
                        unsafe_allow_html=True)

    # Ligand
    with col_l:
        st.markdown('<div class="sec-hdr">🟢 Ligand</div>', unsafe_allow_html=True)
        if lig_mol:
            props = mol_props(lig_mol)
            st.markdown(f"""<div class="metric-row">
              <div class="metric-card"><div class="val">{props['MW']}</div>
                <div class="lbl">MW (Da)</div></div>
              <div class="metric-card"><div class="val">{props['LogP']}</div>
                <div class="lbl">LogP</div></div>
              <div class="metric-card"><div class="val">{props['HBD']}/{props['HBA']}</div>
                <div class="lbl">HBD/HBA</div></div>
              <div class="metric-card"><div class="val">{props['TPSA']}</div>
                <div class="lbl">TPSA</div></div>
            </div>""", unsafe_allow_html=True)

            svg = mol_2d_svg(lig_mol, 360, 280)
            if svg:
                st.markdown(
                    f'<div class="viewer-wrap" style="background:white;padding:12px;'
                    f'text-align:center">{svg}</div>',
                    unsafe_allow_html=True)

            # Multiple ligands indicator
            if len(ligands) > 1:
                st.markdown(
                    f'<div class="ok-box">✅ {len(ligands)} ligands loaded — '
                    f'showing: <b>{lig_data["name"]}</b></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Enter SMILES or upload SDF in the sidebar.</div>',
                        unsafe_allow_html=True)


# ═══════════════ TAB 2 — Docking & Results ════════════════════════════════════
with tab2:

    # Run docking
    if run_btn:
        if not rec:
            st.error("Upload a receptor PDB first.")
        elif not lig_mol:
            st.error("Provide a ligand (SMILES or SDF).")
        else:
            # Receptor prep
            with st.spinner("⚙️ Preparing receptor…"):
                rec_pdbqt = pdb_to_pdbqt(rec)
                st.session_state.rec_pdbqt = rec_pdbqt

            # Ligand prep
            with st.spinner("💊 Preparing ligand…"):
                lig_pdbqt, lig_err = ligand_to_pdbqt(lig_mol)
            if not lig_pdbqt:
                st.error(f"Ligand preparation failed: {lig_err}")
                st.stop()

            # Center
            if auto_center:
                cx_use, cy_use, cz_use = pdb_centroid(rec)
            else:
                cx_use, cy_use, cz_use = cx, cy, cz

            # Dock
            with st.spinner(f"🚀 Running AutoDock Vina (exhaustiveness={exhaust})…"):
                ok, poses_pdbqt, err = run_vina(
                    rec_pdbqt, lig_pdbqt,
                    center=(cx_use, cy_use, cz_use),
                    box_size=(sx, sy, sz),
                    exhaustiveness=exhaust,
                    n_poses=n_poses,
                    scoring=scoring,
                )

            if ok:
                st.session_state.poses_pdbqt  = poses_pdbqt
                st.session_state.scores_df     = parse_scores(poses_pdbqt)
                st.session_state.run_done      = True
                st.session_state.selected_pose = 0
                st.success("✅ Docking complete!")
            else:
                st.error(f"Docking failed: {err}")

    # Results
    if st.session_state.run_done and st.session_state.poses_pdbqt:
        scores_df  = st.session_state.scores_df
        poses      = split_poses(st.session_state.poses_pdbqt)

        if not scores_df.empty:
            best = scores_df.iloc[0]["Affinity"]
            cls  = "aff-strong" if best < -8 else "aff-mod" if best < -6 else "aff-weak"
            st.markdown(
                f'<div class="ok-box">🎯 Best affinity: '
                f'<span class="aff-badge {cls}">{best:.2f} kcal/mol</span> '
                f'(Pose 1) &nbsp;·&nbsp; {len(poses)} poses generated</div>',
                unsafe_allow_html=True)

        col_v, col_t = st.columns([3, 2])

        with col_t:
            st.markdown('<div class="sec-hdr">📋 Pose Scores</div>', unsafe_allow_html=True)
            sel_idx = st.selectbox(
                "View pose",
                range(len(scores_df)),
                format_func=lambda i: f"Pose {int(scores_df.iloc[i]['Pose'])} "
                                      f"({scores_df.iloc[i]['Affinity']:.2f} kcal/mol)",
                key="pose_sel",
            )
            st.session_state.selected_pose = sel_idx

            # Score table
            rows_html = ""
            for _, row in scores_df.iterrows():
                cls = ' class="best"' if row["Pose"] == 1 else (
                      ' class="sel"'  if row["Pose"] == sel_idx+1 else "")
                rows_html += (
                    f"<tr{cls}>"
                    f"<td>{int(row['Pose'])}</td>"
                    f"<td style='font-weight:600'>{row['Affinity']:.2f}</td>"
                    f"<td>{row['RMSD lb']:.2f}</td>"
                    f"<td>{row['RMSD ub']:.2f}</td></tr>"
                )
            st.markdown(f"""<div style="overflow-x:auto">
            <table class="pose-tbl"><thead><tr>
              <th>Pose</th><th>Affinity (kcal/mol)</th>
              <th>RMSD lb (Å)</th><th>RMSD ub (Å)</th>
            </tr></thead><tbody>{rows_html}</tbody></table></div>""",
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download all poses (.pdbqt)",
                data=st.session_state.poses_pdbqt.encode(),
                file_name="docking_poses.pdbqt",
                mime="chemical/x-pdbqt",
                use_container_width=True,
            )

        with col_v:
            st.markdown('<div class="sec-hdr">🔬 3D Viewer</div>', unsafe_allow_html=True)
            sel_pose_pdbqt = poses[sel_idx] if sel_idx < len(poses) else poses[0]
            sel_pose_pdb   = ligand_pdbqt_to_pdb(sel_pose_pdbqt)
            show_rec = st.checkbox("Show receptor", value=True)

            html = ngl_viewer(
                rec if (show_rec and rec) else "END\n",
                ligand_pdb=sel_pose_pdb,
                show_surface=show_surf,
                height=420,
                bg="white",
            )
            components.html(html, height=445, scrolling=False)

    elif not run_btn:
        st.markdown("""<div class="card">
        <b style="font-size:1rem">How to use MasterDock</b><br><br>
        1️⃣ Upload a <b>Receptor PDB</b> file in the sidebar<br>
        2️⃣ Enter <b>SMILES</b> (one per line) or upload <b>multiple SDF files</b><br>
        3️⃣ Set the grid box (or use Auto-center) and parameters<br>
        4️⃣ Click <b>🚀 Run Docking</b><br>
        5️⃣ Explore results in the <b>Analysis</b> tab<br><br>
        <span style="color:#6b7280;font-size:.85rem">
        Docking runs locally via AutoDock Vina — no API, no IP blocking.
        </span>
        </div>""", unsafe_allow_html=True)


# ═══════════════ TAB 3 — Analysis ════════════════════════════════════════════
with tab3:
    if st.session_state.run_done and st.session_state.poses_pdbqt:
        scores_df = st.session_state.scores_df
        poses     = split_poses(st.session_state.poses_pdbqt)
        sel_idx   = st.session_state.selected_pose
        sel_pdbqt = poses[sel_idx] if sel_idx < len(poses) else poses[0]

        # Contacts
        with st.spinner("Computing binding contacts…"):
            contacts_df = find_contacts(
                st.session_state.rec_content or "", sel_pdbqt, cutoff)

        profile = {}
        if not contacts_df.empty:
            for t in contacts_df["Type"]:
                profile[t] = profile.get(t, 0) + 1

        # ── Energy Analysis
        st.markdown('<div class="sec-hdr">📈 Energy Analysis</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Binding affinity per pose")
            st.plotly_chart(affinity_chart(scores_df),
                            use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.caption("RMSD of poses vs best")
            st.plotly_chart(rmsd_chart(scores_df),
                            use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # ── 2D Interaction Profile
        st.markdown('<div class="sec-hdr">🔗 2D Interaction Diagram</div>',
                    unsafe_allow_html=True)
        st.caption("Discovery Studio-style diagram — residue labels with bond type colours")

        c1, c2, c3 = st.columns([2.5, 1.5, 1.5])
        with c1:
            diag_svg = draw_2d_interaction_svg(lig_mol, contacts_df, 540, 460)
            if diag_svg:
                st.markdown(
                    f'<div class="viewer-wrap" style="background:white;padding:4px">'
                    f'{diag_svg}</div>',
                    unsafe_allow_html=True)
            else:
                st.info("Run docking first, or upload a ligand SDF/SMILES.")

        with c2:
            st.caption("Interaction type breakdown")
            if profile:
                fig_d = interaction_donut(profile)
                st.plotly_chart(fig_d, use_container_width=True,
                                config={"displayModeBar": False})

        with c3:
            st.caption("Contact distance histogram")
            if not contacts_df.empty:
                fig_h = distance_hist(contacts_df)
                if fig_h:
                    st.plotly_chart(fig_h, use_container_width=True,
                                    config={"displayModeBar": False})

        st.divider()

        # ── Binding Residues
        st.markdown('<div class="sec-hdr">🤝 Binding Residues</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])

        with c1:
            if not contacts_df.empty:
                show_df = contacts_df[
                    ["Residue","Res name","Min dist (Å)","Nearest atom","Type"]
                ].copy()
                # Colour-code the Type column via display
                st.dataframe(
                    show_df, use_container_width=True, height=340,
                    column_config={
                        "Min dist (Å)": st.column_config.NumberColumn(format="%.2f Å"),
                        "Type": st.column_config.TextColumn(),
                    }
                )
                st.download_button(
                    "⬇️ Download contacts (.csv)",
                    data=contacts_df.to_csv(index=False).encode(),
                    file_name="binding_contacts.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    f'<div class="warn-box">No contacts within {cutoff} Å — '
                    f'increase cutoff or check box center.</div>',
                    unsafe_allow_html=True)

        with c2:
            st.caption("Nearest contact distance per residue")
            cfig = contacts_chart(contacts_df)
            if cfig:
                st.plotly_chart(cfig, use_container_width=True,
                                config={"displayModeBar": False})

        st.divider()

        # ── All Poses Overlay
        st.markdown('<div class="sec-hdr">🔬 All Poses Overlay</div>',
                    unsafe_allow_html=True)

        pose_pdbs = [ligand_pdbqt_to_pdb(p) for p in poses]
        overlay_html = ngl_overlay(
            st.session_state.rec_content or "END\n",
            pose_pdbs, height=460
        )
        components.html(overlay_html, height=480, scrolling=False)

        # Pose colour legend
        COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12',
                  '#9b59b6','#1abc9c','#e67e22','#34495e',
                  '#e91e63','#00bcd4']
        legend = " &nbsp; ".join(
            f'<span style="color:{COLORS[i%len(COLORS)]};font-weight:700">⬤</span> '
            f'<span style="font-size:.8rem">Pose {i+1} '
            f'({scores_df.iloc[i]["Affinity"]:.2f})</span>'
            for i in range(min(len(scores_df), len(poses)))
        )
        st.markdown(
            f'<div style="text-align:center;margin-top:6px">{legend}</div>',
            unsafe_allow_html=True)

    else:
        st.markdown('<div class="info-box">Run a docking calculation first to see analysis.</div>',
                    unsafe_allow_html=True)


# ═══════════════ TAB 4 — Drug-likeness ═══════════════════════════════════════
with tab4:
    if lig_mol:
        props = mol_props(lig_mol)
        rules = lipinski(props)
        passed = sum(1 for _, ok, _ in rules if ok)

        c1, c2 = st.columns([2, 3])

        with c1:
            st.markdown('<div class="sec-hdr">🧪 Molecular Properties</div>',
                        unsafe_allow_html=True)
            prop_rows = [
                ("Molecular Weight",   f"{props['MW']} Da"),
                ("LogP",               str(props["LogP"])),
                ("H-Bond Donors",      str(props["HBD"])),
                ("H-Bond Acceptors",   str(props["HBA"])),
                ("TPSA",               f"{props['TPSA']} Å²"),
                ("Rotatable Bonds",    str(props["RotBonds"])),
                ("Ring Count",         str(props["Rings"])),
                ("Heavy Atoms",        str(props["HeavyAt"])),
                ("Formula",            rdMolDescriptors.CalcMolFormula(lig_mol)),
            ]
            tbl = "".join(
                f"<tr><td style='color:#6b7280;padding:7px 10px;font-size:.85rem'>{k}</td>"
                f"<td style='font-family:var(--mono);padding:7px 10px;font-size:.85rem;"
                f"font-weight:600'>{v}</td></tr>"
                for k, v in prop_rows
            )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;background:white;'
                f'border:1px solid #e5e7eb;border-radius:8px;overflow:hidden">'
                f'<tbody>{tbl}</tbody></table>',
                unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr" style="margin-top:16px">✅ Lipinski Ro5</div>',
                        unsafe_allow_html=True)
            for rule, ok, val in rules:
                icon  = "✅" if ok else "❌"
                color = "#16a34a" if ok else "#dc2626"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid #f3f4f6">'
                    f'<span>{icon} {rule}</span>'
                    f'<span style="color:{color};font-weight:600;'
                    f'font-family:var(--mono)">{val}</span></div>',
                    unsafe_allow_html=True)

            verdict_color = "#16a34a" if passed==5 else "#d97706" if passed>=3 else "#dc2626"
            verdict_text  = "Drug-like" if passed==5 else "Borderline" if passed>=3 else "Unlikely drug-like"
            st.markdown(
                f'<div style="margin-top:12px;padding:10px 14px;background:{verdict_color}20;'
                f'border-radius:8px;text-align:center;font-weight:700;color:{verdict_color}">'
                f'{verdict_text} ({passed}/5 rules passed)</div>',
                unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sec-hdr">🕸️ Property Radar</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(radar_chart(props), use_container_width=True,
                            config={"displayModeBar": False})

            st.markdown('<div class="sec-hdr">🖼️ 2D Structure</div>',
                        unsafe_allow_html=True)
            svg = mol_2d_svg(lig_mol, 420, 300)
            if svg:
                st.markdown(
                    f'<div class="viewer-wrap" style="padding:12px;background:white;'
                    f'text-align:center">{svg}</div>',
                    unsafe_allow_html=True)

            smiles_disp = lig_data["smiles"] if lig_data else ""
            st.markdown(
                f'<div class="info-box" style="margin-top:10px;word-break:break-all">'
                f'<b>SMILES:</b> <code style="font-size:.78rem">{smiles_disp}</code>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Provide a ligand to see drug-likeness analysis.</div>',
                    unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#9ca3af;font-size:.75rem;padding:4px 0">'
    'MasterDock · AutoDock Vina · NGL Viewer · RDKit · Meeko · Biopython'
    '</div>',
    unsafe_allow_html=True)
