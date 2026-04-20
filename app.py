# -*- coding: utf-8 -*-
"""
MasterDock — Advanced Molecular Docking & Analysis Platform
Docking   : AutoDock Vina (Python package, runs locally on Streamlit Cloud)
Prep      : Meeko (ligand PDBQT), OpenBabel (receptor PDBQT)
3D view   : 3Dmol.js via CDN
2D view   : RDKit
Analysis  : Binding interactions, energy plots, pose comparison
"""

import io
import os
import re
import math
import tempfile
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import (AllChem, Draw, rdMolDescriptors,
                        rdMolTransforms, Descriptors)
from rdkit.Chem.Draw import rdMolDraw2D

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MasterDock",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Theme CSS  — dark scientific aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;500;600&display=swap');

:root {
  --bg:      #0d1117;
  --bg2:     #161b22;
  --bg3:     #21262d;
  --border:  #30363d;
  --accent:  #58a6ff;
  --accent2: #3fb950;
  --warn:    #e3b341;
  --danger:  #f85149;
  --text:    #c9d1d9;
  --muted:   #8b949e;
  --mono:    'JetBrains Mono', monospace;
  --sans:    'Space Grotesk', sans-serif;
}

.stApp { background: var(--bg); font-family: var(--sans); color: var(--text); }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--bg2);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--sans); }

/* Section headers */
.sec { 
  display: flex; align-items: center; gap: 10px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px; margin: 20px 0 14px;
  font-size: 1rem; font-weight: 600; color: var(--accent);
  font-family: var(--sans);
}
.sec .icon { font-size: 1.2rem; }

/* Metric cards */
.metrics-row { display: flex; gap: 12px; margin: 10px 0; flex-wrap: wrap; }
.metric-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 110px;
}
.metric-card .val {
  font-size: 1.6rem; font-weight: 600; color: var(--accent);
  font-family: var(--mono);
}
.metric-card .lbl {
  font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .05em; margin-top: 2px;
}

/* Pose table */
.pose-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.85rem; }
.pose-table th { background: var(--bg3); color: var(--muted); padding: 8px 12px;
  text-align: left; border-bottom: 1px solid var(--border); font-weight: 500; }
.pose-table td { padding: 7px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.pose-table tr.best td { color: var(--accent2); font-weight: 600; }
.pose-table tr:hover td { background: var(--bg3); }

/* Info boxes */
.info-box {
  background: var(--bg2); border-left: 3px solid var(--accent);
  padding: 10px 14px; border-radius: 0 6px 6px 0;
  font-size: .85rem; color: var(--text); margin: 8px 0;
}
.warn-box {
  background: #272115; border-left: 3px solid var(--warn);
  padding: 10px 14px; border-radius: 0 6px 6px 0;
  font-size: .85rem; color: var(--text); margin: 8px 0;
}
.ok-box {
  background: #122018; border-left: 3px solid var(--accent2);
  padding: 10px 14px; border-radius: 0 6px 6px 0;
  font-size: .85rem; color: var(--text); margin: 8px 0;
}

/* Pill badge */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 12px;
  font-size: 0.72rem; font-weight: 600; font-family: var(--mono);
}
.badge-blue  { background: #0d2644; color: var(--accent); }
.badge-green { background: #0d2018; color: var(--accent2); }
.badge-warn  { background: #272115; color: var(--warn); }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  3-D Viewer — 3Dmol.js CDN
# ─────────────────────────────────────────────────────────────────────────────
def viewer_html(molecules: list, height: int = 480) -> str:
    """
    molecules = list of {"data": str, "fmt": str, "style": dict, "color": str}
    fmt: 'pdb' | 'pdbqt' | 'sdf' | 'mol2'
    """
    model_js = ""
    for i, mol in enumerate(molecules):
        safe = (mol["data"]
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n"))
        fmt   = mol.get("fmt", "pdb")
        style = mol.get("style", "stick")
        color = mol.get("color", "")
        color_opt = f', color: "{color}"' if color else ""

        if style == "cartoon":
            style_js = f"viewer.setStyle({{model:{i}}}, {{cartoon:{{color:'spectrum'}}}});"
        elif style == "surface":
            style_js = (f"viewer.setStyle({{model:{i}}}, {{stick:{{}}}});"
                        f"viewer.addSurface($3Dmol.SurfaceType.VDW,"
                        f"{{opacity:0.6,color:'lightblue'}},{{model:{i}}});")
        elif style == "sphere":
            style_js = f"viewer.setStyle({{model:{i}}}, {{sphere:{{scale:0.4{color_opt}}}}});"
        else:
            style_js = f"viewer.setStyle({{model:{i}}}, {{stick:{{{color_opt}}}}});"

        model_js += f"""
        viewer.addModel('{safe}', '{fmt}');
        {style_js}"""

    return f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;padding:0;background:#0d1117}}
#v{{width:100%;height:{height}px;position:relative}}</style></head><body>
<div id="v"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
<script>
var viewer = $3Dmol.createViewer(document.getElementById('v'),
  {{backgroundColor:0x0d1117}});
{model_js}
viewer.zoomTo();
viewer.render();
</script></body></html>"""


def viewer_with_box(receptor: str, cx: float, cy: float, cz: float,
                    sx: float, sy: float, sz: float, height: int = 480) -> str:
    """Receptor + docking box visualiser."""
    safe = (receptor.replace("\\", "\\\\")
                    .replace("'", "\\'")
                    .replace("\n", "\\n"))
    return f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;padding:0;background:#0d1117}}
#v{{width:100%;height:{height}px}}</style></head><body>
<div id="v"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
<script>
var viewer = $3Dmol.createViewer(document.getElementById('v'),{{backgroundColor:0x0d1117}});
viewer.addModel('{safe}','pdb');
viewer.setStyle({{}},{{cartoon:{{color:'spectrum'}}}});
viewer.addBox({{center:{{x:{cx},y:{cy},z:{cz}}},
               dimensions:{{w:{sx},h:{sy},d:{sz}}},
               color:'#58a6ff',opacity:0.15,wireframe:true}});
viewer.zoomTo();
viewer.render();
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Chemistry helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_pdb_stats(txt: str) -> dict:
    p = PDBParser(QUIET=True)
    try:
        s = p.get_structure("m", io.StringIO(txt))
    except Exception as e:
        raise PDBConstructionException(str(e)) from e
    d = dict(Models=0, Chains=0, Residues=0, Atoms=0)
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
        if line.startswith(("ATOM", "HETATM")):
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except ValueError:
                pass
    if not xs:
        return 0.0, 0.0, 0.0
    return round(sum(xs)/len(xs), 2), round(sum(ys)/len(ys), 2), round(sum(zs)/len(zs), 2)


def parse_sdf(txt: str):
    sup = Chem.SDMolSupplier()
    sup.SetData(txt, removeHs=False)
    return [m for m in sup if m is not None]


def mol_2d_svg(mol, size=(300, 250)) -> str:
    if mol is None:
        return ""
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        d.drawOptions().clearBackground = False
        d.DrawMolecule(mol)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        # Dark theme SVG
        svg = svg.replace('fill:#FFFFFF', 'fill:#0d1117')
        svg = svg.replace('fill:white',   'fill:#0d1117')
        return svg
    except Exception:
        return ""


def mol_props(mol) -> dict:
    return {
        "MW":          round(Descriptors.MolWt(mol), 2),
        "LogP":        round(Descriptors.MolLogP(mol), 2),
        "HBD":         rdMolDescriptors.CalcNumHBD(mol),
        "HBA":         rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":        round(Descriptors.TPSA(mol), 1),
        "RotBonds":    rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings":       rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtoms":  mol.GetNumHeavyAtoms(),
    }


def lipinski_check(props: dict) -> list:
    rules = [
        ("MW ≤ 500",     props["MW"] <= 500),
        ("LogP ≤ 5",     props["LogP"] <= 5),
        ("HBD ≤ 5",      props["HBD"] <= 5),
        ("HBA ≤ 10",     props["HBA"] <= 10),
        ("TPSA ≤ 140",   props["TPSA"] <= 140),
    ]
    return rules


# ─────────────────────────────────────────────────────────────────────────────
#  Docking preparation (Meeko + OpenBabel)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_receptor_pdbqt(pdb_content: str) -> tuple:
    """Convert PDB → PDBQT via OpenBabel. Returns (pdbqt_string, error_msg)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content); inp = f.name
    out = inp.replace(".pdb", ".pdbqt")
    try:
        r = subprocess.run(
            ["obabel", "-ipdb", inp, "-opdbqt", "-O", out, "-xr", "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=60
        )
        if os.path.exists(out):
            with open(out) as f:
                return f.read(), ""
        return None, r.stderr or "OpenBabel produced no output"
    except FileNotFoundError:
        return None, "openbabel not found — add to packages.txt"
    except Exception as e:
        return None, str(e)
    finally:
        for p in (inp, out):
            if os.path.exists(p): os.remove(p)


def smiles_to_pdbqt(smiles: str) -> tuple:
    """Convert SMILES → PDBQT via Meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        preparator = MoleculePreparation()
        mol_setups  = preparator.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        return pdbqt_str, ""
    except ImportError:
        return smiles_to_pdbqt_obabel(smiles)
    except Exception as e:
        return None, str(e)


def smiles_to_pdbqt_obabel(smiles: str) -> tuple:
    """Fallback: SMILES → PDBQT via OpenBabel."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as f:
        f.write(smiles); inp = f.name
    out = inp.replace(".smi", ".pdbqt")
    try:
        r = subprocess.run(
            ["obabel", "-ismi", inp, "-opdbqt", "-O", out,
             "--gen3d", "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=60
        )
        if os.path.exists(out):
            with open(out) as f:
                return f.read(), ""
        return None, r.stderr or "OpenBabel produced no output"
    except Exception as e:
        return None, str(e)
    finally:
        for p in (inp, out):
            if os.path.exists(p): os.remove(p)


def sdf_to_pdbqt(sdf_content: str) -> tuple:
    """SDF → PDBQT via Meeko (preferred) or OpenBabel."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        sup = Chem.SDMolSupplier()
        sup.SetData(sdf_content)
        mols = [m for m in sup if m is not None]
        if not mols:
            return None, "No valid molecules in SDF"
        mol = mols[0]
        mol = Chem.AddHs(mol)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        return pdbqt_str, ""
    except ImportError:
        # Fallback to obabel
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
            f.write(sdf_content); inp = f.name
        out = inp.replace(".sdf", ".pdbqt")
        try:
            r = subprocess.run(
                ["obabel", "-isdf", inp, "-opdbqt", "-O", out,
                 "--partialcharge", "gasteiger"],
                capture_output=True, text=True, timeout=60
            )
            if os.path.exists(out):
                with open(out) as f:
                    return f.read(), ""
            return None, r.stderr or "No output"
        except Exception as e:
            return None, str(e)
        finally:
            for p in (inp, out):
                if os.path.exists(p): os.remove(p)
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  AutoDock Vina docking (Python API)
# ─────────────────────────────────────────────────────────────────────────────
def run_vina(receptor_pdbqt: str, ligand_pdbqt: str,
             center: tuple, box_size: tuple,
             exhaustiveness: int = 8, n_poses: int = 9,
             scoring: str = "vina") -> tuple:
    """
    Run AutoDock Vina via the pure Python `vina` package.
    Returns (success, poses_pdbqt_string, error_message).
    """
    try:
        from vina import Vina
    except ImportError:
        return False, None, "vina package not installed — add `vina` to requirements.txt"

    rec_f = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
    lig_f = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
    out_f = tempfile.mktemp(suffix="_out.pdbqt")
    rec_f.write(receptor_pdbqt); rec_f.flush()
    lig_f.write(ligand_pdbqt);   lig_f.flush()
    rec_f.close(); lig_f.close()

    try:
        v = Vina(sf_name=scoring, cpu=0, verbosity=0)
        v.set_receptor(rec_f.name)
        v.set_ligand_from_file(lig_f.name)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        v.write_poses(out_f, n_poses=n_poses, overwrite=True)
        with open(out_f) as f:
            poses_str = f.read()
        return True, poses_str, ""
    except Exception as e:
        return False, None, str(e)
    finally:
        for p in (rec_f.name, lig_f.name, out_f):
            if os.path.exists(p): os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
#  Result parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_vina_poses(pdbqt_str: str) -> pd.DataFrame:
    """Extract pose scores from Vina PDBQT output."""
    rows = []
    for line in pdbqt_str.splitlines():
        m = re.match(r"REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if m:
            rows.append({
                "Pose":          len(rows) + 1,
                "Affinity (kcal/mol)": float(m.group(1)),
                "RMSD lb (Å)":   float(m.group(2)),
                "RMSD ub (Å)":   float(m.group(3)),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def split_pdbqt_poses(pdbqt_str: str) -> list:
    """Split multi-pose PDBQT into individual pose strings."""
    poses, current = [], []
    for line in pdbqt_str.splitlines():
        current.append(line)
        if line.startswith("ENDMDL"):
            poses.append("\n".join(current))
            current = []
    if current:
        poses.append("\n".join(current))
    return [p for p in poses if p.strip()]


def pdbqt_to_pdb(pdbqt_str: str) -> str:
    """Strip PDBQT-specific fields to get renderable PDB."""
    lines = []
    for line in pdbqt_str.splitlines():
        if line.startswith(("ATOM", "HETATM", "CONECT", "MODEL", "ENDMDL", "END")):
            lines.append(line[:66])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Interaction analysis
# ─────────────────────────────────────────────────────────────────────────────
def find_contacts(receptor_pdb: str, ligand_pdbqt: str,
                  cutoff: float = 4.5) -> pd.DataFrame:
    """Find residues within cutoff Å of any ligand atom."""
    lig_coords = []
    for line in ligand_pdbqt.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                lig_coords.append([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ])
            except ValueError:
                pass
    if not lig_coords:
        return pd.DataFrame()
    lig_arr = np.array(lig_coords)

    contacts = {}
    for line in receptor_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            rec_coord = np.array([
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            ])
        except ValueError:
            continue
        dists = np.linalg.norm(lig_arr - rec_coord, axis=1)
        min_d = dists.min()
        if min_d <= cutoff:
            res_name = line[17:20].strip()
            chain    = line[21].strip()
            res_num  = line[22:26].strip()
            atom_nm  = line[12:16].strip()
            key = f"{chain}:{res_name}{res_num}"
            if key not in contacts or contacts[key]["Min dist (Å)"] > round(min_d, 2):
                contacts[key] = {
                    "Residue": key,
                    "Res name": res_name,
                    "Chain": chain,
                    "Res num": res_num,
                    "Nearest atom": atom_nm,
                    "Min dist (Å)": round(min_d, 2),
                }
    if not contacts:
        return pd.DataFrame()
    df = pd.DataFrame(contacts.values()).sort_values("Min dist (Å)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Plotly charts
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
    gridcolor="#21262d",
)


def affinity_bar_chart(df: pd.DataFrame) -> go.Figure:
    colors = ["#3fb950" if i == 0 else "#58a6ff" for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df["Pose"],
        y=df["Affinity (kcal/mol)"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in df["Affinity (kcal/mol)"]],
        textposition="outside",
    ))
    fig.update_layout(
        **DARK,
        xaxis_title="Pose", yaxis_title="Affinity (kcal/mol)",
        showlegend=False, height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    return fig


def radar_chart(props: dict) -> go.Figure:
    cats   = ["MW/500", "LogP/5", "HBD/5", "HBA/10", "TPSA/140", "RotBonds/10"]
    vals   = [
        min(props["MW"] / 500, 1.5),
        min((props["LogP"] + 2) / 7, 1.5),
        min(props["HBD"] / 5, 1.5),
        min(props["HBA"] / 10, 1.5),
        min(props["TPSA"] / 140, 1.5),
        min(props["RotBonds"] / 10, 1.5),
    ]
    ideal  = [1.0] * len(cats)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal, theta=cats, fill="toself",
        name="Ideal (Ro5)", fillcolor="rgba(63,185,80,0.1)",
        line=dict(color="#3fb950", width=1, dash="dash")))
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        name="Ligand", fillcolor="rgba(88,166,255,0.2)",
        line=dict(color="#58a6ff", width=2)))
    fig.update_layout(
        **DARK, height=320,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1.5],
                           gridcolor="#30363d", tickfont_color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", tickfont_color="#c9d1d9"),
        ),
        legend=dict(x=0.8, y=1.1, font_size=11),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


def contacts_chart(contacts_df: pd.DataFrame) -> go.Figure:
    if contacts_df.empty:
        return None
    df = contacts_df.head(15).copy()
    colors = ["#3fb950" if d <= 3.5 else "#58a6ff" if d <= 4.0 else "#e3b341"
              for d in df["Min dist (Å)"]]
    fig = go.Figure(go.Bar(
        x=df["Min dist (Å)"], y=df["Residue"],
        orientation="h", marker_color=colors,
        text=[f"{v} Å" for v in df["Min dist (Å)"]],
        textposition="outside",
    ))
    fig.update_layout(
        **DARK, height=max(280, len(df) * 22),
        xaxis_title="Distance (Å)", yaxis_title="",
        showlegend=False,
        margin=dict(l=100, r=80, t=20, b=40),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    return fig


def rmsd_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Pose"], y=df["RMSD lb (Å)"],
        name="RMSD lb", mode="lines+markers",
        line=dict(color="#58a6ff"), marker=dict(size=6)))
    fig.add_trace(go.Scatter(
        x=df["Pose"], y=df["RMSD ub (Å)"],
        name="RMSD ub", mode="lines+markers",
        line=dict(color="#e3b341"), marker=dict(size=6)))
    fig.update_layout(
        **DARK, xaxis_title="Pose", yaxis_title="RMSD (Å)",
        height=260, margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(x=0.7, y=1.0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
defaults = dict(
    rec_pdbqt=None, rec_content=None,
    lig_pdbqt=None, lig_smiles=None, lig_sdf=None,
    poses_pdbqt=None, scores_df=None,
    selected_pose=0,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 10px">
      <div style="font-size:2rem">🧬</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;
                  font-weight:600;color:#58a6ff;letter-spacing:.02em">MasterDock</div>
      <div style="font-size:.72rem;color:#8b949e;margin-top:2px">
        Advanced Molecular Docking Platform
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Receptor
    st.markdown("**📥 Receptor (PDB)**")
    rec_file = st.file_uploader("Upload PDB file", type=["pdb"], key="rec_upload",
                                label_visibility="collapsed")

    # ── Ligand
    st.markdown("**💊 Ligand**")
    lig_tab = st.radio("Input method", ["SMILES", "SDF file"], horizontal=True,
                       label_visibility="collapsed")
    if lig_tab == "SMILES":
        smiles_in = st.text_input("SMILES string",
                                  placeholder="CC(=O)Oc1ccccc1C(=O)O",
                                  label_visibility="collapsed")
        lig_file = None
    else:
        lig_file  = st.file_uploader("SDF file", type=["sdf"], key="lig_upload",
                                     label_visibility="collapsed")
        smiles_in = None

    st.divider()

    # ── Docking box
    st.markdown("**📦 Grid Box**")
    auto_center = st.checkbox("Auto-center on protein", value=True)
    if not auto_center:
        c1, c2, c3 = st.columns(3)
        cx = c1.number_input("X", value=0.0, format="%.1f", key="cx")
        cy = c2.number_input("Y", value=0.0, format="%.1f", key="cy")
        cz = c3.number_input("Z", value=0.0, format="%.1f", key="cz")
    else:
        cx = cy = cz = 0.0

    c1, c2, c3 = st.columns(3)
    sx = c1.number_input("SX", value=25.0, format="%.0f", key="sx")
    sy = c2.number_input("SY", value=25.0, format="%.0f", key="sy")
    sz = c3.number_input("SZ", value=25.0, format="%.0f", key="sz")

    st.divider()

    # ── Docking parameters
    st.markdown("**⚙️ Parameters**")
    scoring   = st.selectbox("Scoring function",
                             ["vina", "vinardo", "ad4"], index=0)
    exhaust   = st.slider("Exhaustiveness", 1, 32, 8)
    n_poses   = st.slider("Number of poses", 1, 20, 9)
    cutoff_Å  = st.slider("Contact cutoff (Å)", 2.5, 6.0, 4.5, 0.5)

    st.divider()
    run_btn = st.button("🚀  Run Docking", use_container_width=True, type="primary",
                        disabled=(rec_file is None or
                                  (smiles_in is None and lig_file is None)))
    if st.session_state.poses_pdbqt:
        reset_btn = st.button("🔄  Reset", use_container_width=True)
        if reset_btn:
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
           font-weight:600;color:#c9d1d9;margin:0 0 4px">
  MasterDock <span style="font-size:1rem;color:#58a6ff;font-weight:400">
  · Advanced Docking Platform</span>
</h1>
<p style="color:#8b949e;font-size:.87rem;margin:0 0 8px">
  AutoDock Vina · 3D visualization · Binding analysis · Drug-likeness · Pose comparison
</p>
""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  Load & prepare structures
# ─────────────────────────────────────────────────────────────────────────────
receptor_content = None
ligand_mol       = None
lig_smiles_clean = None

if rec_file:
    receptor_content = rec_file.read().decode("utf-8")
    st.session_state.rec_content = receptor_content

if smiles_in and smiles_in.strip():
    mol = Chem.MolFromSmiles(smiles_in.strip())
    if mol:
        ligand_mol = mol
        lig_smiles_clean = Chem.MolToSmiles(mol)
        st.session_state.lig_smiles = lig_smiles_clean
elif lig_file:
    sdf_txt = lig_file.read().decode("utf-8")
    mols = parse_sdf(sdf_txt)
    if mols:
        ligand_mol = mols[0]
        lig_smiles_clean = Chem.MolToSmiles(ligand_mol)
        st.session_state.lig_sdf = sdf_txt


# ─────────────────────────────────────────────────────────────────────────────
#  Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Structure Viewer",
    "⚗️  Docking & Results",
    "📊 Analysis",
    "💊 Drug-likeness",
])


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Structure Viewer
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_r, col_l = st.columns(2)

    with col_r:
        st.markdown('<div class="sec"><span class="icon">🔵</span> Receptor</div>',
                    unsafe_allow_html=True)
        if receptor_content:
            try:
                stats = parse_pdb_stats(receptor_content)
                st.markdown(f"""
                <div class="metrics-row">
                  <div class="metric-card"><div class="val">{stats['Chains']}</div>
                    <div class="lbl">Chains</div></div>
                  <div class="metric-card"><div class="val">{stats['Residues']}</div>
                    <div class="lbl">Residues</div></div>
                  <div class="metric-card"><div class="val">{stats['Atoms']}</div>
                    <div class="lbl">Atoms</div></div>
                </div>""", unsafe_allow_html=True)

                view_mode = st.radio("Style", ["Cartoon", "Surface", "Stick"],
                                     horizontal=True, key="rec_style")
                style_map = {"Cartoon": "cartoon", "Surface": "surface",
                             "Stick": "stick"}

                # Show box if auto-center
                if auto_center:
                    cx2, cy2, cz2 = pdb_centroid(receptor_content)
                    html = viewer_with_box(receptor_content,
                                          cx2, cy2, cz2, sx, sy, sz, 420)
                else:
                    html = viewer_html([{
                        "data": receptor_content, "fmt": "pdb",
                        "style": style_map[view_mode]
                    }], 420)
                components.html(html, height=440)

                if auto_center:
                    cx2, cy2, cz2 = pdb_centroid(receptor_content)
                    st.markdown(
                        f'<div class="info-box">Auto center: '
                        f'<code>({cx2}, {cy2}, {cz2})</code> Å</div>',
                        unsafe_allow_html=True)
            except PDBConstructionException as e:
                st.error(f"PDB parse error: {e}")
        else:
            st.markdown('<div class="info-box">Upload a receptor PDB file in the sidebar.</div>',
                        unsafe_allow_html=True)

    with col_l:
        st.markdown('<div class="sec"><span class="icon">🟢</span> Ligand</div>',
                    unsafe_allow_html=True)
        if ligand_mol:
            props = mol_props(ligand_mol)
            st.markdown(f"""
            <div class="metrics-row">
              <div class="metric-card"><div class="val">{props['MW']}</div>
                <div class="lbl">MW (Da)</div></div>
              <div class="metric-card"><div class="val">{props['LogP']}</div>
                <div class="lbl">LogP</div></div>
              <div class="metric-card"><div class="val">{props['HBD']}/{props['HBA']}</div>
                <div class="lbl">HBD/HBA</div></div>
              <div class="metric-card"><div class="val">{props['TPSA']}</div>
                <div class="lbl">TPSA</div></div>
            </div>""", unsafe_allow_html=True)

            # 2D structure
            svg = mol_2d_svg(ligand_mol, (360, 260))
            if svg:
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #30363d;'
                    f'border-radius:8px;padding:8px;text-align:center">'
                    f'{svg}</div>',
                    unsafe_allow_html=True)

            # SMILES display
            smiles_display = lig_smiles_clean or ""
            st.markdown(
                f'<div class="info-box" style="word-break:break-all">'
                f'<b>SMILES:</b> <code style="font-size:.78rem">'
                f'{smiles_display}</code></div>',
                unsafe_allow_html=True)

            # Lipinski quick check
            rules = lipinski_check(props)
            passed = sum(1 for _, ok in rules if ok)
            color  = "#3fb950" if passed == 5 else "#e3b341" if passed >= 3 else "#f85149"
            st.markdown(
                f'<div style="margin-top:8px"><b>Lipinski Ro5:</b> '
                f'<span style="color:{color};font-weight:600">'
                f'{passed}/5 rules satisfied</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Enter a SMILES or upload SDF in the sidebar.</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Docking & Results
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Run docking
    if run_btn:
        if not receptor_content:
            st.error("Please upload a receptor PDB file.")
        elif not ligand_mol:
            st.error("Please provide a ligand (SMILES or SDF).")
        else:
            with st.spinner("Preparing receptor…"):
                rec_pdbqt, rec_err = prepare_receptor_pdbqt(receptor_content)
            if not rec_pdbqt:
                st.error(f"Receptor preparation failed: {rec_err}")
                st.info("Make sure `openbabel` is in your `packages.txt`.")
                st.stop()

            with st.spinner("Preparing ligand…"):
                if st.session_state.lig_sdf:
                    lig_pdbqt, lig_err = sdf_to_pdbqt(st.session_state.lig_sdf)
                else:
                    lig_pdbqt, lig_err = smiles_to_pdbqt(lig_smiles_clean)
            if not lig_pdbqt:
                st.error(f"Ligand preparation failed: {lig_err}")
                st.stop()

            if auto_center:
                cx_use, cy_use, cz_use = pdb_centroid(receptor_content)
            else:
                cx_use, cy_use, cz_use = cx, cy, cz

            with st.spinner(f"Running AutoDock Vina (exhaustiveness={exhaust})…"):
                ok, poses_pdbqt, err = run_vina(
                    rec_pdbqt, lig_pdbqt,
                    center=(cx_use, cy_use, cz_use),
                    box_size=(sx, sy, sz),
                    exhaustiveness=exhaust,
                    n_poses=n_poses,
                    scoring=scoring,
                )

            if ok:
                st.session_state.poses_pdbqt = poses_pdbqt
                st.session_state.rec_pdbqt   = rec_pdbqt
                st.session_state.lig_pdbqt   = lig_pdbqt
                scores_df = parse_vina_poses(poses_pdbqt)
                st.session_state.scores_df   = scores_df
                st.success("✅ Docking complete!")
            else:
                st.error(f"Docking failed: {err}")

    # ── Results display
    if st.session_state.poses_pdbqt and st.session_state.scores_df is not None:
        scores_df = st.session_state.scores_df
        poses     = split_pdbqt_poses(st.session_state.poses_pdbqt)

        # Best result banner
        if not scores_df.empty:
            best_aff = scores_df.iloc[0]["Affinity (kcal/mol)"]
            badge = "badge-green" if best_aff < -8 else "badge-blue" if best_aff < -6 else "badge-warn"
            st.markdown(
                f'<div class="ok-box">🎯 Best affinity: '
                f'<span class="badge {badge}">{best_aff:.2f} kcal/mol</span> '
                f'(Pose 1) &nbsp;·&nbsp; {len(poses)} poses generated</div>',
                unsafe_allow_html=True)

        col_v, col_t = st.columns([3, 2])

        with col_t:
            st.markdown('<div class="sec"><span class="icon">📋</span> Pose Scores</div>',
                        unsafe_allow_html=True)
            # Pose selector
            pose_labels = [f"Pose {r['Pose']}  ({r['Affinity (kcal/mol)']:.2f} kcal/mol)"
                           for _, r in scores_df.iterrows()]
            sel_idx = st.selectbox("Select pose to view", range(len(pose_labels)),
                                   format_func=lambda i: pose_labels[i],
                                   key="pose_sel")
            st.session_state.selected_pose = sel_idx

            # Score table
            rows_html = ""
            for _, row in scores_df.iterrows():
                cls = ' class="best"' if row["Pose"] == 1 else ""
                rows_html += (f"<tr{cls}>"
                              f"<td>{int(row['Pose'])}</td>"
                              f"<td>{row['Affinity (kcal/mol)']:.2f}</td>"
                              f"<td>{row['RMSD lb (Å)']:.2f}</td>"
                              f"<td>{row['RMSD ub (Å)']:.2f}</td></tr>")
            st.markdown(f"""
            <table class="pose-table">
              <thead><tr>
                <th>Pose</th><th>Affinity (kcal/mol)</th>
                <th>RMSD lb</th><th>RMSD ub</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

            # Download
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download all poses (.pdbqt)",
                data=st.session_state.poses_pdbqt.encode(),
                file_name="docking_poses.pdbqt",
                mime="chemical/x-pdbqt",
                use_container_width=True,
            )

        with col_v:
            st.markdown('<div class="sec"><span class="icon">🔬</span> 3D Viewer</div>',
                        unsafe_allow_html=True)
            sel_pose_pdbqt = poses[sel_idx] if sel_idx < len(poses) else poses[0]
            sel_pose_pdb   = pdbqt_to_pdb(sel_pose_pdbqt)
            rec_pdb        = st.session_state.rec_content or ""

            show_receptor = st.checkbox("Show receptor", value=True, key="show_rec")
            show_surface  = st.checkbox("Surface (receptor)", value=False, key="show_surf")

            mols = []
            if show_receptor and rec_pdb:
                style = "surface" if show_surface else "cartoon"
                mols.append({"data": rec_pdb, "fmt": "pdb", "style": style})
            mols.append({"data": sel_pose_pdb, "fmt": "pdb",
                         "style": "stick", "color": "#3fb950"})

            components.html(viewer_html(mols, 460), height=480)

    elif not run_btn:
        st.markdown("""
        <div class="info-box" style="margin-top:30px;padding:20px 24px">
          <b>How to use MasterDock:</b><br><br>
          1. Upload a <b>Receptor PDB</b> in the sidebar<br>
          2. Enter a <b>SMILES string</b> or upload an <b>SDF file</b> for the ligand<br>
          3. Adjust the grid box (or use Auto-center)<br>
          4. Set docking parameters (exhaustiveness, poses, scoring function)<br>
          5. Click <b>🚀 Run Docking</b><br><br>
          Docking runs locally using <b>AutoDock Vina</b> — no external API needed.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.scores_df is not None and not st.session_state.scores_df.empty:
        scores_df = st.session_state.scores_df
        poses     = split_pdbqt_poses(st.session_state.poses_pdbqt)
        sel_idx   = st.session_state.selected_pose

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="sec"><span class="icon">📈</span> Affinity by Pose</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(affinity_bar_chart(scores_df),
                            use_container_width=True)

            st.markdown('<div class="sec"><span class="icon">📐</span> RMSD Comparison</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(rmsd_plot(scores_df), use_container_width=True)

        with col2:
            st.markdown('<div class="sec"><span class="icon">🤝</span> Binding Contacts</div>',
                        unsafe_allow_html=True)
            rec_pdb   = st.session_state.rec_content or ""
            sel_pdbqt = poses[sel_idx] if sel_idx < len(poses) else poses[0]

            with st.spinner("Analysing contacts…"):
                contacts_df = find_contacts(rec_pdb, sel_pdbqt, cutoff=cutoff_Å)

            if not contacts_df.empty:
                cfig = contacts_chart(contacts_df)
                if cfig:
                    st.plotly_chart(cfig, use_container_width=True)

                st.markdown('<div class="sec"><span class="icon">📋</span> Contact Residues</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    contacts_df[["Residue", "Res name", "Min dist (Å)", "Nearest atom"]],
                    use_container_width=True, height=220,
                )
                st.download_button(
                    "⬇️ Download contacts (.csv)",
                    data=contacts_df.to_csv(index=False).encode(),
                    file_name="contacts.csv", mime="text/csv",
                )
            else:
                st.markdown(
                    f'<div class="warn-box">No contacts found within {cutoff_Å} Å. '
                    f'Try increasing the contact cutoff in the sidebar.</div>',
                    unsafe_allow_html=True)

        # ── All poses overlay viewer
        st.markdown('<div class="sec"><span class="icon">🔬</span> All Poses Overlay</div>',
                    unsafe_allow_html=True)
        palette = ["#3fb950", "#58a6ff", "#e3b341", "#f85149",
                   "#bc8cff", "#ff9966", "#79c0ff", "#ffa198",
                   "#56d364", "#d2a8ff"]
        mols_overlay = []
        if st.session_state.rec_content:
            mols_overlay.append({
                "data": st.session_state.rec_content, "fmt": "pdb", "style": "cartoon"
            })
        for i, p in enumerate(poses):
            mols_overlay.append({
                "data": pdbqt_to_pdb(p), "fmt": "pdb",
                "style": "stick", "color": palette[i % len(palette)]
            })
        components.html(viewer_html(mols_overlay, 440), height=460)

    else:
        st.markdown('<div class="info-box">Run a docking calculation first to see analysis.</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — Drug-likeness
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    if ligand_mol:
        props = mol_props(ligand_mol)
        rules = lipinski_check(props)

        col_r, col_p = st.columns([2, 3])

        with col_r:
            st.markdown('<div class="sec"><span class="icon">🧪</span> Molecular Properties</div>',
                        unsafe_allow_html=True)
            prop_items = [
                ("Molecular Weight", f"{props['MW']} Da"),
                ("LogP (Wildman-Crippen)", str(props["LogP"])),
                ("H-Bond Donors", str(props["HBD"])),
                ("H-Bond Acceptors", str(props["HBA"])),
                ("TPSA", f"{props['TPSA']} Å²"),
                ("Rotatable Bonds", str(props["RotBonds"])),
                ("Ring Count", str(props["Rings"])),
                ("Heavy Atom Count", str(props["HeavyAtoms"])),
            ]
            rows = "".join(
                f"<tr><td style='color:#8b949e;padding:6px 10px'>{k}</td>"
                f"<td style='font-family:var(--mono);padding:6px 10px'>{v}</td></tr>"
                for k, v in prop_items
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse'>"
                f"<tbody>{rows}</tbody></table>",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="sec" style="margin-top:20px"><span class="icon">✅</span> Lipinski Ro5</div>',
                        unsafe_allow_html=True)
            for rule, ok in rules:
                icon  = "✅" if ok else "❌"
                color = "#3fb950" if ok else "#f85149"
                st.markdown(
                    f'<div style="padding:5px 0;color:{color};font-size:.88rem">'
                    f'{icon} {rule}</div>',
                    unsafe_allow_html=True)

            passed = sum(1 for _, ok in rules if ok)
            verdict = ("✅ Drug-like" if passed == 5
                       else "⚠️ Borderline" if passed >= 3
                       else "❌ Unlikely drug-like")
            vcolor  = "#3fb950" if passed == 5 else "#e3b341" if passed >= 3 else "#f85149"
            st.markdown(
                f'<div style="margin-top:12px;font-size:1rem;font-weight:600;'
                f'color:{vcolor}">{verdict} ({passed}/5)</div>',
                unsafe_allow_html=True)

        with col_p:
            st.markdown('<div class="sec"><span class="icon">🕸️</span> Property Radar</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(radar_chart(props), use_container_width=True)

            # 2D structure again
            st.markdown('<div class="sec"><span class="icon">🖼️</span> 2D Structure</div>',
                        unsafe_allow_html=True)
            svg = mol_2d_svg(ligand_mol, (400, 280))
            if svg:
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #30363d;'
                    f'border-radius:8px;padding:12px;text-align:center">{svg}</div>',
                    unsafe_allow_html=True)

            # SMILES + formula
            formula = rdMolDescriptors.CalcMolFormula(ligand_mol)
            st.markdown(
                f'<div class="info-box" style="margin-top:10px">'
                f'<b>Formula:</b> <code>{formula}</code> &nbsp;·&nbsp; '
                f'<b>SMILES:</b> <code style="font-size:.78rem;word-break:break-all">'
                f'{lig_smiles_clean}</code></div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Provide a ligand in the sidebar to see drug-likeness analysis.</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#8b949e;font-size:.78rem">'
    'MasterDock · AutoDock Vina · 3Dmol.js · RDKit · Meeko · Biopython'
    '</div>',
    unsafe_allow_html=True,
)
