from pathlib import Path

import pytest

from mdgraph.analysis import analyze
from mdgraph.config import load_config
from mdgraph.potentials import select_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cu_eam_lattice.yaml"


def test_static_catalog_requires_pressure_to_be_in_range():
    cfg = load_config(EXAMPLE)
    catalog = Path(cfg.potential.catalog)
    assert select_entry(cfg, catalog)[0].id == "mishin_2001_cu_eam1_lammps"
    cfg.protocol.pressure_bar = 1
    with pytest.raises(ValueError, match="compatible IDs: \\[\\]"):
        select_entry(cfg, catalog)


@pytest.mark.parametrize(
    "shear,force,marker,issue",
    [
        (0, 1e-12, "MDGRAPH_COMPLETED", None),
        (0.11, 1e-12, "MDGRAPH_COMPLETED", "stress"),
        (0, 1e-4, "MDGRAPH_COMPLETED", "force"),
        (0, 1e-12, 'print "MDGRAPH_COMPLETED"', "completion marker"),
    ],
)
def test_relaxation_uses_final_box_normalized_by_repeat(tmp_path, shear, force, marker, issue):
    cfg = load_config(EXAMPLE)
    cfg.structure.crystal.repeat = (3, 4, 5)
    # Two valid artifacts and a static thermo snapshot; no time trajectory is expected.
    for name in ("final.data", "final.restart"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    (tmp_path / "log.lammps").write_text(
        "Step PotEng Press Fmax\n0 -200 300 0\nLoop time\n"
        "Step Atoms PotEng Press Pxx Pyy Pzz Pxy Pxz Pyz Lx Ly Lz Fmax\n"
        f"34 240 -840 0 0 0 0 {shear} 0 0 10.845 14.46 18.075 {force}\n"
        f"Loop time\n{marker}\n",
        encoding="utf-8",
    )
    result = analyze(cfg, tmp_path, 240)
    if issue:
        assert not result["passed"]
        assert any(issue in detail for detail in result["issues"])
    else:
        assert result["passed"], result
        assert result["metrics"]["lattice_constant_angstrom"] == pytest.approx(3.615)
        assert result["metrics"]["potential_energy_ev_per_atom"] == pytest.approx(-3.5)
        assert (tmp_path / "relaxation.csv").is_file()
