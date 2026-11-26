import pytest

from mdgraph.config import Structure
from mdgraph.structure_tools import available_builders, build_crystal


def test_auto_offers_atomsk_only_for_supported_cells(monkeypatch):
    monkeypatch.setattr("mdgraph.structure_tools.shutil.which", lambda path: "/opt/atomsk")
    fcc = Structure(elements=["Cu"], builder="auto", crystal={"lattice": "fcc", "a_angstrom": 3.5})
    assert available_builders(fcc) == ["ase", "atomsk"]
    assert available_builders(Structure(elements=["Cu"], builder="auto", path="original.data")) == [
        "ase"
    ]
    monkeypatch.setattr("mdgraph.structure_tools.shutil.which", lambda path: None)
    assert available_builders(fcc) == ["ase"]
    with pytest.raises(ValueError, match="not found"):
        available_builders(fcc.model_copy(update={"builder": "atomsk"}))


def test_ase_hcp_preserves_primitive_cell_and_repeat(tmp_path):
    structure = Structure(
        elements=["Ti"],
        crystal={"lattice": "hcp", "a_angstrom": 2.95, "c_angstrom": 4.68, "repeat": [2, 2, 2]},
    )
    atoms, tool = build_crystal(structure, tmp_path, "ase")
    assert len(atoms) == 16
    assert atoms.cell.cellpar() == pytest.approx([5.9, 5.9, 9.36, 90, 90, 120])
    assert tool["name"].startswith("ASE ")
