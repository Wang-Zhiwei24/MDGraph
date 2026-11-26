import hashlib
from pathlib import Path

import pytest
import yaml

from mdgraph.config import TaskConfig, load_config
from mdgraph.graph import potential_plan
from mdgraph.potentials import select_entry


def test_general_catalog_infers_phase_and_property_and_has_verified_al_candidates():
    examples = Path(__file__).resolve().parents[1] / "examples"
    cfg = load_config(examples / "new_al_fcc.yaml")
    catalog = Path(cfg.potential.catalog)
    for candidate in yaml.safe_load(catalog.read_text())["entries"]:
        assert (
            hashlib.sha256((catalog.parent / candidate["model"]).read_bytes()).hexdigest()
            == candidate["model_sha256"]
        )
    entry, evidence = select_entry(
        cfg, catalog, lambda xs: {"choice": xs[1]["id"]}, element_counts={"Al": 256}
    )
    assert entry.id == "al_jacobsen_1987"
    assert evidence["compatible_ids"] == ["al_mishin_1999", "al_jacobsen_1987"]
    assert evidence["scope"]["phase"] == "fcc"
    assert evidence["scope"]["target_property"] == "lattice_constant"
    plan = potential_plan(cfg)
    assert plan["mode"] == "materials catalog" and "application" not in plan
    cfg.structure.crystal.lattice = "bcc"
    with pytest.raises(ValueError, match="No compatible catalog potential"):
        select_entry(
            cfg, catalog, lambda _: pytest.fail("incompatible candidates must not reach the model")
        )


def test_general_catalog_requires_alloy_composition_to_match(tmp_path):
    entry = {
        "id": "alloy",
        "kind": "eam/alloy",
        "elements": ["Cu", "Ni"],
        "units": "metal",
        "phases": ["fcc"],
        "target_properties": ["lattice_constant"],
        "atom_fraction_ranges": {"Cu": [0.4, 0.6], "Ni": [0.4, 0.6]},
        "source": "test fixture",
        "model": "binary.eam.alloy",
        "model_sha256": "0" * 64,
        "temperature_k": [0, 0],
        "pressure_bar": [0, 0],
    }
    legacy = {**entry, "id": "legacy", "application": "replay"}
    catalog = tmp_path / "materials.yaml"
    catalog.write_text(yaml.safe_dump({"entries": [entry, legacy]}), encoding="utf-8")
    cfg = TaskConfig.model_validate(
        {
            "name": "new_alloy",
            "structure": {"path": "input.data", "elements": ["Cu", "Ni"]},
            "potential": {"kind": "eam/alloy", "catalog": str(catalog), "phase": "fcc"},
            "protocol": {"ensemble": "relax", "temperature_k": 0},
        }
    )
    selected, evidence = select_entry(cfg, catalog, element_counts={"Cu": 20, "Ni": 20})
    assert selected.id == "alloy"
    assert evidence["scope"]["atom_fractions"] == {"Cu": 0.5, "Ni": 0.5}
    with pytest.raises(ValueError, match="No compatible catalog potential"):
        select_entry(cfg, catalog, element_counts={"Cu": 31, "Ni": 9})
