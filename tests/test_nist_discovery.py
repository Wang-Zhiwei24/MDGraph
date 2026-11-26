import pytest

from mdgraph.nist_discovery import checked_url, crystal_evidence, index_candidates, validate_setfl


def test_index_ignores_superseded_and_non_nist_links():
    row = """<li><b>LAMMPS pair_style eam/alloy</b>
    <a href="../../entry/example/Al--LAMMPS--ipr1.html">Details</a>
    <div><b>File(s):</b><a href="../../Download/example/Al.set">Al.set</a></div></li>"""
    old = row.replace("ipr1", "ipr2").replace("File(s):", "File(s): superseded")
    foreign = row.replace("ipr1", "ipr3").replace(
        "../../Download/example/Al.set", "https://other.test/model.set"
    )
    found = index_candidates(
        "https://www.ctcms.nist.gov/potentials/system/Al/", (row + old + foreign).encode()
    )
    assert [r["id"] for r in found] == ["Al--LAMMPS--ipr1"]
    with pytest.raises(ValueError):
        checked_url("https://www.ctcms.nist.gov/potentials/system/../foo")


def test_crystal_evidence_and_download_are_distinct_checks():
    values = b"prototype,transformed,method,a,b,c,alpha,beta,gamma\nA1--Cu--fcc,False,static,4,4,4,90,90,90\n"
    assert crystal_evidence(values, "fcc")["lattice_angstrom"] == [4, 4, 4]
    assert crystal_evidence(values, "bcc") is None
    assert crystal_evidence(values.replace(b"False", b"True"), "fcc") is None
    table = b"comment\ncomment\ncomment\n1 Al\n4 .1 4 .1 .3\n13 26.98 4.05 fcc\n" + b"0 " * 12
    validate_setfl(table, "Al")
    for invalid in [table[:-2], table.replace(b"1 Al", b"1 Cu"), table[:-2] + b"nan"]:
        with pytest.raises(ValueError, match="valid single-element"):
            validate_setfl(invalid, "Al")
