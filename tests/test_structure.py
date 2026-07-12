from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from crystalpath.core.structure import CrystalDocument, site_element, site_occupancy
from crystalpath.rendering.geometry import RenderAtom, build_coordination_scene


SIMPLE_CIF = """data_NaCl
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0 0 0 1
Cl1 Cl 0.5 0.5 0.5 1
"""


@pytest.fixture
def document(tmp_path: Path) -> CrystalDocument:
    cif = tmp_path / "simple.cif"
    cif.write_text(SIMPLE_CIF, encoding="utf-8")
    return CrystalDocument.from_cif(cif)


def test_cif_loads_and_summarizes(document: CrystalDocument):
    summary = document.summary()
    assert summary.site_count == 2
    assert summary.is_ordered
    assert summary.a == pytest.approx(5.64)
    assert {site_element(site) for site in document.original} == {"Na", "Cl"}


def test_supercell_is_rebuilt_from_original(document: CrystalDocument):
    document.set_supercell(2, 1, 3)
    assert len(document.display) == 12
    assert document.supercell == (2, 1, 3)
    document.set_supercell(1, 1, 1)
    assert len(document.display) == 2


def test_occupancy_is_retained(tmp_path: Path):
    cif = tmp_path / "partial.cif"
    cif.write_text(SIMPLE_CIF.replace("Na1 Na 0 0 0 1", "Na1 Na 0 0 0 0.5"), encoding="utf-8")
    document = CrystalDocument.from_cif(cif)
    sodium = next(site for site in document.original if site_element(site) == "Na")
    assert site_occupancy(sodium) == pytest.approx(0.5)
    assert not document.original.is_ordered


def test_coordination_scene_only_connects_requested_pair():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Nb", "O", "Nb"],
        [[0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5]],
    )
    atoms, bonds = build_coordination_scene(structure, max_distance=1.20)
    assert len(bonds) == 2
    assert all(site_element(structure[bond.first.site_index]) == "Nb" for bond in bonds)
    assert all(site_element(structure[bond.second.site_index]) == "O" for bond in bonds)


def test_external_ligand_image_completes_center_coordination():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Nb", "O"],
        [[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]],
    )
    atoms, bonds = build_coordination_scene(
        structure,
        max_distance=0.50,
        include_external_ligands=True,
    )
    assert len(bonds) == 1
    assert RenderAtom(1, (1, 0, 0)) in atoms


def test_center_on_zero_face_is_repeated_on_one_face():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Nb", "O"],
        [[0, 0.5, 0.5], [0.1, 0.5, 0.5]],
    )
    atoms, _bonds = build_coordination_scene(
        structure,
        max_distance=0.50,
        include_external_ligands=True,
    )
    assert RenderAtom(0, (0, 0, 0)) in atoms
    assert RenderAtom(0, (1, 0, 0)) in atoms
