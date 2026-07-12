import math

import pytest
from pymatgen.core import Lattice, Structure

from crystalpath.analysis.interstitials import (
    InterstitialAnalyzer,
    fractional_position_category,
    supercell_translations,
)


@pytest.fixture
def fcc_structure():
    radius = 1.0
    lattice_parameter = 2.0 * math.sqrt(2.0) * radius
    structure = Structure(
        Lattice.cubic(lattice_parameter),
        ["Cu"] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    return structure, radius


def test_fcc_finds_tetrahedral_and_octahedral_sites(fcc_structure):
    structure, radius = fcc_structure
    result = InterstitialAnalyzer().analyze(structure, radii={"Cu": radius})
    tetra = result.sites_of_kind("四面体型")
    octa = result.sites_of_kind("八面体型")
    assert len(tetra) == 8
    assert len(octa) == 4
    assert all(site.coordination_number == 4 for site in tetra)
    assert all(len(site.polyhedron_faces) == 4 for site in tetra)
    assert all(site.coordination_number == 6 for site in octa)
    assert all(len(site.polyhedron_faces) == 8 for site in octa)


def test_fcc_free_radii_match_ideal_values(fcc_structure):
    structure, radius = fcc_structure
    result = InterstitialAnalyzer().analyze(structure, radii={"Cu": radius})
    tetra = result.sites_of_kind("四面体型")
    octa = result.sites_of_kind("八面体型")
    assert tetra[0].free_radius == pytest.approx((math.sqrt(6) / 2 - 1) * radius)
    assert octa[0].free_radius == pytest.approx((math.sqrt(2) - 1) * radius)


def test_fcc_expected_representative_coordinates(fcc_structure):
    structure, radius = fcc_structure
    result = InterstitialAnalyzer().analyze(structure, radii={"Cu": radius})
    tetra_coords = {tuple(round(value, 6) for value in site.frac_coords) for site in result.sites_of_kind("四面体型")}
    octa_coords = {tuple(round(value, 6) for value in site.frac_coords) for site in result.sites_of_kind("八面体型")}
    assert (0.25, 0.25, 0.25) in tetra_coords
    assert (0.5, 0.5, 0.5) in octa_coords
    assert (0.5, 0.0, 0.0) in octa_coords


def test_single_atom_simple_cubic_cell_is_supported():
    structure = Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]])
    result = InterstitialAnalyzer().analyze(structure, radii={"Fe": 1.0})
    cubic = result.sites_of_kind("立方型")
    assert len(cubic) == 1
    assert cubic[0].coordination_number == 8
    assert cubic[0].frac_coords == pytest.approx((0.5, 0.5, 0.5))


def test_two_atom_bcc_cell_is_supported():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    result = InterstitialAnalyzer(minimum_free_radius=0.0).analyze(
        structure,
        radii={"Fe": 1.0},
    )
    tetra = result.sites_of_kind("四面体型")
    octa = result.sites_of_kind("八面体型")
    assert len(tetra) == 12
    assert len(octa) == 6


def test_bcc_expected_representative_coordinates_and_distance_shells():
    lattice_parameter = 4.0
    structure = Structure(
        Lattice.cubic(lattice_parameter),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    result = InterstitialAnalyzer(minimum_free_radius=0.0).analyze(
        structure,
        radii={"Fe": 1.0},
    )
    tetra = result.sites_of_kind("四面体型")
    octa = result.sites_of_kind("八面体型")
    tetra_coords = {
        tuple(round(value, 6) for value in site.frac_coords) for site in tetra
    }
    octa_coords = {
        tuple(round(value, 6) for value in site.frac_coords) for site in octa
    }
    assert (0.5, 0.0, 0.25) in tetra_coords
    assert (0.5, 0.0, 0.0) in octa_coords
    assert tetra_coords == {
        (0.0, 0.25, 0.5),
        (0.0, 0.5, 0.25),
        (0.0, 0.5, 0.75),
        (0.0, 0.75, 0.5),
        (0.25, 0.0, 0.5),
        (0.25, 0.5, 0.0),
        (0.5, 0.0, 0.25),
        (0.5, 0.0, 0.75),
        (0.5, 0.25, 0.0),
        (0.5, 0.75, 0.0),
        (0.75, 0.0, 0.5),
        (0.75, 0.5, 0.0),
    }
    assert octa_coords == {
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    }
    assert [fractional_position_category(site.frac_coords) for site in octa].count(
        "晶胞面上"
    ) == 3
    assert [fractional_position_category(site.frac_coords) for site in octa].count(
        "晶胞棱上"
    ) == 3
    assert all(
        fractional_position_category(site.frac_coords) == "晶胞面上"
        for site in tetra
    )

    boundary_octa = next(
        site
        for site in octa
        if tuple(round(value, 6) for value in site.frac_coords) == (0.5, 0.0, 0.0)
    )
    distances = sorted(neighbor.distance for neighbor in boundary_octa.neighbors)
    assert distances[:2] == pytest.approx([lattice_parameter / 2.0] * 2)
    assert distances[2:] == pytest.approx([lattice_parameter / math.sqrt(2.0)] * 4)
    assert len(boundary_octa.polyhedron_faces) == 8


def test_bcc_boundary_octahedron_uses_periodic_neighbor_images():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    result = InterstitialAnalyzer(minimum_free_radius=0.0).analyze(
        structure,
        radii={"Fe": 1.0},
    )
    boundary_octa = next(
        site
        for site in result.sites_of_kind("八面体型")
        if tuple(round(value, 6) for value in site.frac_coords) == (0.5, 0.0, 0.0)
    )
    assert any(neighbor.offset != (0, 0, 0) for neighbor in boundary_octa.neighbors)
    ys = [point[1] for point in boundary_octa.polyhedron_points]
    zs = [point[2] for point in boundary_octa.polyhedron_points]
    assert min(ys) < 0.0 < max(ys)
    assert min(zs) < 0.0 < max(zs)


def test_supercell_translations_cover_every_display_cell():
    lattice = Lattice.orthorhombic(2.0, 3.0, 4.0)
    translations = supercell_translations(lattice.matrix, (2, 1, 3))
    assert len(translations) == 6
    assert (0.0, 0.0, 0.0) in translations
    assert (2.0, 0.0, 8.0) in translations
