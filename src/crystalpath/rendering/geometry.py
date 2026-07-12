from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from crystalpath.core.structure import site_element


@dataclass(frozen=True, order=True)
class RenderAtom:
    site_index: int
    offset: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class RenderBond:
    first: RenderAtom
    second: RenderAtom
    distance: float


def atom_fractional_coords(structure, atom: RenderAtom) -> np.ndarray:
    return np.asarray(structure[atom.site_index].frac_coords) + np.asarray(atom.offset)


def atom_cartesian_coords(structure, atom: RenderAtom) -> np.ndarray:
    return atom_fractional_coords(structure, atom) @ np.asarray(structure.lattice.matrix)


def _inside_cell(frac: np.ndarray, tolerance: float = 1e-7) -> bool:
    return bool(np.all(frac >= -tolerance) and np.all(frac <= 1.0 + tolerance))


def build_coordination_scene(
    structure,
    center_element: str = "Nb",
    ligand_element: str = "O",
    max_distance: float = 2.40,
    include_external_ligands: bool = True,
):
    """Build a center-driven coordination scene similar to VESTA.

    The chemical count uses center atoms in [0, 1), while the visual scene uses
    the closed [0, 1] cell. Center sites on a zero face are therefore repeated
    on the equivalent one face. Only ligand neighbors within ``max_distance``
    are added, and their periodic images may lie outside the geometric cell so
    every displayed center keeps its coordination shell. No center-center or
    ligand-ligand bonds are made.
    """

    base_centers = {
        RenderAtom(index)
        for index, site in enumerate(structure)
        if site_element(site) == center_element
    }
    centers_to_render = set(base_centers)
    if include_external_ligands:
        for center in base_centers:
            frac = np.asarray(structure[center.site_index].frac_coords)
            for offset in product((-1, 0, 1), repeat=3):
                shifted = frac + np.asarray(offset)
                if _inside_cell(shifted):
                    centers_to_render.add(
                        RenderAtom(center.site_index, tuple(int(value) for value in offset))
                    )
    atoms = set(centers_to_render)
    bonds: dict[tuple[RenderAtom, RenderAtom], RenderBond] = {}
    centers, neighbors, offsets, distances = structure.get_neighbor_list(max_distance)
    neighbor_map: dict[int, list[tuple[int, tuple[int, int, int], float]]] = {}
    for first_index, second_index, offset, distance in zip(
        centers, neighbors, offsets, distances
    ):
        first_index = int(first_index)
        second_index = int(second_index)
        if site_element(structure[first_index]) != center_element:
            continue
        if site_element(structure[second_index]) != ligand_element:
            continue
        neighbor_map.setdefault(first_index, []).append(
            (
                second_index,
                tuple(int(value) for value in offset),
                float(distance),
            )
        )
    for center in centers_to_render:
        for second_index, relative_offset, distance in neighbor_map.get(
            center.site_index, []
        ):
            ligand_offset = tuple(
                center.offset[axis] + relative_offset[axis] for axis in range(3)
            )
            ligand = RenderAtom(second_index, ligand_offset)
            if not include_external_ligands and not _inside_cell(
                atom_fractional_coords(structure, ligand)
            ):
                continue
            endpoints = (center, ligand)
            bonds[endpoints] = RenderBond(center, ligand, distance)
            atoms.add(ligand)
    return sorted(atoms), list(bonds.values())
