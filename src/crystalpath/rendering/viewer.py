from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer
from pyvistaqt import QtInteractor

from crystalpath.analysis.interstitials import InterstitialResult, supercell_translations
from crystalpath.core.structure import CrystalDocument, site_element, site_occupancy
from crystalpath.rendering.geometry import (
    RenderAtom,
    atom_cartesian_coords,
    build_coordination_scene,
)
from crystalpath.rendering.palette import color_for, radius_for


class CrystalViewer(QtInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.document: CrystalDocument | None = None
        self.visible_elements: set[str] = set()
        self.element_radii: dict[str, float] = {}
        self.highlighted_element: str | None = None
        self.show_bonds = True
        self.complete_boundary = True
        self.bond_radius = 0.085
        self.bond_center_element = "Nb"
        self.bond_ligand_element = "O"
        self.bond_max_distance = 2.40
        self.on_atom_picked: Callable[[int, tuple[int, int, int]], None] | None = None
        self.selected_atom: RenderAtom | None = None
        self._home_camera_position = None
        self._home_view_angle: float | None = None
        self._home_parallel_scale: float | None = None

        # Hundreds of atoms are rendered as a few element/occupancy batches.
        self._atom_batches: dict[str, list[RenderAtom]] = {}
        self._scene_cache_key = None
        self._scene_cache = ([], [])
        self._bond_mesh_cache_key = None
        self._bond_mesh_cache: dict[str, pv.PolyData] = {}
        self.interstitial_result: InterstitialResult | None = None
        self.visible_interstitial_kind: str | None = None
        self.visible_interstitial_site_ids: set[int] | None = None
        self.selected_interstitial_id: int | None = None
        self.interstitial_opacity = 0.32
        self.show_interstitial_centers = True
        self.show_interstitial_free_spheres = False
        self._interstitial_poly_cache: dict[tuple, pv.PolyData] = {}

        self.set_background("#111722", top="#263548")
        self.add_axes(interactive=False, line_width=2)

    def set_document(self, document: CrystalDocument) -> None:
        self.document = document
        self.visible_elements = {site_element(site) for site in document.display}
        for element in self.visible_elements:
            self.element_radii.setdefault(element, radius_for(element))
        self.highlighted_element = None
        self.selected_atom = None
        self.clear_interstitials(render=False)
        self._invalidate_scene_cache()
        self.render_structure(reset_camera=True)
        self.set_home_view()

    def set_home_view(self) -> None:
        if self.document is None:
            return
        self.view_isometric()
        self.reset_camera()
        self._home_camera_position = tuple(
            tuple(float(value) for value in part)
            for part in self.camera_position
        )
        self._home_view_angle = float(self.camera.GetViewAngle())
        self._home_parallel_scale = float(self.camera.GetParallelScale())
        self.render()

    def reset_to_home_view(self) -> None:
        if self._home_camera_position is None:
            return
        self.camera_position = self._home_camera_position
        if self._home_view_angle is not None:
            self.camera.SetViewAngle(self._home_view_angle)
        if self._home_parallel_scale is not None:
            self.camera.SetParallelScale(self._home_parallel_scale)
        self.render()

    def zoom_by(self, factor: float) -> None:
        if self.document is None:
            return
        self.camera.zoom(factor)
        self.render()

    def set_interstitial_result(
        self,
        result: InterstitialResult,
        kind: str | None = None,
    ) -> None:
        self.interstitial_result = result
        self.visible_interstitial_kind = kind or (result.kinds[0] if result.kinds else None)
        self.visible_interstitial_site_ids = {
            site.site_id
            for site in result.sites
            if site.kind == self.visible_interstitial_kind
        }
        self.selected_interstitial_id = None
        self._interstitial_poly_cache.clear()
        self.render_structure(reset_camera=True)

    def set_interstitial_kind(self, kind: str) -> None:
        self.visible_interstitial_kind = kind
        if self.interstitial_result is not None:
            self.visible_interstitial_site_ids = {
                site.site_id for site in self.interstitial_result.sites_of_kind(kind)
            }
        self.selected_interstitial_id = None
        self.render_structure(reset_camera=True)

    def set_visible_interstitial_sites(self, kind: str, site_ids) -> None:
        self.visible_interstitial_kind = kind
        self.visible_interstitial_site_ids = {int(site_id) for site_id in site_ids}
        if self.selected_interstitial_id not in self.visible_interstitial_site_ids:
            self.selected_interstitial_id = None
        self.render_structure(reset_camera=False)

    def select_interstitial(self, site_id: int) -> None:
        if self.interstitial_result is None:
            return
        site = next(
            (item for item in self.interstitial_result.sites if item.site_id == site_id),
            None,
        )
        if site is None:
            return
        self.visible_interstitial_kind = site.kind
        self.selected_interstitial_id = site_id
        self.render_structure(reset_camera=False)

    def set_interstitial_opacity(self, opacity: float) -> None:
        self.interstitial_opacity = max(0.05, min(0.95, opacity))
        self.render_structure(reset_camera=False)

    def set_show_interstitial_centers(self, enabled: bool) -> None:
        self.show_interstitial_centers = enabled
        self.render_structure(reset_camera=False)

    def set_show_interstitial_free_spheres(self, enabled: bool) -> None:
        self.show_interstitial_free_spheres = enabled
        self.render_structure(reset_camera=False)

    def clear_interstitials(self, render: bool = True) -> None:
        self.interstitial_result = None
        self.visible_interstitial_kind = None
        self.visible_interstitial_site_ids = None
        self.selected_interstitial_id = None
        self._interstitial_poly_cache.clear()
        if render and self.document is not None:
            self.render_structure(reset_camera=False)

    def set_element_visible(self, element: str, visible: bool) -> None:
        if (element in self.visible_elements) == visible:
            return
        if visible:
            self.visible_elements.add(element)
        else:
            self.visible_elements.discard(element)
        self.render_structure(reset_camera=False)

    def highlight_element(self, element: str | None) -> None:
        self.highlighted_element = element
        self.render_structure(reset_camera=False)

    def set_element_radius(self, element: str, radius: float) -> None:
        self.element_radii[element] = radius
        self.render_structure(reset_camera=False)

    def set_show_bonds(self, enabled: bool) -> None:
        if self.show_bonds == enabled:
            return
        self.show_bonds = enabled
        self.render_structure(reset_camera=False)

    def set_complete_boundary(self, enabled: bool) -> None:
        if self.complete_boundary == enabled:
            return
        self.complete_boundary = enabled
        self._invalidate_scene_cache()
        self.render_structure(reset_camera=True)

    def set_bond_settings(self, center: str, ligand: str, max_distance: float) -> None:
        if (
            self.bond_center_element == center
            and self.bond_ligand_element == ligand
            and abs(self.bond_max_distance - max_distance) < 1e-9
        ):
            return
        self.bond_center_element = center
        self.bond_ligand_element = ligand
        self.bond_max_distance = max_distance
        self._invalidate_scene_cache()
        self.render_structure(reset_camera=False)

    def _invalidate_scene_cache(self) -> None:
        self._scene_cache_key = None
        self._bond_mesh_cache_key = None
        self._bond_mesh_cache.clear()

    def _get_scene(self):
        assert self.document is not None
        key = (
            id(self.document.display),
            self.bond_center_element,
            self.bond_ligand_element,
            round(self.bond_max_distance, 6),
            self.complete_boundary,
        )
        if key != self._scene_cache_key:
            self._scene_cache = build_coordination_scene(
                self.document.display,
                center_element=self.bond_center_element,
                ligand_element=self.bond_ligand_element,
                max_distance=self.bond_max_distance,
                include_external_ligands=self.complete_boundary,
            )
            self._scene_cache_key = key
        return self._scene_cache

    def render_structure(self, reset_camera: bool = False) -> None:
        camera = None if reset_camera else self.camera_position
        self.clear()
        self._atom_batches.clear()
        self.add_axes(interactive=False, line_width=2)
        if self.document is None:
            return

        self._draw_cell()
        atoms, bonds = self._get_scene()
        if self.show_bonds:
            self._draw_bonds(bonds)
        self._draw_atom_batches(atoms)
        self._draw_interstitials()
        self._draw_selected_atom()

        if reset_camera:
            self.reset_camera()
        elif camera is not None:
            self.camera_position = camera
        self._configure_picker()
        self.render()

    def _draw_atom_batches(self, atoms) -> None:
        assert self.document is not None
        structure = self.document.display
        grouped: dict[tuple[str, float], list[RenderAtom]] = {}
        for atom in atoms:
            site = structure[atom.site_index]
            element = site_element(site)
            if element not in self.visible_elements:
                continue
            opacity = max(0.28, min(1.0, site_occupancy(site)))
            grouped.setdefault((element, round(opacity, 3)), []).append(atom)

        for (element, opacity), batch in grouped.items():
            radius = self.element_radii.get(element, radius_for(element))
            element_selected = element == self.highlighted_element
            if element_selected:
                radius *= 1.25
            points = np.asarray(
                [atom_cartesian_coords(structure, atom) for atom in batch],
                dtype=float,
            )
            cloud = pv.PolyData(points)
            cloud["atom_slot"] = np.arange(len(batch), dtype=np.int32)
            sphere = pv.Sphere(
                radius=radius,
                theta_resolution=12,
                phi_resolution=12,
            )
            glyphs = cloud.glyph(geom=sphere, scale=False, orient=False)
            actor = self.add_mesh(
                glyphs,
                color="#FFD54F" if element_selected else color_for(element),
                opacity=opacity,
                smooth_shading=True,
                specular=0.25,
                name=f"atoms-{element}-{opacity:.3f}",
                pickable=True,
            )
            self._atom_batches[self._actor_key(actor)] = batch

    def _draw_selected_atom(self) -> None:
        if self.document is None or self.selected_atom is None:
            return
        site = self.document.display[self.selected_atom.site_index]
        element = site_element(site)
        if element not in self.visible_elements:
            return
        center = atom_cartesian_coords(self.document.display, self.selected_atom)
        radius = self.element_radii.get(element, radius_for(element)) * 1.35
        marker = pv.Sphere(
            radius=radius,
            center=center,
            theta_resolution=14,
            phi_resolution=14,
        )
        self.add_mesh(
            marker,
            color="#FFD54F",
            smooth_shading=True,
            specular=0.3,
            pickable=False,
            name="selected-atom",
        )

    @staticmethod
    def _interstitial_color(kind: str) -> str:
        if kind == "四面体型":
            return "#AB47BC"
        if kind == "八面体型":
            return "#26A69A"
        if kind == "立方型":
            return "#42A5F5"
        if kind.startswith("畸变"):
            return "#FFA726"
        return "#78909C"

    @staticmethod
    def _site_polyhedron(site, translation=None):
        points = np.asarray(site.polyhedron_points, dtype=float)
        if translation is not None:
            points = points + np.asarray(translation, dtype=float)
        faces = np.hstack([[3, *face] for face in site.polyhedron_faces])
        return pv.PolyData(points, faces=faces)

    def _interstitial_instances(self, sites):
        assert self.document is not None
        instances = []
        for translation in supercell_translations(
            self.document.original.lattice.matrix,
            self.document.supercell,
        ):
            instances.extend((site, np.asarray(translation)) for site in sites)
        return instances

    def _draw_interstitials(self) -> None:
        if self.interstitial_result is None or self.visible_interstitial_kind is None:
            return
        kind = self.visible_interstitial_kind
        sites = self.interstitial_result.sites_of_kind(kind)
        if self.visible_interstitial_site_ids is not None:
            sites = tuple(
                site for site in sites if site.site_id in self.visible_interstitial_site_ids
            )
        if not sites:
            return
        instances = self._interstitial_instances(sites)
        cache_key = (
            kind,
            tuple(sorted(site.site_id for site in sites)),
            self.document.supercell,
        )
        if cache_key not in self._interstitial_poly_cache:
            meshes = [
                self._site_polyhedron(site, translation)
                for site, translation in instances
            ]
            self._interstitial_poly_cache[cache_key] = pv.merge(meshes, merge_points=False)
        self.add_mesh(
            self._interstitial_poly_cache[cache_key],
            color=self._interstitial_color(kind),
            opacity=self.interstitial_opacity,
            show_edges=True,
            edge_color="#ECEFF1",
            line_width=1,
            pickable=False,
            name="interstitial-polyhedra",
        )

        centers = np.asarray(
            [np.asarray(site.cart_coords) + translation for site, translation in instances],
            dtype=float,
        )
        if self.show_interstitial_centers:
            cloud = pv.PolyData(centers)
            marker = pv.Sphere(radius=0.13, theta_resolution=10, phi_resolution=10)
            glyphs = cloud.glyph(geom=marker, scale=False, orient=False)
            self.add_mesh(
                glyphs,
                color=self._interstitial_color(kind),
                smooth_shading=True,
                pickable=False,
                name="interstitial-centers",
            )

        if self.show_interstitial_free_spheres:
            positive_instances = [
                (site, translation)
                for site, translation in instances
                if site.free_radius > 0
            ]
            if positive_instances:
                sphere_cloud = pv.PolyData(
                    np.asarray(
                        [
                            np.asarray(site.cart_coords) + translation
                            for site, translation in positive_instances
                        ],
                        dtype=float,
                    )
                )
                sphere_cloud["radius"] = np.asarray(
                    [site.free_radius for site, _translation in positive_instances],
                    dtype=float,
                )
                unit_sphere = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
                free_spheres = sphere_cloud.glyph(
                    scale="radius",
                    geom=unit_sphere,
                    orient=False,
                )
                self.add_mesh(
                    free_spheres,
                    color="#FFEE58",
                    opacity=0.22,
                    smooth_shading=True,
                    pickable=False,
                    name="interstitial-free-spheres",
                )

        if self.selected_interstitial_id is not None:
            selected_instances = [
                (site, translation)
                for site, translation in instances
                if site.site_id == self.selected_interstitial_id
            ]
            if selected_instances:
                selected_mesh = pv.merge(
                    [
                        self._site_polyhedron(site, translation)
                        for site, translation in selected_instances
                    ],
                    merge_points=False,
                )
                self.add_mesh(
                    selected_mesh,
                    color="#FFEB3B",
                    opacity=min(0.75, self.interstitial_opacity + 0.25),
                    show_edges=True,
                    edge_color="#FFFDE7",
                    line_width=3,
                    pickable=False,
                    name="selected-interstitial",
                )

    def _draw_bonds(self, bonds) -> None:
        assert self.document is not None
        if not {
            self.bond_center_element,
            self.bond_ligand_element,
        }.issubset(self.visible_elements):
            return
        structure = self.document.display
        if self._bond_mesh_cache_key != self._scene_cache_key:
            meshes_by_element: dict[str, list] = {}
            for bond in bonds:
                first_site = structure[bond.first.site_index]
                second_site = structure[bond.second.site_index]
                first_element = site_element(first_site)
                second_element = site_element(second_site)
                first = atom_cartesian_coords(structure, bond.first)
                second = atom_cartesian_coords(structure, bond.second)
                midpoint = (first + second) / 2.0
                for start, end, element in (
                    (first, midpoint, first_element),
                    (midpoint, second, second_element),
                ):
                    direction = end - start
                    length = float(np.linalg.norm(direction))
                    if length <= 1e-8:
                        continue
                    cylinder = pv.Cylinder(
                        center=(start + end) / 2.0,
                        direction=direction,
                        radius=self.bond_radius,
                        height=length,
                        resolution=8,
                    )
                    meshes_by_element.setdefault(element, []).append(cylinder)
            self._bond_mesh_cache = {
                element: pv.merge(meshes, merge_points=False)
                for element, meshes in meshes_by_element.items()
            }
            self._bond_mesh_cache_key = self._scene_cache_key

        for element, combined in self._bond_mesh_cache.items():
            self.add_mesh(
                combined,
                color=color_for(element),
                smooth_shading=True,
                pickable=False,
                name=f"bonds-{element}",
            )

    def _draw_cell(self) -> None:
        assert self.document is not None
        matrix = np.asarray(self.document.display.lattice.matrix)
        origin = np.zeros(3)
        corners = np.array([
            origin,
            matrix[0],
            matrix[1],
            matrix[2],
            matrix[0] + matrix[1],
            matrix[0] + matrix[2],
            matrix[1] + matrix[2],
            matrix[0] + matrix[1] + matrix[2],
        ])
        edges = [
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        ]
        lines = np.hstack([[2, start, end] for start, end in edges])
        cell = pv.PolyData(corners, lines=lines)
        self.add_mesh(
            cell,
            color="#F4F7FA",
            line_width=2,
            pickable=False,
            name="unit-cell",
        )

    def _configure_picker(self) -> None:
        try:
            self.disable_picking()
        except Exception:
            pass
        self.enable_point_picking(
            callback=self._picked_point,
            show_point=False,
            show_message=False,
            left_clicking=True,
            use_picker=True,
            picker="cell",
        )

    @staticmethod
    def _actor_key(actor) -> str:
        if actor is None:
            return ""
        if hasattr(actor, "memory_address"):
            return str(actor.memory_address)
        return str(actor.GetAddressAsString(""))

    def _picked_point(self, _point, picker) -> None:
        batch = self._atom_batches.get(self._actor_key(picker.GetActor()))
        cell_id = int(picker.GetCellId())
        dataset = picker.GetDataSet()
        if batch is None or dataset is None or cell_id < 0:
            return
        mesh = pv.wrap(dataset)
        cell = mesh.get_cell(cell_id)
        if not cell.point_ids or "atom_slot" not in mesh.point_data:
            return
        slot = int(mesh.point_data["atom_slot"][cell.point_ids[0]])
        if not 0 <= slot < len(batch):
            return
        atom = batch[slot]
        QTimer.singleShot(0, lambda selected=atom: self._select_atom(selected))

    def _select_atom(self, atom: RenderAtom) -> None:
        self.selected_atom = atom
        if self.on_atom_picked is not None:
            self.on_atom_picked(atom.site_index, atom.offset)
        self.render_structure(reset_camera=False)
