from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError, Voronoi

from crystalpath.core.structure import site_element


# Independent from the visual sphere radii used by the renderer.
REFERENCE_ANALYSIS_RADII = {
    "H": 0.31, "Li": 1.28, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "K": 2.03, "Ca": 1.76,
    "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Nb": 1.64,
}


@dataclass(frozen=True)
class InterstitialNeighbor:
    site_index: int
    offset: tuple[int, int, int]
    element: str
    distance: float
    cart_coords: tuple[float, float, float]


@dataclass(frozen=True)
class InterstitialSite:
    site_id: int
    kind: str
    frac_coords: tuple[float, float, float]
    cart_coords: tuple[float, float, float]
    free_radius: float
    coordination_number: int
    distortion: float
    neighbors: tuple[InterstitialNeighbor, ...]
    polyhedron_points: tuple[tuple[float, float, float], ...]
    polyhedron_faces: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class InterstitialResult:
    sites: tuple[InterstitialSite, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        preferred = ["四面体型", "八面体型", "立方型"]
        present = {site.kind for site in self.sites}
        ordered = [kind for kind in preferred if kind in present]
        ordered.extend(sorted(present - set(ordered)))
        return tuple(ordered)

    def sites_of_kind(self, kind: str) -> tuple[InterstitialSite, ...]:
        return tuple(site for site in self.sites if site.kind == kind)


def fractional_position_category(
    frac_coords: tuple[float, float, float],
    tolerance: float = 1e-6,
) -> str:
    """Classify a periodic point by its position in the displayed unit cell."""

    wrapped = np.mod(np.asarray(frac_coords, dtype=float), 1.0)
    boundary_axes = int(
        np.count_nonzero(
            np.isclose(wrapped, 0.0, atol=tolerance)
            | np.isclose(wrapped, 1.0, atol=tolerance)
        )
    )
    return {
        0: "晶胞内部",
        1: "晶胞面上",
        2: "晶胞棱上",
        3: "晶胞角点",
    }[boundary_axes]


@dataclass(frozen=True)
class _InterstitialCandidate:
    """A periodic candidate point and the Voronoi feature that generated it."""

    frac_coords: tuple[float, float, float]
    source: str


class InterstitialAnalyzer:
    def __init__(
        self,
        merge_tolerance: float = 1e-4,
        shell_tolerance: float = 0.04,
        minimum_free_radius: float = 0.01,
        prototype_tolerance: float = 0.30,
    ):
        self.merge_tolerance = merge_tolerance
        self.shell_tolerance = shell_tolerance
        self.minimum_free_radius = minimum_free_radius
        self.prototype_tolerance = prototype_tolerance

    def analyze(
        self,
        structure,
        obstacle_elements: set[str] | None = None,
        radii: dict[str, float] | None = None,
    ) -> InterstitialResult:
        if obstacle_elements is None:
            obstacle_elements = {site_element(site) for site in structure}
        radii = {**REFERENCE_ANALYSIS_RADII, **(radii or {})}
        obstacle_indices = [
            index
            for index, site in enumerate(structure)
            if site_element(site) in obstacle_elements
        ]
        if not obstacle_indices:
            raise ValueError("间隙分析至少需要1个障碍原子位点")

        image_points, image_metadata = self._periodic_points(structure, obstacle_indices)
        if np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 3:
            raise ValueError("周期展开后的障碍原子没有形成三维结构")
        voronoi = Voronoi(image_points, qhull_options="Qbb Qc Qz")
        inv_lattice = np.linalg.inv(np.asarray(structure.lattice.matrix))
        candidates = self._voronoi_candidates(
            voronoi,
            inv_lattice,
            np.asarray(structure.lattice.matrix),
        )

        sites: list[InterstitialSite] = []
        for candidate in candidates:
            frac = np.asarray(candidate.frac_coords)
            center = frac @ np.asarray(structure.lattice.matrix)
            site = self._make_site(
                len(sites) + 1,
                frac,
                center,
                image_points,
                image_metadata,
                radii,
                candidate.source,
            )
            if site is not None and site.free_radius >= self.minimum_free_radius:
                sites.append(site)

        sites.sort(key=lambda item: (item.kind, item.frac_coords))
        sites = [
            InterstitialSite(
                site_id=index + 1,
                kind=site.kind,
                frac_coords=site.frac_coords,
                cart_coords=site.cart_coords,
                free_radius=site.free_radius,
                coordination_number=site.coordination_number,
                distortion=site.distortion,
                neighbors=site.neighbors,
                polyhedron_points=site.polyhedron_points,
                polyhedron_faces=site.polyhedron_faces,
            )
            for index, site in enumerate(sites)
        ]
        return InterstitialResult(tuple(sites))

    def _voronoi_candidates(self, voronoi, inv_lattice, lattice_matrix):
        """Generate candidates from vertices, face centers and ridge centers.

        Voronoi vertices are local maxima of the atom-distance field and find
        sites such as the BCC tetrahedral voids.  A distorted interstitial can
        instead lie on a lower-dimensional Voronoi feature: the BCC octahedral
        site is the center of a face shared by its two closest atoms.  Finite
        face centers and face-edge centers are therefore considered as
        additional candidates, but later require a recognized closed
        coordination polyhedron to be accepted.
        """

        candidates: list[_InterstitialCandidate] = []
        smallest_lattice_scale = max(
            float(np.min(np.linalg.svd(lattice_matrix, compute_uv=False))),
            1e-12,
        )
        fractional_tolerance = self.merge_tolerance / smallest_lattice_scale
        bin_count = max(1, int(np.floor(1.0 / max(fractional_tolerance, 1e-12))))
        buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
        for vertex in voronoi.vertices:
            self._add_candidate(
                candidates,
                buckets,
                bin_count,
                vertex,
                "vertex",
                inv_lattice,
                lattice_matrix,
            )

        for ridge_vertices in voronoi.ridge_vertices:
            if len(ridge_vertices) < 3 or -1 in ridge_vertices:
                continue
            face_points = np.asarray(voronoi.vertices[ridge_vertices], dtype=float)
            self._add_candidate(
                candidates,
                buckets,
                bin_count,
                face_points.mean(axis=0),
                "face",
                inv_lattice,
                lattice_matrix,
            )
            for first, second in self._polygon_edges(face_points):
                self._add_candidate(
                    candidates,
                    buckets,
                    bin_count,
                    (first + second) / 2.0,
                    "ridge",
                    inv_lattice,
                    lattice_matrix,
                )
        return candidates

    def _add_candidate(
        self,
        candidates,
        buckets,
        bin_count,
        cart_coords,
        source,
        inv_lattice,
        lattice_matrix,
    ):
        frac = np.asarray(cart_coords) @ inv_lattice
        if not (np.all(frac >= -1e-8) and np.all(frac < 1.0 - 1e-8)):
            return
        wrapped = np.mod(frac, 1.0)
        wrapped[np.isclose(wrapped, 0.0, atol=1e-8)] = 0.0
        wrapped[np.isclose(wrapped, 1.0, atol=1e-8)] = 0.0
        key = tuple((np.floor(wrapped * bin_count).astype(int) % bin_count).tolist())
        for delta in product((-1, 0, 1), repeat=3):
            neighbor_key = tuple(
                (key[axis] + delta[axis]) % bin_count for axis in range(3)
            )
            if self._contains_fractional(
                buckets.get(neighbor_key, ()),
                wrapped,
                lattice_matrix,
            ):
                return
        candidates.append(
            _InterstitialCandidate(
                tuple(float(value) for value in wrapped),
                source,
            )
        )
        buckets.setdefault(key, []).append(wrapped)

    @staticmethod
    def _polygon_edges(points):
        """Return the perimeter edges of a planar 3-D Voronoi face."""

        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
        try:
            outline = ConvexHull(projected)
        except QhullError:
            return ()
        order = outline.vertices
        return tuple(
            (points[order[index]], points[order[(index + 1) % len(order)]])
            for index in range(len(order))
        )

    @staticmethod
    def _periodic_points(structure, obstacle_indices):
        points: list[np.ndarray] = []
        metadata: list[tuple[int, tuple[int, int, int], str]] = []
        matrix = np.asarray(structure.lattice.matrix)
        for offset in product((-1, 0, 1), repeat=3):
            offset_array = np.asarray(offset)
            for index in obstacle_indices:
                site = structure[index]
                points.append((site.frac_coords + offset_array) @ matrix)
                metadata.append((index, tuple(offset), site_element(site)))
        return np.asarray(points, dtype=float), metadata

    def _contains_fractional(self, candidates, frac, lattice_matrix) -> bool:
        for existing in candidates:
            delta = frac - existing
            delta -= np.round(delta)
            if np.linalg.norm(delta @ lattice_matrix) <= self.merge_tolerance:
                return True
        return False

    def _make_site(
        self,
        site_id,
        frac,
        center,
        image_points,
        image_metadata,
        radii,
        candidate_source,
    ):
        distances = np.linalg.norm(image_points - center, axis=1)
        order = np.argsort(distances)
        nearest_distance = float(distances[order[0]])
        shell_width = max(self.shell_tolerance, nearest_distance * 0.015)
        first_shell = [
            int(index)
            for index in order
            if distances[index] <= nearest_distance + shell_width
        ]

        surface_clearances = [
            float(distances[index]) - radii.get(image_metadata[index][2], 0.90)
            for index in range(len(image_points))
        ]
        free_radius = min(surface_clearances)
        if free_radius < self.minimum_free_radius:
            return None
        geometry = self._best_coordination_geometry(
            center,
            distances,
            order,
            image_points,
        )
        if geometry is None and candidate_source == "vertex" and len(first_shell) >= 4:
            geometry = self._fallback_geometry(center, first_shell, image_points)
        if geometry is None:
            return None

        shell_indices, faces, kind, distortion = geometry
        # Face/ridge centers are plentiful generic sampling points.  Retain
        # them only when they reconstruct a recognized closed polyhedron;
        # Voronoi vertices remain useful even for low-symmetry CN environments.
        if candidate_source != "vertex" and kind not in {
            "四面体型",
            "八面体型",
            "立方型",
        }:
            return None

        neighbors = []
        for image_index in shell_indices:
            original_index, offset, element = image_metadata[image_index]
            neighbors.append(
                InterstitialNeighbor(
                    site_index=original_index,
                    offset=offset,
                    element=element,
                    distance=float(distances[image_index]),
                    cart_coords=tuple(float(value) for value in image_points[image_index]),
                )
            )
        points = np.asarray([neighbor.cart_coords for neighbor in neighbors], dtype=float)
        return InterstitialSite(
            site_id=site_id,
            kind=kind,
            frac_coords=tuple(float(value) for value in frac),
            cart_coords=tuple(float(value) for value in center),
            free_radius=float(free_radius),
            coordination_number=len(neighbors),
            distortion=float(distortion),
            neighbors=tuple(neighbors),
            polyhedron_points=tuple(tuple(float(value) for value in point) for point in points),
            polyhedron_faces=faces,
        )

    def _best_coordination_geometry(self, center, distances, order, image_points):
        options = []
        for coordination_number in (4, 6, 8):
            if len(order) < coordination_number:
                continue
            boundary_distance = float(distances[order[coordination_number - 1]])
            if len(order) > coordination_number:
                next_distance = float(distances[order[coordination_number]])
                tie_width = max(self.shell_tolerance, boundary_distance * 0.015)
                if next_distance <= boundary_distance + tie_width:
                    # Do not split an equal-distance shell merely to obtain a
                    # desired coordination number.
                    continue
            indices = [int(value) for value in order[:coordination_number]]
            points = np.asarray(image_points[indices], dtype=float)
            hull = self._closed_hull(points, center)
            if hull is None:
                continue
            faces = tuple(tuple(int(value) for value in face) for face in hull.simplices)
            kind, score = self._classify(
                points - center,
                coordination_number,
                len(faces),
            )
            options.append((score, indices, faces, kind))

        if not options:
            return None
        score, indices, faces, kind = min(options, key=lambda option: option[0])
        if score > self.prototype_tolerance:
            return None
        return indices, faces, kind, score

    def _fallback_geometry(self, center, indices, image_points):
        points = np.asarray(image_points[indices], dtype=float)
        hull = self._closed_hull(points, center)
        if hull is None:
            return None
        faces = tuple(tuple(int(value) for value in face) for face in hull.simplices)
        kind, score = self._classify(points - center, len(indices), len(faces))
        return list(indices), faces, kind, score

    @staticmethod
    def _closed_hull(points, center):
        try:
            hull = ConvexHull(points, qhull_options="Qt")
        except QhullError:
            return None
        if len(hull.vertices) != len(points):
            return None
        signed_distances = hull.equations[:, :3] @ center + hull.equations[:, 3]
        if float(np.max(signed_distances)) >= -1e-7:
            return None
        return hull

    @staticmethod
    def _classify(vectors, coordination_number, face_count):
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors / norms[:, None]
        dots = np.array([
            float(np.dot(unit[i], unit[j]))
            for i in range(len(unit))
            for j in range(i + 1, len(unit))
        ])
        radial_distortion = float(np.std(norms) / max(np.mean(norms), 1e-12))

        if coordination_number == 4 and face_count == 4:
            angular = float(np.sqrt(np.mean((dots + 1.0 / 3.0) ** 2)))
            score = angular + radial_distortion
            return ("四面体型" if score <= 0.28 else "畸变CN4型", score)

        if coordination_number == 6 and face_count == 8:
            expected = np.array([-1.0] * 3 + [0.0] * 12)
            angular = float(np.sqrt(np.mean((np.sort(dots) - expected) ** 2)))
            score = angular + radial_distortion
            return ("八面体型" if score <= 0.28 else "畸变CN6型", score)

        if coordination_number == 8:
            expected = np.array([-1.0] * 4 + [-1.0 / 3.0] * 12 + [1.0 / 3.0] * 12)
            angular = float(np.sqrt(np.mean((np.sort(dots) - expected) ** 2)))
            score = angular + radial_distortion
            return ("立方型" if score <= 0.28 else "畸变CN8型", score)
        return f"其他CN{coordination_number}型", radial_distortion


def supercell_translations(lattice_matrix, repeats: tuple[int, int, int]):
    """Return Cartesian translations for every original cell in a supercell."""

    lattice = np.asarray(lattice_matrix, dtype=float)
    repeat_a, repeat_b, repeat_c = repeats
    return tuple(
        tuple(float(value) for value in (ia * lattice[0] + ib * lattice[1] + ic * lattice[2]))
        for ia in range(repeat_a)
        for ib in range(repeat_b)
        for ic in range(repeat_c)
    )
