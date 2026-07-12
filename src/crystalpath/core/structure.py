from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclass(frozen=True)
class StructureSummary:
    source: Path
    formula: str
    reduced_formula: str
    space_group: str
    space_group_number: int | None
    site_count: int
    is_ordered: bool
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float


class CrystalDocument:
    """Owns an immutable source structure and a replaceable display supercell."""

    def __init__(self, source: Path, structure: Structure, parser_warnings: list[str]):
        self.source = source
        self.original = structure.copy()
        self.display = structure.copy()
        self.supercell = (1, 1, 1)
        self.parser_warnings = parser_warnings

    @classmethod
    def from_cif(cls, path: str | Path) -> "CrystalDocument":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structure = Structure.from_file(source, primitive=False)
        return cls(source, structure, [str(item.message) for item in caught])

    def set_supercell(self, a: int, b: int, c: int) -> None:
        if min(a, b, c) < 1:
            raise ValueError("Supercell repetitions must be positive integers")
        self.display = self.original.copy()
        self.display.make_supercell((a, b, c))
        self.supercell = (a, b, c)

    def reset_supercell(self) -> None:
        self.set_supercell(1, 1, 1)

    def summary(self) -> StructureSummary:
        lattice = self.original.lattice
        symbol = "Unknown"
        number: int | None = None
        try:
            analyzer = SpacegroupAnalyzer(self.original, symprec=0.01)
            symbol = analyzer.get_space_group_symbol()
            number = analyzer.get_space_group_number()
        except Exception:
            pass
        return StructureSummary(
            source=self.source,
            formula=self.original.composition.formula,
            reduced_formula=self.original.composition.reduced_formula,
            space_group=symbol,
            space_group_number=number,
            site_count=len(self.original),
            is_ordered=self.original.is_ordered,
            a=lattice.a,
            b=lattice.b,
            c=lattice.c,
            alpha=lattice.alpha,
            beta=lattice.beta,
            gamma=lattice.gamma,
            volume=lattice.volume,
        )


def site_element(site) -> str:
    species, _occupancy = max(site.species.items(), key=lambda pair: pair[1])
    element = getattr(species, "element", species)
    return element.symbol


def site_occupancy(site) -> float:
    return float(sum(site.species.values()))

