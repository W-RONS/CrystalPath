ELEMENT_COLORS = {
    "H": "#F5F5F5",
    "Li": "#CC80FF",
    "C": "#444444",
    "N": "#3050F8",
    "O": "#E53935",
    "F": "#90E050",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Nb": "#73C2C9",
}

ELEMENT_RADII = {
    "H": 0.20,
    "Li": 0.45,
    "C": 0.30,
    "N": 0.30,
    "O": 0.30,
    "F": 0.28,
    "Na": 0.50,
    "Mg": 0.45,
    "Al": 0.45,
    "Si": 0.40,
    "P": 0.40,
    "S": 0.38,
    "Cl": 0.38,
    "K": 0.58,
    "Ca": 0.55,
    "Ti": 0.48,
    "V": 0.46,
    "Cr": 0.46,
    "Mn": 0.46,
    "Fe": 0.44,
    "Co": 0.43,
    "Ni": 0.43,
    "Cu": 0.44,
    "Zn": 0.45,
    "Nb": 0.50,
}


def color_for(element: str) -> str:
    return ELEMENT_COLORS.get(element, "#D0D0D0")


def radius_for(element: str) -> float:
    return ELEMENT_RADII.get(element, 0.70)
