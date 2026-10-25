"""Unit table (Civ6 ancient/classical stats)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitType:
    name: str
    code: str  # short label for rendering
    cs: int  # melee combat strength
    rs: int  # ranged strength (0 = melee unit)
    range: int
    mp: int
    cost: int  # production
    cls: str  # recon, melee, ranged, anti_cav, light_cav, heavy_cav, siege
    tech: str | None = None
    resource: str | None = None


UNIT_TYPES = [
    UnitType("scout", "Sc", 10, 0, 0, 3, 30, "recon"),
    UnitType("warrior", "W", 20, 0, 0, 2, 40, "melee"),
    UnitType("slinger", "Sl", 5, 15, 1, 2, 35, "ranged"),
    UnitType("archer", "A", 15, 25, 2, 2, 60, "ranged", tech="archery"),
    UnitType("spearman", "Sp", 25, 0, 0, 2, 65, "anti_cav", tech="bronze_working"),
    UnitType("heavy_chariot", "HC", 28, 0, 0, 2, 65, "heavy_cav", tech="the_wheel"),
    UnitType("horseman", "H", 36, 0, 0, 4, 80, "light_cav", tech="horseback_riding", resource="horses"),
    UnitType("swordsman", "Sw", 35, 0, 0, 2, 90, "melee", tech="iron_working", resource="iron"),
]
UNIT_INDEX = {u.name: i for i, u in enumerate(UNIT_TYPES)}
N_UNIT_TYPES = len(UNIT_TYPES)
CAVALRY = {"light_cav", "heavy_cav"}
