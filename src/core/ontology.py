"""
Ontology definitions for the Temporal Knowledge Graph (TKG).
Constrains relation domains, ranges, and continuous intensity values/units.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class RelationSchema:
    name: str
    domain_type: str
    range_type: str
    unit: str
    min_intensity: float
    max_intensity: float
    default_intensity: float
    description: str


class Ontology:
    """Ontology schema defining legal entities, relations, and units."""

    def __init__(self):
        self.entity_types: Set[str] = {
            "State", "Sector", "Entity", "Alliance", "Institution"
        }
        self.relations: Dict[str, RelationSchema] = {
            "Tariff": RelationSchema(
                name="Tariff",
                domain_type="State",
                range_type="Sector",
                unit="percentage",
                min_intensity=0.0,
                max_intensity=100.0,
                default_intensity=10.0,
                description="Bilateral tariff rate imposed on trade in a sector."
            ),
            "Sanctions": RelationSchema(
                name="Sanctions",
                domain_type="State",
                range_type="Entity",
                unit="severity_index",
                min_intensity=0.0,
                max_intensity=1.0,
                default_intensity=0.5,
                description="Economic or trade sanctions severity (0 to 1)."
            ),
            "TradeAgreement": RelationSchema(
                name="TradeAgreement",
                domain_type="State",
                range_type="State",
                unit="integration_level",
                min_intensity=0.0,
                max_intensity=1.0,
                default_intensity=0.7,
                description="Degree of trade integration or preferential access."
            ),
            "AllianceCohesion": RelationSchema(
                name="AllianceCohesion",
                domain_type="State",
                range_type="Alliance",
                unit="cohesion_index",
                min_intensity=0.0,
                max_intensity=1.0,
                default_intensity=0.8,
                description="Security or diplomatic alliance commitment strength."
            ),
            "TechExportControl": RelationSchema(
                name="TechExportControl",
                domain_type="State",
                range_type="Entity",
                unit="restriction_index",
                min_intensity=0.0,
                max_intensity=1.0,
                default_intensity=0.6,
                description="Technology transfer and critical export controls."
            ),
            "DiplomaticTension": RelationSchema(
                name="DiplomaticTension",
                domain_type="State",
                range_type="State",
                unit="tension_index",
                min_intensity=0.0,
                max_intensity=1.0,
                default_intensity=0.3,
                description="Degree of diplomatic hostility or tension."
            ),
        }
        self.entities: Dict[str, str] = {
            "USA": "State",
            "CHN": "State",
            "EU": "State",
            "TWN": "State",
            "JPN": "State",
            "KOR": "State",
            "Semiconductors": "Sector",
            "Energy": "Sector",
            "Automotive": "Sector",
            "CriticalMinerals": "Sector",
            "WesternAlliance": "Alliance",
            "RegionalPact": "Alliance",
            "TechEntity_Alpha": "Entity",
            "TechEntity_Beta": "Entity",
        }

    def validate_fact(self, i: str, k: str, j: str, w: float) -> Tuple[bool, Optional[str]]:
        """Validates a proposed fact against the ontology."""
        if k not in self.relations:
            return False, f"Unknown relation: {k}"
        rel = self.relations[k]
        if i not in self.entities:
            return False, f"Unknown entity: {i}"
        if j not in self.entities:
            return False, f"Unknown entity: {j}"
        if not (rel.min_intensity <= w <= rel.max_intensity):
            return False, f"Intensity {w} outside allowed range [{rel.min_intensity}, {rel.max_intensity}] for {k}"
        return True, None

    def add_entity(self, name: str, entity_type: str) -> None:
        """Admits a new entity after formal review."""
        if entity_type not in self.entity_types:
            raise ValueError(f"Invalid entity type: {entity_type}")
        self.entities[name] = entity_type

    def add_relation(self, schema: RelationSchema) -> None:
        """Admits a new relation schema after formal review."""
        self.relations[schema.name] = schema
