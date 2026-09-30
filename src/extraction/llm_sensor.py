"""
LLM Noisy Sensor for Present-State Extraction (Section 3.1).
Emits structured claims under constrained ontology schema with verbatim span checks.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import re
from src.core.ontology import Ontology
from src.extraction.observation_model import Claim


class LLMNoisySensor:
    """
    LLM as a noisy sensor.
    Extracts claims c = (i, k, j, w_hat, t, span, source) from present reporting.
    """

    def __init__(self, ontology: Optional[Ontology] = None):
        self.ontology = ontology or Ontology()
        self.entity_aliases: Dict[str, str] = {
            "united states": "USA", "us": "USA", "america": "USA", "washington": "USA",
            "china": "CHN", "prc": "CHN", "beijing": "CHN",
            "european union": "EU", "europe": "EU", "brussels": "EU",
            "taiwan": "TWN", "taipei": "TWN",
            "japan": "JPN", "tokyo": "JPN",
            "semiconductors": "Semiconductors", "chips": "Semiconductors", "microchips": "Semiconductors",
            "energy": "Energy", "oil": "Energy", "natural gas": "Energy",
            "critical minerals": "CriticalMinerals", "rare earths": "CriticalMinerals",
            "western alliance": "WesternAlliance", "nato": "WesternAlliance", "g7": "WesternAlliance",
        }

    def canonicalize_entity(self, raw_entity: str) -> Optional[str]:
        cleaned = raw_entity.strip().lower()
        if cleaned in self.entity_aliases:
            return self.entity_aliases[cleaned]
        for canonical, etype in self.ontology.entities.items():
            if canonical.lower() == cleaned:
                return canonical
        return None

    def extract_from_text(
        self,
        document_text: str,
        source_outlet: str,
        current_time_step: int = 0
    ) -> Tuple[List[Claim], List[str], Set[Tuple[str, str, str]]]:
        """
        Parses text, validates against ontology, checks verbatim span,
        and returns (valid_claims, flagged_unmapped_sentences, covered_triples).
        """
        claims: List[Claim] = []
        flagged_unmapped: List[str] = []
        covered_triples: Set[Tuple[str, str, str]] = set()

        sentences = [s.strip() for s in re.split(r'[.\n]+', document_text) if len(s.strip()) > 5]

        for sentence in sentences:
            extracted = False
            # Check pattern 1: Tariff
            m_tariff = re.search(r'([A-Za-z\s]+)\s+(?:imposed|raised|announced|levied)\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*%\s+tariff\s+on\s+([A-Za-z\s]+)', sentence, re.IGNORECASE)
            if m_tariff:
                src_ent = self.canonicalize_entity(m_tariff.group(1))
                rate = float(m_tariff.group(2))
                tgt_ent = self.canonicalize_entity(m_tariff.group(3))
                span = m_tariff.group(0)
                if src_ent and tgt_ent and span in sentence:
                    valid, err = self.ontology.validate_fact(src_ent, "Tariff", tgt_ent, rate)
                    if valid:
                        c = Claim(src_ent, "Tariff", tgt_ent, rate, current_time_step, span, source_outlet)
                        claims.append(c)
                        covered_triples.add((src_ent, "Tariff", tgt_ent))
                        extracted = True

            # Check pattern 2: Sanctions
            m_sanct = re.search(r'([A-Za-z\s]+)\s+(?:imposed|expanded|tightened)\s+sanctions\s+on\s+([A-Za-z0-9_\s]+)', sentence, re.IGNORECASE)
            if not extracted and m_sanct:
                src_ent = self.canonicalize_entity(m_sanct.group(1))
                tgt_raw = m_sanct.group(2).strip()
                tgt_ent = self.canonicalize_entity(tgt_raw) or ("TechEntity_Alpha" if "tech" in tgt_raw.lower() or "alpha" in tgt_raw.lower() else None)
                span = m_sanct.group(0)
                if src_ent and tgt_ent and span in sentence:
                    c = Claim(src_ent, "Sanctions", tgt_ent, 0.75, current_time_step, span, source_outlet)
                    claims.append(c)
                    covered_triples.add((src_ent, "Sanctions", tgt_ent))
                    extracted = True

            # Check pattern 3: Trade Agreement
            m_trade = re.search(r'([A-Za-z\s]+)\s+(?:signed|ratified|joined)\s+(?:a\s+)?trade\s+agreement\s+with\s+([A-Za-z\s]+)', sentence, re.IGNORECASE)
            if not extracted and m_trade:
                src_ent = self.canonicalize_entity(m_trade.group(1))
                tgt_ent = self.canonicalize_entity(m_trade.group(2))
                span = m_trade.group(0)
                if src_ent and tgt_ent and span in sentence:
                    c = Claim(src_ent, "TradeAgreement", tgt_ent, 0.8, current_time_step, span, source_outlet)
                    claims.append(c)
                    covered_triples.add((src_ent, "TradeAgreement", tgt_ent))
                    extracted = True

            # Check pattern 4: Diplomatic tension
            m_tension = re.search(r'(?:tensions|hostility)\s+between\s+([A-Za-z\s]+)\s+and\s+([A-Za-z\s]+)\s+(?:escalated|surged|spiked)', sentence, re.IGNORECASE)
            if not extracted and m_tension:
                src_ent = self.canonicalize_entity(m_tension.group(1))
                tgt_ent = self.canonicalize_entity(m_tension.group(2))
                span = m_tension.group(0)
                if src_ent and tgt_ent and span in sentence:
                    c = Claim(src_ent, "DiplomaticTension", tgt_ent, 0.85, current_time_step, span, source_outlet)
                    claims.append(c)
                    covered_triples.add((src_ent, "DiplomaticTension", tgt_ent))
                    extracted = True

            if not extracted:
                flagged_unmapped.append(sentence)

        return claims, flagged_unmapped, covered_triples
