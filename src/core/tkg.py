"""
Sparse Temporal Knowledge Graph (TKG) implementation.
Facts are represented as tuples (i, k, j, w) for entities i, j, relation k, intensity w.
Computation scales with |F_t| << N^2 M.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterator
import copy


@dataclass(frozen=True)
class Fact:
    source: str       # i
    relation: str     # k
    target: str       # j
    weight: float     # w

    def as_tuple(self) -> Tuple[str, str, str, float]:
        return (self.source, self.relation, self.target, self.weight)

    def triple(self) -> Tuple[str, str, str]:
        return (self.source, self.relation, self.target)


class SparseTKG:
    """Sparse weighted Temporal Knowledge Graph at a given time point."""

    def __init__(self, time_step: int = 0):
        self.time_step: int = time_step
        # Map (i, k, j) -> weight w for O(1) lookup
        self.facts: Dict[Tuple[str, str, str], float] = {}
        # Forward adjacency: source -> set of (relation, target)
        self.out_edges: Dict[str, Set[Tuple[str, str]]] = {}
        # Backward adjacency: target -> set of (source, relation)
        self.in_edges: Dict[str, Set[Tuple[str, str]]] = {}
        # Relation index: relation -> set of (source, target)
        self.rel_edges: Dict[str, Set[Tuple[str, str]]] = {}
        # Active entities
        self.entities: Set[str] = set()

    def add_fact(self, i: str, k: str, j: str, w: float) -> None:
        """Adds or updates a weighted fact (i, k, j, w)."""
        triple = (i, k, j)
        self.facts[triple] = float(w)
        self.entities.add(i)
        self.entities.add(j)

        if i not in self.out_edges:
            self.out_edges[i] = set()
        self.out_edges[i].add((k, j))

        if j not in self.in_edges:
            self.in_edges[j] = set()
        self.in_edges[j].add((i, k))

        if k not in self.rel_edges:
            self.rel_edges[k] = set()
        self.rel_edges[k].add((i, j))

    def remove_fact(self, i: str, k: str, j: str) -> bool:
        """Removes a fact triple if it exists."""
        triple = (i, k, j)
        if triple in self.facts:
            del self.facts[triple]
            self.out_edges[i].discard((k, j))
            self.in_edges[j].discard((i, k))
            self.rel_edges[k].discard((i, j))
            return True
        return False

    def get_weight(self, i: str, k: str, j: str, default: Optional[float] = None) -> Optional[float]:
        """Retrieves weight of fact (i, k, j)."""
        return self.facts.get((i, k, j), default)

    def has_triple(self, i: str, k: str, j: str) -> bool:
        """Checks existence of triple (i, k, j)."""
        return (i, k, j) in self.facts

    def get_out_neighbors(self, i: str, relation: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Returns list of (k, j, w) for facts originating from entity i."""
        res = []
        for k, j in self.out_edges.get(i, set()):
            if relation is None or k == relation:
                w = self.facts[(i, k, j)]
                res.append((k, j, w))
        return res

    def get_in_neighbors(self, j: str, relation: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Returns list of (i, k, w) for facts pointing to entity j."""
        res = []
        for i, k in self.in_edges.get(j, set()):
            if relation is None or k == relation:
                w = self.facts[(i, k, j)]
                res.append((i, k, w))
        return res

    def get_facts_by_relation(self, k: str) -> List[Tuple[str, str, float]]:
        """Returns list of (i, j, w) for relation k."""
        res = []
        for i, j in self.rel_edges.get(k, set()):
            res.append((i, j, self.facts[(i, k, j)]))
        return res

    def all_facts(self) -> List[Fact]:
        """Returns all facts as Fact dataclasses."""
        return [Fact(i, k, j, w) for (i, k, j), w in self.facts.items()]

    def num_facts(self) -> int:
        return len(self.facts)

    def clone(self, new_time_step: Optional[int] = None) -> "SparseTKG":
        """Deep copy of the sparse graph."""
        t = self.time_step if new_time_step is None else new_time_step
        new_g = SparseTKG(time_step=t)
        new_g.facts = dict(self.facts)
        new_g.out_edges = {k: set(v) for k, v in self.out_edges.items()}
        new_g.in_edges = {k: set(v) for k, v in self.in_edges.items()}
        new_g.rel_edges = {k: set(v) for k, v in self.rel_edges.items()}
        new_g.entities = set(self.entities)
        return new_g

    def to_dict(self) -> Dict:
        """Serializes graph to dict format for API and frontend."""
        return {
            "time_step": self.time_step,
            "entities": sorted(list(self.entities)),
            "facts": [
                {"source": i, "relation": k, "target": j, "weight": round(w, 4)}
                for (i, k, j), w in self.facts.items()
            ],
            "num_facts": len(self.facts),
        }
