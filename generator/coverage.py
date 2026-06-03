from dataclasses import dataclass, field

from certfe.dsl.family import FeatureFamily


@dataclass
class CoverageEntry:
    family: FeatureFamily
    col_group: frozenset[int]  # set of column indices
    n_accepted: int = 0
    n_rejected: int = 0
    last_round: int = -1


@dataclass
class CoverageMap:
    entries: dict[tuple[str, frozenset[int]], CoverageEntry] = field(default_factory=dict)
    current_round: int = 0

    def _key(self, family: FeatureFamily, col_group: frozenset[int]) -> tuple:
        return (family.value, col_group)

    def record_accepted(self, family: FeatureFamily, col_group: frozenset[int]):
        k = self._key(family, col_group)
        if k not in self.entries:
            self.entries[k] = CoverageEntry(family=family, col_group=col_group)
        self.entries[k].n_accepted += 1
        self.entries[k].last_round = self.current_round

    def record_rejected(self, family: FeatureFamily, col_group: frozenset[int]):
        k = self._key(family, col_group)
        if k not in self.entries:
            self.entries[k] = CoverageEntry(family=family, col_group=col_group)
        self.entries[k].n_rejected += 1
        self.entries[k].last_round = self.current_round

    def is_covered(self, family: FeatureFamily, col_group: frozenset[int]) -> bool:
        k = self._key(family, col_group)
        return k in self.entries and self.entries[k].n_accepted > 0

    def undercovered_families(self, all_cols: set[int]) -> list[dict]:
        """Return families and column groups with low or no coverage, for prompt injection."""
        result: list[dict] = []
        # Report families that have not been explored at all
        from certfe.dsl.family import IID_FAMILIES
        seen_families = {e.family for e in self.entries.values() if e.n_accepted + e.n_rejected > 0}
        for fam in IID_FAMILIES:
            if fam not in seen_families:
                result.append({
                    "family": fam.value,
                    "status": "unexplored",
                    "suggested_cols": sorted(all_cols),
                })
        # Report explored but under-covered
        for (fam_val, cg), entry in self.entries.items():
            total = entry.n_accepted + entry.n_rejected
            if total > 0 and entry.n_accepted == 0:
                result.append({
                    "family": fam_val,
                    "status": "all_rejected",
                    "col_group": sorted(cg),
                    "n_rejected": entry.n_rejected,
                })
        return result
