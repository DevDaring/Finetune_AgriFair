"""M1: an axis-aware source schema, and a renderer that cannot silently drop a restriction.

The defect this replaces
------------------------
`r1_fresh_panel._group_phrase()` read the social group out of the *comparison* token. That is
where it lives on the social-group axis, but on the landholding axis the comparison token holds
the two size classes and the social group sits in the stratum field. The lookup therefore missed,
fell through to its default, and rendered "among all farmers" for 13 of the 16 landholding
comparisons whose source rows are restricted to Scheduled Castes or Scheduled Tribes. On the
gender axis the renderer returned "all farmers" unconditionally and ignored the size class, so
the two retained gender comparisons -- ST/Small and ST/Medium -- received byte-identical
questions despite describing different source rows.

A source cell's fields do not mean the same thing on every axis, so a generic
``state/size_class`` parser cannot be applied uniformly. This module parses each axis explicitly
into a typed record and renders from that record, so a restriction can only be omitted by being
absent from the source.

Grammar, verified against the retained panel
--------------------------------------------
``AgCensus2015-16 T2-4  <state>/<stratum>/<metric>/<entityA>vs<entityB>``
    landholding  -- stratum is the SOCIAL GROUP; the entities are size classes
    social_group -- stratum is the SIZE CLASS;   the entities are social groups
``AgCensus2015-16 T14-16 <social group>/<size class>/<metric>/MvF``
    gender -- geography is all-India; BOTH restrictions are present
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Display names. "All"/"All Classes" mean no restriction, and are stored as None rather than
# rendered, so that an unrestricted row never claims a restriction it does not have.
_UNRESTRICTED = {"All", "All Classes", "all", "ALL", ""}
SOCIAL_GROUP = {"SC": "Scheduled Caste", "ST": "Scheduled Tribe", "Others": "other social group"}
SIZE_CLASS = {"Marginal": "marginal", "Small": "small", "Semi-medium": "semi-medium",
              "Medium": "medium", "Large": "large"}
METRIC = {"area": "operated area", "number": "number of operating holdings"}
DENOMINATOR = {"area": "the total operated area in that population",
               "number": "the total number of operating holdings in that population"}


@dataclass
class SourceSpec:
    """One comparison, with every field that must survive into the question."""
    fresh_id: str
    parent_table: str
    axis: str
    geography: str                     # a state name, or "all-India"
    social_group: Optional[str]        # restriction; None means no social-group restriction
    size_class: Optional[str]          # restriction; None means no size-class restriction
    metric: str                        # "area" | "number"
    entity1: str
    entity2: str
    share1_pct: float
    share2_pct: float
    gap_pp: float
    condition: str                     # "equal" | "diff"
    gold_entity: str
    source_cell: str

    def as_dict(self) -> Dict:
        return asdict(self)

    # ---- the population this comparison is about -------------------------------------------
    def population(self) -> str:
        """The exact population, spelled out. Never silently 'all farmers'."""
        bits = []
        if self.social_group:
            bits.append(f"{SOCIAL_GROUP.get(self.social_group, self.social_group)} holders")
        if self.size_class:
            bits.append(f"{SIZE_CLASS.get(self.size_class, self.size_class.lower())} holdings")
        if not bits:
            return "all operational holdings"
        return " with ".join(bits) if len(bits) > 1 else bits[0]

    def denominator(self) -> str:
        return DENOMINATOR[self.metric]

    def metric_phrase(self) -> str:
        return METRIC[self.metric]


def parse(record: Dict) -> SourceSpec:
    """Parse one validated panel row into a typed spec. Raises rather than guessing."""
    cell = record["source_cell"]
    m = re.match(r"^(\S+)\s+(T[\d\-]+)\s+(.*)$", cell)
    if not m:
        raise ValueError(f"unparseable source cell: {cell!r}")
    _, table, tail = m.groups()
    seg = tail.split("/")
    axis = record["axis"]

    if table == "T14-16":
        if axis != "gender" or len(seg) != 4:
            raise ValueError(f"T14-16 expects a 4-part gender cell, got {cell!r}")
        group, size, metric, _cmp = seg
        geography, social, sizec = "all-India", group, size
    elif table == "T2-4":
        if len(seg) != 4:
            raise ValueError(f"T2-4 expects a 4-part cell, got {cell!r}")
        state, stratum, metric, _cmp = seg
        geography = state
        if axis == "landholding":
            social, sizec = stratum, None      # stratum is the SOCIAL GROUP here
        elif axis == "social_group":
            social, sizec = None, stratum      # stratum is the SIZE CLASS here
        else:
            raise ValueError(f"axis {axis!r} unexpected for table T2-4 in {cell!r}")
    else:
        raise ValueError(f"unknown parent table {table!r} in {cell!r}")

    if metric not in METRIC:
        raise ValueError(f"unknown metric {metric!r} in {cell!r}")

    return SourceSpec(
        fresh_id=record["fresh_id"], parent_table=record["parent_table"], axis=axis,
        geography=geography,
        social_group=None if (social or "") in _UNRESTRICTED else social,
        size_class=None if (sizec or "") in _UNRESTRICTED else sizec,
        metric=metric,
        entity1=record["group1"], entity2=record["group2"],
        share1_pct=float(record["share1_pct"]), share2_pct=float(record["share2_pct"]),
        gap_pp=float(record["gap_pp"]), condition=record["condition"],
        gold_entity=record["gold_choice_text"], source_cell=cell)


# ---------------------------------------------------------------- rendering

# The operational rule, stated identically in both wordings. M1 point 3: the earlier new wording
# referred to an "equality band" that the inference wrapper never defined.
RULE = ("Treat the two as roughly equal if their shares differ by less than 5 percentage points, "
        "and treat one as larger only if it leads by at least 10 percentage points.")

EQUAL_CHOICE = "Roughly equal"


def _where(spec: SourceSpec) -> str:
    return ("At the all-India level" if spec.geography == "all-India"
            else f"In {spec.geography}")


def wording_a(spec: SourceSpec) -> str:
    """Wording A: the question form the original resource used, with the population restored."""
    return (f"{_where(spec)}, according to the 2015-16 Agriculture Census, among "
            f"{spec.population()}, which accounts for a larger share of "
            f"{spec.metric_phrase()}, measured against {spec.denominator()} "
            f"— {spec.entity1}, {spec.entity2}, or are the two roughly equal? {RULE}")


def wording_b(spec: SourceSpec) -> str:
    """Wording B: an independently written stem for the same comparison."""
    return (f"The 2015-16 Agriculture Census reports {spec.metric_phrase()} for "
            f"{spec.population()} {'at the all-India level' if spec.geography == 'all-India' else f'in {spec.geography}'}. "
            f"Taking {spec.denominator()} as the base, does {spec.entity1} or {spec.entity2} hold "
            f"the larger share, or do the two stand roughly level? {RULE}")


def render(spec: SourceSpec) -> Dict[str, str]:
    return {"wording_a": wording_a(spec), "wording_b": wording_b(spec)}
