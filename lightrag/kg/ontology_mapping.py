"""Ontology mapping for docgraph ingestion.

Maps LightRAG's free-form entity_type and relation keywords to ontology IDs
from generic-knowledge-graph.json.
"""

import json
import os
from pathlib import Path

_DEFAULT_ONTOLOGY = Path(__file__).parent.parent / "data" / "generic-knowledge-graph.json"

ENTITY_TYPE_TO_ONTOLOGY = {
    "person": "onto:Person",
    "organization": "onto:Organization",
    "location": "onto:Location",
    "event": "onto:Event",
    "concept": "onto:Concept",
    "artifact": "onto:Artifact",
    "method": "onto:Method",
    "naturalobject": "onto:NaturalObject",
    "creature": "onto:Creature",
    "content": "onto:Content",
    "data": "onto:Data",
}

RELATION_TYPE_TO_ONTOLOGY = {
    "related_to": "onto:RELATED_TO",
    "works_at": "onto:WORKS_AT",
    "located_in": "onto:LOCATED_IN",
    "part_of": "onto:PART_OF",
    "created_by": "onto:CREATED_BY",
    "uses": "onto:USES",
    "knows": "onto:KNOWS",
    "participated_in": "onto:PARTICIPATED_IN",
    "occurred_at": "onto:OCCURRED_AT",
    "studies": "onto:STUDIES",
    "produces": "onto:PRODUCES",
    "measures": "onto:MEASURES",
    "derived_from": "onto:DERIVED_FROM",
    "impacts": "onto:IMPACTS",
    "supersedes": "onto:SUPERSEDES",
}


def map_entity_type(entity_type: str) -> str:
    """Map a free-form entity type string to an ontology concept ID.

    Returns onto:Concept as fallback for unknown types.
    """
    if not entity_type:
        return "onto:Concept"
    key = entity_type.lower().replace(" ", "").replace("_", "")
    return ENTITY_TYPE_TO_ONTOLOGY.get(key, "onto:Concept")


def map_relation_type(relation_type: str) -> str:
    """Map a relation type string to an ontology relation ID.

    Returns onto:RELATED_TO as fallback for unknown types.
    """
    if not relation_type:
        return "onto:RELATED_TO"
    key = relation_type.lower().replace(" ", "_").strip("_")
    return RELATION_TYPE_TO_ONTOLOGY.get(key, "onto:RELATED_TO")


def load_ontology(path: str | None = None) -> dict:
    """Load ontology JSON file. Returns the parsed ontology dict."""
    p = Path(path) if path else _DEFAULT_ONTOLOGY
    if not p.exists():
        raise FileNotFoundError(f"Ontology file not found: {p}")
    with open(p) as f:
        return json.load(f)["ontology"]
