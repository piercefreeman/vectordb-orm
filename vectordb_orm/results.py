from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectordb_orm.base import VectorSchemaBase

@dataclass
class QueryResult:
    result: "VectorSchemaBase"

    # Score and distance is only returned for queries requesting vector-similarity
    score: float | None = None
    distance: float | None = None
