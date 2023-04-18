from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectordb_orm.base import MilvusBase

@dataclass
class QueryResult:
    result: "MilvusBase"

    # Score and distance is only returned for queries requesting vector-similarity
    score: float | None = None
    distance: float | None = None
