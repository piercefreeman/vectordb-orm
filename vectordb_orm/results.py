from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vectordb_orm.base import VectorSchemaBase

@dataclass
class QueryResult:
    result: "VectorSchemaBase"

    # Score and distance is only returned for queries requesting vector-similarity
    score: Optional[float] = None
    distance: Optional[float] = None
