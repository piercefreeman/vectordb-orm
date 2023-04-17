from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectordb_orm.base import MilvusBase

@dataclass
class QueryResult:
    result: "MilvusBase"
    score: float
    distance: float
