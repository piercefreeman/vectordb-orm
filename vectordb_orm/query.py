from pymilvus import Milvus
from pymilvus.client.abstract import ChunkedQueryResult
from vectordb_orm.attributes import AttributeCompare
from typing import TYPE_CHECKING
from vectordb_orm.results import QueryResult

if TYPE_CHECKING:
    from vectordb_orm.base import MilvusBase


class MilvusQueryBuilder:
    """
    Recursive query builder to allow for chaining of queries.

    Example:
        >>> query().filter(MyObject.text == "hello").limit(10).all()

    """
    def __init__(self, milvus_client: Milvus, cls: "MilvusBase"):
        self.milvus_client = milvus_client
        self.cls = cls

        # Queries created over time
        self._filters: list[str] = []
        self._limit : int | None = None

        # Set if similarity ranking is used
        self._similarity_attribute : AttributeCompare | None = None
        self._similarity_value = None

    def filter(self, *filters: AttributeCompare):
        for f in filters:
            self._filters.append(f.to_expression())
        return self

    def limit(self, limit: int | None):
        self._limit = limit
        return self

    def order_by_similarity(self, embedding_attr: AttributeCompare, query_embedding):
        if self._similarity_attribute is not None:
            raise ValueError("Only one similarity ordering can be used per query.")
        self._similarity_attribute = embedding_attr
        self._similarity_value = query_embedding
        return self

    def all(self):
        filters = " and ".join(self._filters) if self._filters else None
        query_records = [self._similarity_value.values] if self._similarity_value else None
        embedding_field_name = self._similarity_attribute.attr if self._similarity_attribute else None

        params = {"nprobe": 16}

        search_result = self.milvus_client.search(
            data=query_records,
            anns_field=embedding_field_name,
            param=params,
            limit=self._limit,
            collection_name=self.cls.collection_name(),
            expression=filters,
            # TODO: Exclude embeddings by default
            output_fields=["id", "text"]
        )
        return self._result_to_objects(search_result)

    def _result_to_objects(self, search_result: ChunkedQueryResult):
        query_results : list[QueryResult] = []

        for hit in search_result:
            for result in hit:
                obj = self.cls(text=result.entity.get("text"), embedding=None)
                obj.id = result.id
                query_results.append(QueryResult(obj, result.score, result.distance))

        return query_results
