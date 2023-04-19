import pytest
from pymilvus import Milvus
from vectordb_orm import MilvusSession, EmbeddingField, PrimaryKeyField, MilvusBase
import numpy as np
from vectordb_orm.indexes import IndexBase, FLOATING_INDEXES, BINARY_INDEXES
from vectordb_orm.similarity import FloatSimilarityMetric, BinarySimilarityMetric
from itertools import product

# Different index definitions require different kwarg arguments; we centralize
# them here for ease of accessing them during different test runs
INDEX_DEFAULTS = {
    "IVF_FLAT": dict(
        cluster_units=128,
    ),
    "IVF_SQ8": dict(
        cluster_units=128,
    ),
    "IVF_PQ": dict(
        cluster_units=128,
        product_quantization=16,
        low_dimension_bits=16,
    ),
    "HNSW": dict(
        max_degree=4,
        search_scope_index=16,
        search_scope_inference=128,
    ),
    "BIN_IVF_FLAT": dict(
        cluster_units=128,
    )
}

@pytest.mark.parametrize(
    "index_cls,metric_type",
    product(
        FLOATING_INDEXES,
        [item for item in FloatSimilarityMetric],
    )
)
def test_floating_index(
    milvus_client: Milvus,
    session: MilvusSession,
    index_cls: IndexBase,
    metric_type: FloatSimilarityMetric,
):
    class IndexSubclassObject(MilvusBase):
        __collection_name__ = 'index_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray = EmbeddingField(
            dim=128,
            index=index_cls(
                metric_type=metric_type,
                **INDEX_DEFAULTS.get(index_cls.index_type, {})
            )
        )

    # Drop the existing collection
    milvus_client.drop_collection(collection_name=IndexSubclassObject.collection_name())

    # Create the collection
    collection = IndexSubclassObject._create_collection(milvus_client)

    # Insert an object
    my_object = IndexSubclassObject(embedding=np.array([1.0] * 128))
    my_object.insert(milvus_client)

    # Flush and load the collection
    collection.flush()
    collection.load()

    # Perform a query
    results = session.query(IndexSubclassObject).order_by_similarity(IndexSubclassObject.embedding, np.array([1.0] * 128)).all()
    assert len(results) == 1
    assert results[0].result.id == my_object.id


@pytest.mark.parametrize(
    "index_cls,metric_type",
    product(
        BINARY_INDEXES,
        [item for item in BinarySimilarityMetric],
    )
)
def test_binary_index(
    milvus_client: Milvus,
    session: MilvusSession,
    index_cls: IndexBase,
    metric_type: BinarySimilarityMetric,
):
    class IndexSubclassObject(MilvusBase):
        __collection_name__ = 'index_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray[np.bool_] = EmbeddingField(
            dim=128,
            index=index_cls(
                metric_type=metric_type,
                **INDEX_DEFAULTS.get(index_cls.index_type, {})
            )
        )

    # Drop the existing collection
    milvus_client.drop_collection(collection_name=IndexSubclassObject.collection_name())

    # Create the collection
    collection = IndexSubclassObject._create_collection(milvus_client)

    # Insert an object
    my_object = IndexSubclassObject(embedding=np.array([True] * 128))
    my_object.insert(milvus_client)

    # Flush and load the collection
    collection.flush()
    collection.load()

    # Perform a query
    results = session.query(IndexSubclassObject).order_by_similarity(IndexSubclassObject.embedding, np.array([True] * 128)).all()
    assert len(results) == 1
    assert results[0].result.id == my_object.id
