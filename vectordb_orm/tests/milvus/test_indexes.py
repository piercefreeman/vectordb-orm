from itertools import product

import numpy as np
import pytest
from pymilvus import Milvus

from vectordb_orm import (EmbeddingField, PrimaryKeyField, VectorSchemaBase,
                          VectorSession)
from vectordb_orm.backends.milvus.indexes import (BINARY_INDEXES,
                                                  FLOATING_INDEXES,
                                                  MilvusIndexBase)
from vectordb_orm.backends.milvus.similarity import (
    MilvusBinarySimilarityMetric, MilvusFloatSimilarityMetric)

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
        [item for item in MilvusFloatSimilarityMetric],
    )
)
def test_floating_index(
    milvus_session: VectorSession,
    index_cls: MilvusIndexBase,
    metric_type: MilvusFloatSimilarityMetric,
):
    class IndexSubclassObject(VectorSchemaBase):
        __collection_name__ = 'index_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray = EmbeddingField(
            dim=128,
            index=index_cls(
                metric_type=metric_type,
                **INDEX_DEFAULTS.get(index_cls.index_type, {})
            )
        )

    milvus_session.delete_collection(IndexSubclassObject)
    milvus_session.create_collection(IndexSubclassObject)

    # Insert an object
    my_object = IndexSubclassObject(embedding=np.array([1.0] * 128))
    milvus_session.insert(my_object)

    # Flush and load the collection
    milvus_session.flush(IndexSubclassObject)
    milvus_session.load(IndexSubclassObject)

    # Perform a query
    results = milvus_session.query(IndexSubclassObject).order_by_similarity(IndexSubclassObject.embedding, np.array([1.0] * 128)).all()
    assert len(results) == 1
    assert results[0].result.id == my_object.id


@pytest.mark.parametrize(
    "index_cls,metric_type",
    product(
        BINARY_INDEXES,
        [item for item in MilvusBinarySimilarityMetric],
    )
)
def test_binary_index(
    milvus_session: VectorSession,
    index_cls: MilvusIndexBase,
    metric_type: MilvusBinarySimilarityMetric,
):
    class IndexSubclassObject(VectorSchemaBase):
        __collection_name__ = 'index_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray[np.bool_] = EmbeddingField(
            dim=128,
            index=index_cls(
                metric_type=metric_type,
                **INDEX_DEFAULTS.get(index_cls.index_type, {})
            )
        )

    milvus_session.delete_collection(IndexSubclassObject)
    milvus_session.create_collection(IndexSubclassObject)

    # Insert an object
    my_object = IndexSubclassObject(embedding=np.array([True] * 128))
    milvus_session.insert(my_object)

    # Flush and load the collection
    milvus_session.flush(IndexSubclassObject)
    milvus_session.load(IndexSubclassObject)

    # Perform a query
    results = milvus_session.query(IndexSubclassObject).order_by_similarity(IndexSubclassObject.embedding, np.array([True] * 128)).all()
    assert len(results) == 1
    assert results[0].result.id == my_object.id
