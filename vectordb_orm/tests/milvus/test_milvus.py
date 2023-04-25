import numpy as np
import pytest

from vectordb_orm import VectorSession
from vectordb_orm.tests.models import MilvusBinaryEmbeddingObject


# @pierce 04-21- 2023: Currently flaky
# https://github.com/piercefreeman/vectordb-orm/pull/5
@pytest.mark.xfail(strict=False)
def test_binary_collection_query(session: VectorSession):
    # Create some MyObject instances
    obj1 = MilvusBinaryEmbeddingObject(embedding=np.array([True] * 128))
    obj2 = MilvusBinaryEmbeddingObject(embedding=np.array([False] * 128))

    # Insert the objects into Milvus
    session.insert(obj1)
    session.insert(obj2)

    session.flush(MilvusBinaryEmbeddingObject)
    session.load(MilvusBinaryEmbeddingObject)  

    # Test our ability to recall 1:1 the input content
    results = session.query(MilvusBinaryEmbeddingObject).order_by_similarity(MilvusBinaryEmbeddingObject.embedding, np.array([True]*128)).limit(2).all()
    assert len(results) == 2
    assert results[0].result.id == obj1.id

    results = session.query(MilvusBinaryEmbeddingObject).order_by_similarity(BinaryEmbeddingObject.embedding, np.array([False]*128)).limit(2).all()
    assert len(results) == 2
    assert results[0].result.id == obj2.id
