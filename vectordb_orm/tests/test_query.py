import pytest
from pymilvus import Milvus, connections
from vectordb_orm import MilvusSession
from vectordb_orm.tests.models import MyObject, BinaryEmbeddingObject
import numpy as np
from time import sleep

def test_query(collection, milvus_client: Milvus, session: MilvusSession):
    """
    General test of querying and query chaining
    """
    # Create some MyObject instances
    obj1 = MyObject(text="foo", embedding=np.array([1.0] * 128))
    obj2 = MyObject(text="bar", embedding=np.array([4.0] * 128))
    obj3 = MyObject(text="baz", embedding=np.array([7.0] * 128))

    # Insert the objects into Milvus
    obj1.insert(milvus_client)
    obj2.insert(milvus_client)
    obj3.insert(milvus_client)

    collection.flush()
    collection.load()  

    # Test a simple filter and similarity ranking
    results = session.query(MyObject).filter(MyObject.text == 'bar').order_by_similarity(MyObject.embedding, np.array([1.0]*128)).limit(2).all()
    assert len(results) == 1
    assert results[0].result.id == obj2.id

    # Test combined filtering
    results = session.query(MyObject).filter(MyObject.text == 'baz', MyObject.id > 1).order_by_similarity(MyObject.embedding, np.array([8.0]*128)).limit(2).all()
    assert len(results) == 1
    assert results[0].result.id == obj3.id

def test_binary_collection_query(binary_collection, milvus_client: Milvus, session: MilvusSession):
    # Create some MyObject instances
    obj1 = BinaryEmbeddingObject(embedding=np.array([True] * 128))
    obj2 = BinaryEmbeddingObject(embedding=np.array([False] * 128))

    # Insert the objects into Milvus
    obj1.insert(milvus_client)
    obj2.insert(milvus_client)

    binary_collection.flush()
    binary_collection.load()  

    # Test our ability to recall 1:1 the input content
    results = session.query(BinaryEmbeddingObject).order_by_similarity(BinaryEmbeddingObject.embedding, np.array([True]*128)).limit(2).all()
    assert len(results) == 2
    assert results[0].result.id == obj1.id

    results = session.query(BinaryEmbeddingObject).order_by_similarity(BinaryEmbeddingObject.embedding, np.array([False]*128)).limit(2).all()
    assert len(results) == 2
    assert results[0].result.id == obj2.id

def test_query_default_ignores_embeddings(collection, milvus_client: Milvus, session: MilvusSession):
    """
    Ensure that querying on the class by default ignores embeddings that are included
    within the type definition.
    """
    obj1 = MyObject(text="foo", embedding=np.array([1.0] * 128))
    obj1.insert(milvus_client)

    collection.flush()
    collection.load()  

    # Test a simple filter and similarity ranking
    results = session.query(MyObject).filter(MyObject.text == 'foo').limit(2).all()
    assert len(results) == 1

    result : MyObject = results[0].result
    assert result.embedding is None


def test_query_with_fields(collection, milvus_client: Milvus, session: MilvusSession):
    """
    Test querying with specific fields
    """
    obj1 = MyObject(text="foo", embedding=np.array([1.0] * 128))
    obj1.insert(milvus_client)

    collection.flush()
    collection.load()  

    # We can query for regular attributes
    results = session.query(MyObject.text).filter(MyObject.text == 'foo').all()
    assert len(results) == 1

    # We shouldn't be able to query for embeddings
    with pytest.raises(ValueError):
        session.query(MyObject.embedding).filter(MyObject.text == 'foo').all()
