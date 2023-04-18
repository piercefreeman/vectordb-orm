import pytest
from pymilvus import Milvus, connections
from vectordb_orm import MilvusSession
from vectordb_orm.tests.models import MyObject
import numpy as np

milvus_client = Milvus()
session = MilvusSession(milvus_client)
connections.connect("default", host="localhost", port="19530")

@pytest.fixture()
def drop_collection():
    # Wipe the collection
    milvus_client.drop_collection(MyObject.collection_name())


def test_query(drop_collection):
    """
    General test of querying and query chaining
    """
    collection = MyObject._create_collection(milvus_client)

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


def test_query_default_ignores_embeddings(drop_collection):
    """
    Ensure that querying on the class by default ignores embeddings that are included
    within the type definition.
    """
    collection = MyObject._create_collection(milvus_client)

    obj1 = MyObject(text="foo", embedding=np.array([1.0] * 128))
    obj1.insert(milvus_client)

    collection.flush()
    collection.load()  

    # Test a simple filter and similarity ranking
    results = session.query(MyObject).filter(MyObject.text == 'foo').limit(2).all()
    assert len(results) == 1

    result : MyObject = results[0].result
    assert result.embedding is None


def test_query_with_fields(drop_collection):
    """
    Test querying with specific fields
    """
    collection = MyObject._create_collection(milvus_client)

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
