import pytest
from pymilvus import Milvus, connections
from vectordb_orm import MilvusSession
from vectordb_orm.tests.models import MyObject
import numpy as np

milvus_client = Milvus()
session = MilvusSession(milvus_client)
connections.connect("default", host="localhost", port="19530")


def test_query():
    # Wipe the collection
    milvus_client.drop_collection(MyObject.collection_name())

    collection = MyObject._create_collection(milvus_client)

    # Create some MyObject instances
    obj1 = MyObject('foo', np.array([1.0] * 128))
    obj2 = MyObject('bar', np.array([4.0] * 128))
    obj3 = MyObject('baz', np.array([7.0] * 128))

    # Insert the objects into Milvus
    obj1.insert(milvus_client)
    obj2.insert(milvus_client)
    obj3.insert(milvus_client)

    collection.flush()
    collection.load()  

    # Test a simple filter and similarity ranking
    print("result", session.query(MyObject).filter(MyObject.text == 'bar').order_by_similarity(MyObject.embedding, np.array([1.0]*128)).limit(2))
    results = session.query(MyObject).filter(MyObject.text == 'bar').order_by_similarity(MyObject.embedding, np.array([1.0]*128)).limit(2).all()
    assert len(results) == 1
    assert results[0].result.id == obj2.id

    # Test a more complex filter and similarity ranking
    results = session.query(MyObject).filter(MyObject.text == 'baz', MyObject.id > 1).order_by_similarity(MyObject.embedding, np.array([8.0]*128)).limit(2).all()
    assert len(results) == 1
    assert results[0].result.id == obj3.id

    # Clean up
    milvus_client.drop_collection(MyObject.collection_name())
