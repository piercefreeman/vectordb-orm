import pytest
from pymilvus import Milvus, connections
from vectordb_orm import VectorSession, MilvusBackend
from vectordb_orm.tests.models import MyObject, BinaryEmbeddingObject
from time import sleep


@pytest.fixture()
def session():
    session = VectorSession(MilvusBackend(Milvus()))
    connections.connect("default", host="localhost", port="19530")

    # Wipe the previous collections
    session.delete_collection(MyObject)
    session.delete_collection(BinaryEmbeddingObject)

    # Flush
    sleep(1)

    session.create_collection(MyObject)
    session.create_collection(BinaryEmbeddingObject)

    return session
