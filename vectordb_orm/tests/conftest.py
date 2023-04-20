import pytest
from pymilvus import Milvus, connections
from vectordb_orm import MilvusSession
from vectordb_orm.tests.models import MyObject, BinaryEmbeddingObject
from time import sleep

@pytest.fixture()
def milvus_client():
    return Milvus()

@pytest.fixture()
def session(milvus_client):
    session = MilvusSession(milvus_client)
    connections.connect("default", host="localhost", port="19530")
    return session

@pytest.fixture()
def collection(session: MilvusSession, milvus_client: Milvus):
    # Wipe the collection
    milvus_client.drop_collection(MyObject.collection_name())

    # Flush
    sleep(1)

    # Create a new default one
    return MyObject._create_collection(milvus_client)

@pytest.fixture()
def binary_collection(session: MilvusSession, milvus_client: Milvus):
    # Wipe the collection
    milvus_client.drop_collection(BinaryEmbeddingObject.collection_name())

    # Flush
    sleep(1)

    # Create a new default one
    return BinaryEmbeddingObject._create_collection(milvus_client)
