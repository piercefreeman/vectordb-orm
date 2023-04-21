from os import getenv
from time import sleep

import pytest
from dotenv import load_dotenv
from pymilvus import Milvus, connections

from vectordb_orm import MilvusBackend, PineconeBackend, VectorSession
from vectordb_orm.tests.models import BinaryEmbeddingObject, MyObject


@pytest.fixture()
def milvus_session():
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

@pytest.fixture()
def pinecone_session():
    load_dotenv()
    session = VectorSession(
        PineconeBackend(
            api_key=getenv("PINECONE_API_KEY"),
            environment=getenv("PINECONE_ENVIRONMENT"),
        )
    )

    # Wipe the previous collections
    # Pinecone doesn't have the notion of binary objects like Milvus does, so we
    # only create one object. Their free tier also doesn't support more than 1
    # collection, so that's another limitation that encourages a single index here.
    session.create_collection(MyObject)
    session.clear_collection(MyObject)

    return session

SESSION_FIXTURE_KEYS = ["milvus_session", "pinecone_session"]
