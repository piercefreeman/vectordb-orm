from os import getenv
from time import sleep

import pytest
from dotenv import load_dotenv
from pymilvus import Milvus, connections

from vectordb_orm import MilvusBackend, PineconeBackend, VectorSession
from vectordb_orm.tests.models import (MilvusBinaryEmbeddingObject,
                                       MilvusMyObject, PineconeMyObject)


@pytest.fixture()
def milvus_session():
    session = VectorSession(MilvusBackend(Milvus()))
    connections.connect("default", host="localhost", port="19530")

    # Wipe the previous collections
    session.delete_collection(MilvusMyObject)
    session.delete_collection(MilvusBinaryEmbeddingObject)

    # Flush
    sleep(1)

    session.create_collection(MilvusMyObject)
    session.create_collection(MilvusBinaryEmbeddingObject)

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
    session.create_collection(PineconeMyObject)
    session.clear_collection(PineconeMyObject)

    return session

SESSION_FIXTURE_KEYS = ["milvus_session", "pinecone_session"]
SESSION_MODEL_PAIRS = [
    (
        "milvus_session",
        MilvusMyObject,
    ),
    (
        "pinecone_session",
        PineconeMyObject,
    ),
]