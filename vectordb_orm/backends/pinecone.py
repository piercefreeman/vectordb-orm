import pinecone
from vectordb_orm.backends.base import BackendBase

class PineconeBackend(BackendBase):
    def __init__(self, pinecone_api_key: str, pinecone_namespace: str):
        pinecone.init(
            api_key=pinecone_api_key,
            environment="us-west1-gcp",
        )
        self.namespace = pinecone_namespace

    def create_collection(self, collection_name: str):
        # Pinecone allows for dynamic keys on each object
        # However we need to pre-provide the keys we want to search on
        metadata_config = {
            "indexed": ["color"]
        }

        pinecone.create_index(index_name=collection_name, metric="euclidean", metadata_config=metadata_config)
        return pinecone.Index(index_name=collection_name)

    def delete_collection(self, collection_name: str):
        pinecone.delete_index(collection_name)

    def insert(self, collection_name: str, entities: dict) -> list:
        primary_keys = entities.pop("id")
        vectors = entities.pop("embedding")
        self.client.upsert(items=zip(primary_keys, vectors))
        return primary_keys

    def delete(self, collection_name: str, expr: str):
        primary_key = int(expr.split("[")[1].split("]")[0])
        self.client.delete(ids=[primary_key])

    def search(
        self,
        collection_name: str,
        query_vectors,
        filters: dict,
        top_k: int,
    ):
        return self.client.fetch(ids=query_vectors, max_results=search_params["top_k"])
