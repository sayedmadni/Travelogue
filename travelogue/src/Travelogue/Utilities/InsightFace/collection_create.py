#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2025.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from svlearn_vlu import config

def create_collection(recreate=False):
    """Create a collection in Qdrant if it doesn't exist, or recreate it if it does.

    Args:
        recreate (bool, optional): Recreate the collection if it exists. Defaults to False.
    """
    collection_name = config["deepface"]["collection_name"]
    qdrant = QdrantClient(host=config["vector-db"]["host"], port=config["vector-db"]["port"])
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
    elif recreate:
        qdrant.delete_collection(collection_name=collection_name)
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
    else:
        print(f"Collection {collection_name} already exists")

if __name__ == "__main__":
    create_collection(recreate=False)