#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import torch
from qdrant_client import QdrantClient
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from svlearn_vlu import config
import matplotlib.pyplot as plt


# configure model and processor
model_name = config["clip"]["model"]
port = config["vector-db"]["port"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# configure vector database
client = QdrantClient(url=f"http://localhost:{port}")
collection_name = config["clip"]["collection_name"]

#  -------------------------------------------------------------------------------------------------

def search_similar_images(query: str, top_k=5):
    """
    Search images in the vector db collection.
    Args:
        query (str): search query as text.
        top_k (int): Number of search results to return.

    """
    # Encode text query
    inputs = processor(text=query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy().flatten()
        text_embedding = text_embedding / (text_embedding ** 2).sum() ** 0.5  # Normalize

    # Search in Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=text_embedding.tolist(),
        limit=top_k,
    )
    return results

#  -------------------------------------------------------------------------------------------------

def show_images_as_subplots(results):
    """
    Displays images in a grid of subplots.

    Args:
        image_paths (list): List of file paths to the images.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
    """
    ncols = len(results)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))
    axes = axes.flatten()  # Flatten to make iteration easier
    
    for i, ax in enumerate(axes):
        result = results[i]
        img = Image.open(result.payload['path'])
        ax.imshow(img)
        ax.axis('off')  # Turn off axis


    plt.tight_layout()
    plt.show()

#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    query = "a photo of a lake"
    results = search_similar_images(query, top_k=2)

    # Display Results
    for result in results:
        print(f"Score: {result.score}, Path: {result.payload['path']}")
        img = Image.open(result.payload['path'])
        img.show()