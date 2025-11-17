#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import os
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from svlearn_vlu import config
from svlearn_vlu.utils.qdrant_setup import start_vector_db, PORT
from tqdm import tqdm


#  -------------------------------------------------------------------------------------------------

class CLIPEncoder:
    def __init__(self, config):
        """
        Initializes the CLIPEncoder with the CLIP model, processor, and Qdrant client.
        
        Args:
            config (dict): Configuration dictionary containing model, database, and dataset paths.
        """
        # Load CLIP model and processor
        self.model_name = config["clip"]["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

        # Connect to Qdrant
        start_vector_db()
        self.client = QdrantClient(url=f"http://localhost:{PORT}")

        # Initialize collection name
        self.collection_name = config["clip"]["collection_name"]

    # -------------------------------------------------------------------------------------------------

    def create_collection(self):
        """
        Creates a collection in Qdrant for storing image embeddings.
        If the collection already exists, it deletes the existing collection and recreates it.
        """
        # Check if the collection already exists
        existing_collections = self.client.get_collections()
        collection_names = [collection.name for collection in existing_collections.collections]
        
        if self.collection_name in collection_names:
            print(f"Collection '{self.collection_name}' already exists. Deleting it...")
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted.")
        
        # Create the collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"Collection '{self.collection_name}' created in Qdrant.")

    # -------------------------------------------------------------------------------------------------

    def encode_images(self, image_dir):
        """
        Encodes images from the specified directory and stores their embeddings in Qdrant.
        
        Args:
            image_dir (str): Path to the directory containing images.
        """
        # Prepare image paths
        image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) 
                       if fname.endswith(('.png', '.jpg', '.jpeg'))]

        # Store image embeddings in Qdrant
        for idx, img_path in enumerate(tqdm(image_paths, desc="Encoding images", unit="image")):
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
            
            # Generate image embedding
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs).cpu().numpy().flatten()

            # Store embedding in Qdrant with metadata
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": idx,  # Unique ID for the image
                        "vector": embedding.tolist(),
                        "payload": {"path": img_path},  # Metadata
                    }
                ]
            )

        print(f"Stored {len(image_paths)} images in Qdrant.")

    # -------------------------------------------------------------------------------------------------

    def run(self, image_dir):
        """
        Runs the entire pipeline: creates a collection and encodes images.
        
        Args:
            image_dir (str): Path to the directory containing images.
        """
        self.create_collection()
        self.encode_images(image_dir)


if __name__ == "__main__":
    # Initialize CLIPEncoder
    clip_encoder = CLIPEncoder(config)

    # Run the pipeline
    clip_encoder.run(config["datasets"]["trees"])