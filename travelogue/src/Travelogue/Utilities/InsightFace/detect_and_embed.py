#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2025.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from insightface.app import FaceAnalysis
import uuid
from qdrant_client import QdrantClient
import numpy as np
import os
import cv2
from svlearn_vlu import config
import platform
import torch
from svlearn_vlu.face_detection.collection_create import create_collection

def get_ctx_id() -> int:
    if "darwin" in platform.system().lower():
        return 0  # allow CoreML provider to be used
    return 0 if torch.cuda.is_available() else -1

qdrant = QdrantClient(host=config["vector-db"]["host"], port=config["vector-db"]["port"])

# Initialize once (outside the function, e.g., global or singleton)
app = FaceAnalysis(name="buffalo_l")   # includes RetinaFace detector + ArcFace embedder
print(f"Using ctx_id: {get_ctx_id()}")
app.prepare(ctx_id=get_ctx_id())                  # ctx_id=0 for CPU on Mac; set to -1 to fully disable GPU

def get_face_embeddings(image_path: str) -> list[dict]:
    """Get face embeddings from an image using InsightFace (TensorFlow-free).

    Args:
        image_path (str): Path to the image to get face embeddings from.

    Returns:
        list[dict]: List of face data (bbox, landmarks, embedding).
    """
    # Load image explicitly to avoid passing path strings
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    faces = app.get(img)
    results = []
    
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return results
    
    for i, face in enumerate(faces):
        # Check if face has the required attributes
        if face.bbox is None or face.embedding is None:
            print(f"Skipping face {i} in {image_path}: missing bbox or embedding")
            continue

        try:
            results.append({
                "facial_area": face.bbox.tolist(),           # [x1, y1, x2, y2]
                "embedding": face.embedding.tolist(),        # 512-D ArcFace vector
                "det_score": float(face.det_score),          # detection confidence
                "landmarks": face.landmark.tolist() if face.landmark is not None else []
            })
        except Exception as e:
            print(f"Error processing face {i} in {image_path}: {e}")
            continue
    
    return results

def find_or_insert_face(
    qdrant: QdrantClient,
    embedding: list[float],
    bbox: list[float],
    image_path: str,
    faces_dir: str,
    threshold: float = 0.4,
    min_size: int = 160
) -> str:
    """
    Search for a close match; if none, insert new and save reference face.

    Args:
        qdrant (QdrantClient): Qdrant client.
        embedding (list[float]): Embedding to search for.
        bbox (list[float]): Bounding box of the face.
        image_path (str): Path to the image.
        faces_dir (str): Path to the directory to save the face.
        threshold (float, optional): Threshold for matching. Defaults to 0.4.
        min_size (int, optional): Minimum size of the face. Defaults to 160.
    Returns:
        str: Person ID if found, otherwise new ID.
    """
    search_results = qdrant.search(
        collection_name=config["deepface"]["collection_name"],
        query_vector=embedding,
        limit=1
    )

    # --- Load and crop the face ---
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        print(f"Invalid crop for {image_path}, skipping")
        return "skipped"

    h, w = face_crop.shape[:2]
    if h < min_size or w < min_size:
        print(f"Skipping small face ({w}x{h}) in {image_path} (<{min_size}px)")
        return "too_small"

    # --- If match found, skip saving new ---
    if search_results and search_results[0].score > threshold:
        person_id = search_results[0].payload.get("person_id")
        print(f"Match found: {person_id} (score={search_results[0].score:.3f})")
        return person_id

    # --- Otherwise, save + insert new identity ---
    new_id = str(uuid.uuid4())
    os.makedirs(faces_dir, exist_ok=True)
    save_path = os.path.join(faces_dir, f"{new_id}.jpg")
    cv2.imwrite(save_path, face_crop)

    qdrant.upsert(
        collection_name=config["deepface"]["collection_name"],
        points=[
            {
                "id": new_id,
                "vector": embedding,
                "payload": {"person_id": new_id, "image_path": save_path}
            }
        ]
    )
    print(f"Added new face: {new_id} ({w}x{h}) → saved at {save_path}")
    return new_id

def process_image(image_path: str) -> None:
    """Process an image to get face embeddings and insert them into Qdrant.

    Args:
        image_path (str): Path to the image to process.

    Returns:
        None
    """
    faces = get_face_embeddings(image_path)
    
    if len(faces) == 0:
        print(f"No faces found in {image_path}, skipping")
        return
    
    faces_dir = config["datasets"]["reference_faces_dir"]

    for face in faces:
        embedding = np.array(face["embedding"], dtype=np.float32)
        threshold = config["deepface"]["threshold"]
        bbox = face["facial_area"]
        person_id = find_or_insert_face(
            qdrant=qdrant,
            embedding=embedding.tolist(),
            bbox=bbox,
            image_path=image_path,
            faces_dir=faces_dir,
            threshold=threshold,
            min_size=90
        )
        print(f"Processed face → ID: {person_id}")

if __name__ == "__main__":
    create_collection(recreate=False)
    ## process images from a folder
    image_folder = config["datasets"]["faces"]
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for item in os.listdir(image_folder):
        full_path = os.path.join(image_folder, item)
        # Only process files (not directories) with image extensions
        if os.path.isfile(full_path) and os.path.splitext(item.lower())[1] in image_extensions:
            print(f"Processing: {item}")
            try:
                process_image(full_path)
            except Exception as e:
                print(f"Error processing {item}: {e}")
