"""
Family reference flow (Streamlit):
- Upload family photos
- Detect and crop faces with InsightFace
- Collect metadata per face (name, relation, notes, date)
- Persist originals, crops, and metadata under the configured family data root
- Upsert face embeddings with payload into Qdrant
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import hashlib

from Utilities.InsightFace.detect_and_embed import get_face_embeddings
from Utilities.qdrant_setup import load_config, start_vector_db


config = load_config()


# ---------- Paths ----------
configured_family_root = config.get("paths", {}).get("family_data")
FAMILY_ROOT = (
    Path(configured_family_root).expanduser()
    if configured_family_root
    else Path("/home/anuragd/labshare/travelogue/family_photos")
)
ORIGINALS_DIR = FAMILY_ROOT / "originals"
CROPS_DIR = FAMILY_ROOT / "crops"
METADATA_DIR = FAMILY_ROOT / "metadata"
for _dir in (ORIGINALS_DIR, CROPS_DIR, METADATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def compute_file_hash(file_bytes: bytes) -> str:
    """Deterministic hash to avoid reprocessing the same upload on reruns."""
    return hashlib.md5(file_bytes).hexdigest()


def get_exif_date(image_path: Path):
    """Return a date object from EXIF DateTimeOriginal if present."""
    try:
        img = Image.open(image_path)
        exifdata = img.getexif()
        if not exifdata:
            return None
        date_str = exifdata.get(36867) or exifdata.get(36868) or exifdata.get(306)
        if not date_str:
            return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(str(date_str), fmt).date()
            except ValueError:
                continue
        return None
    except Exception:
        return None


def save_uploaded_file(uploaded_file, file_hash: str) -> Path:
    """Persist uploaded file to originals dir with a stable name (hash based)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    dest = ORIGINALS_DIR / f"{file_hash}_{Path(uploaded_file.name).stem}{suffix}"
    image = Image.open(uploaded_file)
    image.save(dest)
    return dest


def crop_and_save(image_path: Path, bbox: List[float], face_id: str) -> Path:
    """Crop face using bbox [x1, y1, x2, y2] and save to crops dir."""
    img = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = map(int, bbox)
    crop = img.crop((x1, y1, x2, y2))
    crop_name = f"{image_path.stem}_face_{face_id}.jpg"
    crop_path = CROPS_DIR / crop_name
    crop.save(crop_path)
    return crop_path


def ensure_collection(client: QdrantClient, collection: str, vector_size: int):
    """Create Qdrant collection if missing."""
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_face(
    client: QdrantClient,
    collection: str,
    point_id: str,
    face_vector: List[float],
    payload: Dict,
):
    """Upsert single face embedding with payload."""
    client.upsert(
        collection_name=collection,
        points=[
            {
                "id": point_id,
                "vector": face_vector,
                "payload": payload,
            }
        ],
    )


# ---------- Streamlit ----------
st.set_page_config(page_title="Family Reference", layout="wide")
st.title("Family Reference Builder")
st.caption(
    "Upload family photos, label faces, and index embeddings + metadata in Qdrant. "
    f"Files are stored under {FAMILY_ROOT}."
)

# Session state setup
state_defaults = {
    "uploaded_paths": [],
    "faces_by_image": {},  # {image_path_str: [face dicts]}
    "metadata_by_face": {},  # {face_id: metadata}
    "processed_hashes": set(),  # to prevent duplicate processing on rerun
}
for key, default_val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


qdrant_host = config.get("vector-db", {}).get("host", "localhost")
qdrant_port = config.get("vector-db", {}).get("port", 6333)
collection_name = "family_faces"


st.subheader("1) Upload family photos")
uploads = st.file_uploader(
    "Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploads:
    new_images = 0
    for uf in uploads:
        file_bytes = uf.getvalue()
        file_hash = compute_file_hash(file_bytes)
        if file_hash in st.session_state.processed_hashes:
            continue  # already processed this upload in a prior rerun

        # reset pointer for PIL after reading bytes
        uf.seek(0)

        saved_path = save_uploaded_file(uf, file_hash)
        st.session_state.uploaded_paths.append(str(saved_path))

        # Detect faces immediately
        detections = get_face_embeddings(str(saved_path))
        faces = []
        for idx, det in enumerate(detections):
            face_id = f"{saved_path.stem}_{idx}_{uuid.uuid4().hex}"
            qdrant_point_id = str(uuid.uuid4())
            crop_path = crop_and_save(saved_path, det["facial_area"], face_id)
            faces.append(
                {
                    "face_id": face_id,
                    "point_id": qdrant_point_id,
                    "bbox": det["facial_area"],
                    "det_score": det["det_score"],
                    "embedding": det["embedding"],
                    "crop_path": str(crop_path),
                    "source_image": str(saved_path),
                    "original_filename": uf.name,
                    "captured_at": str(get_exif_date(saved_path) or ""),
                }
            )
        st.session_state.faces_by_image[str(saved_path)] = faces
        st.session_state.processed_hashes.add(file_hash)
        new_images += 1

    if new_images:
        st.success(f"Saved {new_images} new image(s) and detected faces.")
    else:
        st.info("No new images to process (duplicates skipped).")

st.markdown("---")
st.subheader("2) Review faces and enter metadata")

all_faces: List[Dict] = []
for faces in st.session_state.faces_by_image.values():
    all_faces.extend(faces)

if not all_faces:
    st.info("Upload a photo to begin. Faces will show up here for labeling.")
else:
    cols = st.columns(3)
    for idx, face in enumerate(all_faces):
        with cols[idx % 3]:
            st.image(face["crop_path"], caption=f"Score {face['det_score']:.2f}", use_container_width=True)
            name = st.text_input(
                "Name", key=f"name_{face['face_id']}", value=st.session_state.metadata_by_face.get(face["face_id"], {}).get("person_name", "")
            )
            relation = st.text_input(
                "Relation (e.g., wife, son)", key=f"rel_{face['face_id']}",
                value=st.session_state.metadata_by_face.get(face["face_id"], {}).get("relation", "")
            )
            date_val = st.date_input(
                "Date (optional)",
                key=f"date_{face['face_id']}",
                value=datetime.strptime(face["captured_at"], "%Y-%m-%d").date() if face.get("captured_at") else datetime.utcnow().date(),
            )
            notes = st.text_area(
                "Notes",
                key=f"notes_{face['face_id']}",
                value=st.session_state.metadata_by_face.get(face["face_id"], {}).get("notes", ""),
                height=60,
            )
            st.session_state.metadata_by_face[face["face_id"]] = {
                "person_name": name.strip(),
                "relation": relation.strip(),
                "notes": notes.strip(),
                "captured_at": str(date_val),
            }

st.markdown("---")
st.subheader("3) Persist to disk + Qdrant")

if st.button("Save metadata and index faces", type="primary"):
    try:
        # Health check Qdrant before connecting
        if not start_vector_db():
            st.error("Qdrant is not reachable. Please start the container.")
            st.stop()

        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        vector_size = len(all_faces[0]["embedding"]) if all_faces else 0
        if vector_size == 0:
            st.error("No embeddings found.")
        else:
            ensure_collection(client, collection_name, vector_size)

            indexed = 0
            for face in all_faces:
                meta = st.session_state.metadata_by_face.get(face["face_id"], {})
                payload = {
                    "face_id": face["face_id"],
                    "point_id": face["point_id"],
                    "person_name": meta.get("person_name", ""),
                    "relation": meta.get("relation", ""),
                    "notes": meta.get("notes", ""),
                    "captured_at": meta.get("captured_at", face.get("captured_at", "")),
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "source_image_path": face["source_image"],
                    "crop_path": face["crop_path"],
                    "bbox": face["bbox"],
                    "det_score": face["det_score"],
                    "original_filename": face["original_filename"],
                }

                # Persist per-face metadata to disk
                meta_path = METADATA_DIR / f"{face['face_id']}.json"
                with open(meta_path, "w") as f:
                    json.dump(payload, f, indent=2)

                upsert_face(client, collection_name, face["point_id"], face["embedding"], payload)
                indexed += 1

            st.success(f"Indexed {indexed} face(s) into Qdrant and saved metadata.")
    except Exception as e:
        st.error(f"Failed to save/index: {e}")

st.caption(f"Storage: originals -> {ORIGINALS_DIR} | crops -> {CROPS_DIR} | metadata -> {METADATA_DIR}")
